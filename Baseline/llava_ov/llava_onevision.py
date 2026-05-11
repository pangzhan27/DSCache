import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import sys
sys.path.append('./Baseline/llava_ov')

import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
import os
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"
from packaging import version
from tqdm import tqdm
import ffmpeg
from transformers import AutoConfig
from . import utils

eval_logger = logging.getLogger("lmms-eval")

try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )

except ImportError as e:
    raise ValueError(f"LLaVA is not installed. Please install LLaVA to use this model.")

# previously this is from llava, now we change to own defined so that we can include modified MLLM (KV cache w/o PE)
from builder import load_pretrained_model, Instance

class Llava_OneVision():
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = "sdpa",
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
    ) -> None:
        self._device = torch.device(device)
        self.device_map = device_map
        llava_model_args = { "multimodal": True }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation

        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:

            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name,  device_map=self.device_map, **llava_model_args) #torch_dtype="float32",
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name,  device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        # self.model.to(self._device)
        self._rank = 0
        self._world_size = 1


    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        uniform_sampled_frames = np.unique(uniform_sampled_frames) ## added by PANG
        frame_idx = uniform_sampled_frames.tolist()
        try:
            spare_frames = vr.get_batch(frame_idx).asnumpy()
        except Exception as e:
            print(f"Failed to load {video_path}: {e}", flush=True)

        return spare_frames  # (frames, height, width, channels)

    def load_video_span(self, video_path, max_frames_num, start, end, fps=None):
        try:
            if type(video_path) == str:
                vr = VideoReader(video_path, ctx=cpu(0))
            else:
                vr = VideoReader(video_path[0], ctx=cpu(0))
        except Exception as e:
            print(f"Failed to load {video_path}: {e}", flush=True)

        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        start_f = min(max(0, int(start * orig_fps)), total_frame_num-1)
        end_f = min(total_frame_num-1, int(end * orig_fps))

        if end_f <= start_f:
            end_f = min(start_f + 1, total_frame_num-1)

        if fps is None: # use max frame
            frame_indices = np.arange(start_f, end_f+1)
            if len(frame_indices) > max_frames_num:
                frame_indices = np.linspace(start_f, end_f, max_frames_num).astype(int)
            unique_indices = np.unique(frame_indices)

        else: # use fps
            duration = max(0, end - start)
            total_frames = int(duration * fps) + 1  # include end
            timestamps = np.linspace(start_f, end_f, total_frames)
            indices = timestamps.astype(int)
            indices = np.clip(indices, 0, total_frame_num - 1)
            if max_frames_num is None or len(indices) <= max_frames_num:
                unique_indices = np.unique(indices)
            else:
                frame_indices = np.linspace(start_f, end_f, max_frames_num).astype(int)
                unique_indices = np.unique(frame_indices)

        try:
            frames = vr.get_batch(unique_indices).asnumpy()
        except Exception as e:
            print(f"Failed to load {video_path}: {e}", flush=True)

        return frames

    def load_video_ls(self, video_path, start, end, long=32, fps=1, l_fps=1/2):
        try:
            if type(video_path) == str:
                vr = VideoReader(video_path, ctx=cpu(0))
            else:
                vr = VideoReader(video_path[0], ctx=cpu(0))
        except Exception as e:
            print(f"Failed to load {video_path}: {e}", flush=True)

        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        long_start_f = min(max(0, int((start-long) * orig_fps)), total_frame_num - 1)
        start_f = min(max(0, int(start * orig_fps)), total_frame_num - 1)
        end_f = min(total_frame_num - 1, int(end * orig_fps))

        if end_f <= start_f:
            end_f = min(start_f + 1, total_frame_num - 1)

        # sample for short
        duration = max(0, end - start)
        total_frames = int(duration * fps) + 1  # include end
        timestamps = np.linspace(start_f, end_f, total_frames)
        indices = timestamps.astype(int)
        indices = np.clip(indices, 0, total_frame_num - 1)

        #sample for long
        if start > 0 and long > 0:
            l_duration = max(0, start - max(0, start - long))
            l_total_frames = int(l_duration * l_fps) + 1  # include end
            l_timestamps = np.linspace(long_start_f, start_f, l_total_frames)
            l_indices = l_timestamps.astype(int)
            l_indices = np.clip(l_indices, 0, total_frame_num - 1)

            indices = np.concatenate((indices, l_indices))

        unique_indices = np.unique(indices)
        frames = vr.get_batch(unique_indices).asnumpy()

        return frames
    
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            # toks = self.tok_encode(x[0])
            # return -len(toks), x[0]
            return x[3], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            visuals, start_time, end_time, target_fps = batched_doc_to_visual[0]  # [B, N]
            batched_visuals = [visuals]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        raise ValueError('not Supported')
                        # assert type(visual) == list, "sample_frames must be specified for video task"
                        # sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        # visual = [visual[i] for i in sample_indices]
                        # assert len(visual) == metadata["sample_frames"]
                        #
                        # image_tensor = process_images(visual, self._image_processor, self._config)
                        # if type(image_tensor) is list:
                        #     image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        # else:
                        #     image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                        #
                        # task_type = "video"
                        # placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        raise ValueError('not Supported')
                        # image_tensor = process_images(visual, self._image_processor, self._config)
                        # if type(image_tensor) is list:
                        #     image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        # else:
                        #     image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                        #
                        # task_type = "image"
                        # placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video_span(visual, self.max_frames_num, start_time, end_time, fps=target_fps)
                                #frames = self.load_video_span(visual, self.max_frames_num, start_time, end_time, fps=None)
                                #frames = self.load_video_ls(visual, start_time, end_time, long=32, fps=target_fps, l_fps=1/8)
                            elif self.video_decode_backend == "pyav":
                                raise ValueError('Unsupported backend')
                                #frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device) #.half()
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                raise ValueError('not Supported')
                #gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # When do_sample=False, remove sampling-related parameters to avoid warnings
            # These might be in gen_kwargs or in the model's generation_config
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                    # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            # self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
