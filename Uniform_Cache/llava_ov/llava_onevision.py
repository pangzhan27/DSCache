import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import sys
sys.path.append('./Uniform_Cache/llava_ov')

import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig
from types import MethodType
import utils
from transformers.models.qwen2.modeling_qwen2 import  DynamicCache
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
        context_len: int=8,
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
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map,  **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map,  **llava_model_args)

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

        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        from generate_opt import ReID_Qwen2Attention_forward, ReID_Qwen2Model_forward, LlavaQwenForCausalLM_get_initial_cache_position
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, Qwen2Model

        Qwen2Attention.forward = ReID_Qwen2Attention_forward
        Qwen2Model.forward = ReID_Qwen2Model_forward
        LlavaQwenForCausalLM._get_initial_cache_position = LlavaQwenForCausalLM_get_initial_cache_position

        ## cache the system prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
        prompt= conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        split_index = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
        input_ids = input_ids[:split_index]
        output = self.model(input_ids.unsqueeze(0).to(self.device), past_key_values=None, use_cache=True)
        # <|im_start|>system\n You are a helpful assistant.<|im_end|> \n <|im_start|>user \n
        self.system_cache = output.past_key_values
        #self.visual_cache = None
        self.visual_context_size = 196 * (context_len + 1) + self.system_cache[0][0].shape[2]
        self.context_len = context_len
        a= 1

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

    @torch.no_grad()
    def visual_cache_encode(self, frames, use_system=False):
        # TODO: visual encoding consider system prompt
        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
        encoded_image_features = self.model.encode_images(frames)
        video_features = self.model.get_2dPool(encoded_image_features)
        video_features = video_features[-1:]
        output = self.model(inputs_embeds=video_features,  past_key_values=self.visual_cache, use_cache=True)
        self.visual_cache = output.past_key_values

        if self.visual_cache[0][0].shape[2] > self.visual_context_size:
            to_remove = 196 # every frame has 196 tokens
            for i, (k_layer, v_layer) in enumerate(self.visual_cache):
                seq_len = k_layer.shape[2]
                indices_to_keep = list(range(self.system_cache[0][0].shape[2])) + list(range(self.system_cache[0][0].shape[2] + to_remove, seq_len))
                #indices_to_keep = list(range(to_remove, seq_len))
                indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
                self.visual_cache.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
                self.visual_cache.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)
            del k_layer, v_layer
            torch.cuda.empty_cache()

    def report_gpu_memory(self, tag=""):
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"[{tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB | peak={peak:.1f} MB", flush=True)

    def clone_dynamic_cache(self, cache):
        if cache is None:
            return None
        # shallow copy container
        new_cache = copy.copy(cache)
        # clone all key/value tensors
        new_cache.key_cache = [k.detach().clone() for k in cache.key_cache]
        new_cache.value_cache = [v.detach().clone() for v in cache.value_cache]
        return new_cache

    def generate_until(self, requests):
        res = []
        total_requests = 0
        for k, v in requests.items():
            total_requests += len(v)
        pbar = tqdm(total=total_requests, disable=(self.rank != 0), desc="Model Responding")
        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
            self._config.image_aspect_ratio = origin_image_aspect_ratio
            eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()


        self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
        self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        
        kk = 0
        for video_path, chunk in requests.items():
            # if kk <= 1200:
            #     pbar.update(len(chunk))
            #     kk += len(chunk)
            #     continue

            res.append([])
            _, _, batched_doc_to_visual, _, _, _ = zip(chunk[0].arguments)
            target_fps = batched_doc_to_visual[0][-1]
            self.visual_cache = self.clone_dynamic_cache(self.system_cache) # each video, reset visual cache

            ### prepare query
            queries, tmp_times = [], []
            for sep_chunk in chunk:
                batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(sep_chunk.arguments)
                visuals, end_time, _ = batched_doc_to_visual[0]
                queries.append((end_time, (batched_contexts, all_gen_kwargs, batched_doc_id)))
                tmp_times.append(end_time)
            max_time = max(tmp_times)

            ##### read the whole video
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            orig_fps = vr.get_avg_fps()
            duration = len(vr) / vr.get_avg_fps()
            #timestamps = np.arange(0, duration, 1.0 / target_fps)
            timestamps = np.arange(0, min(max_time + 1.0 / target_fps, duration), 1.0 / target_fps)
            print('modified version, {}'.format(self.visual_context_size))
            frame_indices = (timestamps * orig_fps).astype(int)
            frame_indices = np.clip(frame_indices, 0, total_frame_num - 1)
            unique_indices = np.unique(frame_indices)
            try:
                frames = vr.get_batch(unique_indices).asnumpy()
            except Exception as e:
                eval_logger.error(f"Error {e} in loading video")
                frames = None
            #print(video_path, tmp_times, timestamps, len(timestamps), len(frames) if frames is not None else -1, flush=True)
            
            frame_ptr = 0  # how many sampled frames have been encoded
            for qid, (query_time, query) in enumerate(queries):
                if frames is not None:
                    # 1. Encode frames strictly before (or up to) query_time
                    while frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                        #self.report_gpu_memory(frame_ptr)
                        self.visual_cache_encode(frames[frame_ptr:frame_ptr + 1], use_system=False)
                        # self.visual_cache_encode(frames[max(0, frame_ptr-self.context_len):frame_ptr + 1], use_system=False)
                        frame_ptr += 1

                # 2. get to query time
                batched_contexts, all_gen_kwargs, batched_doc_id = query
                gen_kwargs = all_gen_kwargs[0]
                if "until" in gen_kwargs:
                    gen_kwargs.pop("until")

                question = "\n" + batched_contexts[0] + '<|im_end|>\n<|im_start|>assistant\n'
                input_ids_list = [tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]

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
                    ### prepare cache
                    # past_key_values = DynamicCache()
                    # for layer_i in range(len(self.visual_cache.key_cache)):
                    #     past_key_values.key_cache.append(torch.cat((self.system_cache.key_cache[layer_i], self.visual_cache.key_cache[layer_i]), dim=-2))
                    #     past_key_values.value_cache.append(torch.cat((self.system_cache.value_cache[layer_i], self.visual_cache.value_cache[layer_i]), dim=-2))
                    past_key_values = self.clone_dynamic_cache(self.visual_cache)

                    #new_line
                    if frames is not None:
                        tempt_line = self.model.model.image_newline[None].to(self.model.device).unsqueeze(0)
                        tempt_output = self.model(inputs_embeds=tempt_line, past_key_values=past_key_values,
                                                  use_cache=True)
                        past_key_values = tempt_output.past_key_values


                    with torch.inference_mode():
                        cont = self.model.generate(input_ids, attention_mask=attention_masks,
                                                   pad_token_id=pad_token_ids, past_key_values = past_key_values,
                                                   use_cache=self.use_cache, **gen_kwargs)
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e

                text_outputs = [response.strip() for response in text_outputs]
                res[-1].extend(text_outputs)
                pbar.update(1)

        pbar.close()
        return res
