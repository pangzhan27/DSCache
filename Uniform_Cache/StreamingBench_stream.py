from importlib.metadata import metadata
from logging import logThreads

import torch
import logging
import pandas as pd
import time
from tqdm import tqdm
import os.path as osp
import os
import collections
from collections import defaultdict
from loguru import logger as eval_logger
import json, re
from datetime import datetime
import argparse
import copy, pickle
import matplotlib.pyplot as plt
import sys

sys.path.append('ADD your project path to Llava folder here')
from llava_ov.builder import load_pretrained_model, Instance
from llava_ov.llava_onevision import Llava_OneVision

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
LOG_PATH = "log/{model}_{task}_{curr_time}_{type}.log"
TASK_CSV = "../datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR = "../datasets/StreamingBench/Real-Time Visual Understanding"
PRE_PROMPT = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
POST_PROMPT = "Answer with the option's letter from the given choices directly."

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




def time_to_seconds(time_str):
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def contiguous_kv(past_key_values):
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        past_key_values.key_cache[i] = k_layer.contiguous()
        past_key_values.value_cache[i] = v_layer.contiguous()
    return past_key_values


@torch.no_grad()
def visual_cache_encode_llava(self, frames, use_system=False):
    output_log = ''
    # TODO: visual encoding consider system prompt
    frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
    assert (frames.shape[-1]) == 384 and (frames.shape[-2] == 384)
    encoded_image_features = self.model.encode_images(frames)
    video_features = self.model.get_2dPool(encoded_image_features)
    video_features = video_features[-1:]
    assert video_features.shape[1] == 196
    output = self.model(inputs_embeds=video_features, past_key_values=self.visual_cache, use_cache=True)
    self.visual_cache = output.past_key_values

    if self.visual_cache[0][0].shape[2] > self.visual_context_size:
        to_remove = 196  # every frame has 196 tokens
        for i, (k_layer, v_layer) in enumerate(self.visual_cache):
            seq_len = k_layer.shape[2]
            indices_to_keep = list(range(self.system_cache[0][0].shape[2])) + list(
                range(self.system_cache[0][0].shape[2] + to_remove, seq_len))
            # indices_to_keep = list(range(to_remove, seq_len))
            indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
            self.visual_cache.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
            self.visual_cache.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)
        del k_layer, v_layer
        torch.cuda.empty_cache()
        output_log += f'cut cache, {self.visual_cache.key_cache[0].shape}'
        return output_log

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
        # each video, reset visual cache
        self.visual_cache = self.clone_dynamic_cache(self.system_cache)

        ### prepare query
        queries, tmp_times = [], []
        for sep_chunk in chunk:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(
                sep_chunk.arguments)
            visuals, end_time, _ = batched_doc_to_visual[0]
            queries.append((end_time, (batched_contexts, all_gen_kwargs, batched_doc_id)))
            tmp_times.append(end_time)
        max_time = max(tmp_times)

        ##### read the whole video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        duration = len(vr) / vr.get_avg_fps()
        # timestamps = np.arange(0, duration, 1.0 / target_fps)
        timestamps = np.arange(0, min(max_time + 1.0 / target_fps, duration), 1.0 / target_fps)
        print('modified version, {}'.format(self.visual_context_size))
        frame_indices = (timestamps * orig_fps).astype(int)
        frame_indices = np.clip(frame_indices, 0, total_frame_num - 1)
        unique_indices = np.unique(frame_indices)

        for _ in range(5):
            try:
                frames = vr.get_batch(unique_indices).asnumpy()
                if frames is not None:
                    break
            except Exception as e:
                frames = None

        if frames is None:
            logger.error(f"Error in loading video {video_path}")
            continue

        # print(video_path, tmp_times, timestamps, len(timestamps), len(frames) if frames is not None else -1, flush=True)

        frame_ptr = 0  # how many sampled frames have been encoded
        for qid, (query_time, query) in enumerate(queries):
            if frames is not None:
                # 1. Encode frames strictly before (or up to) query_time
                while frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                    mm_log = report_gpu_memory(frame_ptr, print_ok=False)
                    output_log = self.visual_cache_encode(frames[frame_ptr:frame_ptr + 1], use_system=False)
                    # self.visual_cache_encode(frames[max(0, frame_ptr-self.context_len):frame_ptr + 1], use_system=False)
                    frame_ptr += 1
                    #print('[Stream]  ', mm_log, ' | ', output_log, flush=True )

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
                past_key_values = self.clone_dynamic_cache(self.visual_cache)
                # new_line
                with torch.inference_mode():
                    if frames is not None:
                        tempt_line = self.model.model.image_newline[None].to(self.model.device).unsqueeze(0)
                        tempt_output = self.model(inputs_embeds=tempt_line, past_key_values=past_key_values,
                                                  use_cache=True)
                        past_key_values = tempt_output.past_key_values

                    cont = self.model.generate(input_ids, attention_mask=attention_masks,
                                               pad_token_id=pad_token_ids, past_key_values=past_key_values,
                                               use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                #print(text_outputs)
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res[-1].extend(text_outputs)
            pbar.update(1)

    pbar.close()
    return res


def llava_baseline(ckpt_path='lmms-lab/llava-onevision-qwen2-7b-ov', context_len=32):
    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Llava_ov', task=task, curr_time=curr_time, type=f'stream{context_len}'))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    output_jsonl = f"results/llava_ov_{task}_{curr_time}_stream{context_len}.jsonl"

    ## 1. first build instances
    task_df = pd.read_csv(TASK_CSV)
    generation_kwargs = {'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False,
                         'until': ['\n\n']}
    split = 'test'
    requests = collections.defaultdict(dict)
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        # try:
        question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
            row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type

        # create doc
        doc = [question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type]
        doc_id = row[0]

        ## doc2video
        video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        time_stamp_sec = time_to_seconds(time_stamp)
        fps = 1
        doc_to_visual = ([video_path], time_stamp_sec, fps)

        ## doc_2_text, pre_prompt + question + post prompt
        option = "\n".join([f"{opt}" for i, opt in enumerate(eval(options))])
        question = question + "\n" + option
        ctx = PRE_PROMPT + "\n" + question + "\n" + POST_PROMPT

        ## construct_requests
        arguments = (ctx, generation_kwargs, doc_to_visual, doc_id, task, split)
        kwargs = {'metadata': {"task": task, "doc_id": doc_id, "repeats": 1, "split": split}}
        inst = Instance(request_type='generate_until', arguments=arguments, idx=0, doc=doc, **kwargs)  #
        reqtype = inst.request_type
        if video_path not in requests[reqtype]:
            requests[reqtype][video_path] = []
        requests[reqtype][video_path].append(inst)

    for reqtype, reqs in requests.items():
        for video_path in reqs:
            reqs[video_path].sort(key=lambda inst: inst.arguments[2][1])  # doc[3] = timestamp

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    #
    Llava_OneVision.generate_until = generate_until
    Llava_OneVision.visual_cache_encode = visual_cache_encode_llava
    model = Llava_OneVision(pretrained=ckpt_path,
                                 conv_template='qwen_1_5',
                                 model_name='llava_qwen',
                                 attn_implementation='sdpa',
                                 device_map='cuda',
                                 device='cuda',
                                 context_len=context_len,
                                 )

    logger.info(f"Load model and processor from {ckpt_path}")

    ## 2. generate response
    output_dict = []
    start_time = time.time()
    for reqtype, reqs in requests.items():
        from itertools import islice
        # reqs = dict(islice(reqs.items(), 2))
        eval_logger.info(f"Processing {reqtype} with {len(reqs)} requests")
        eval_logger.info("Running {} requests".format(reqtype))

        responses = getattr(model, reqtype)(reqs)
        logger.info(f"Totally get {len(responses)} responses")

        for req_a, response_a in zip(reqs.values(), responses):
            for req, response in zip(req_a, response_a):
                question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = req.doc
                output_dict.append({
                    'question_id': question_id,
                    'task_type': task_type,
                    'question': question,
                    'time_stamp': time_stamp,
                    'answer': answer,
                    'options': eval(options),
                    'frames_required': frames_required,
                    'temporal_clue_type': temporal_clue_type,
                    'response': response
                })

    with open(output_jsonl, "w") as f:
        json.dump(output_dict, f, indent=4)

    ## 3. calcuate results
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    for item in output_dict:
        cnt_total['overall'] += 1
        cnt_total[item['task_type']] += 1
        if extract_characters_regex(item['response']) == item['answer']:
            cnt_correct['overall'] += 1
            cnt_correct[item['task_type']] += 1
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception',
                  'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning',
                  'Spatial Understanding', 'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(
                f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100 * cnt_correct[task_type] / cnt_total[task_type]:.2f}%")
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%")
    end_time = time.time()
    cost_time = int(end_time - start_time)


# ------------------------------------------------------------------------------------------------------------------------

from qwen_vl_utils.vision_process import (smart_nframes, smart_resize)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from llava_ov.utils import StreamingArgs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import numpy as np

Qwen_prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

Answer with the option's letter from the given choices directly."""

TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "user": 872,
    "assistant": 77091,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|video_pad|>": 151656,
    "\n": 198,
    'system': 8948,
    'previous text': [19702, 1467],
    'Time': 1462,
}


def report_gpu_memory(tag="", print_ok=True):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    if print_ok:
       print(f"[{tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB | peak={peak:.1f} MB", flush=True)
    return f"[{tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB | peak={peak:.1f} MB"


def clone_dynamic_cache(cache):
    if cache is None:
        return None
    # shallow copy container
    new_cache = copy.copy(cache)
    # clone all key/value tensors
    new_cache.key_cache = [k.detach().clone() for k in cache.key_cache]
    new_cache.value_cache = [v.detach().clone() for v in cache.value_cache]
    return new_cache


@torch.no_grad()
def visual_cache_encode(model, processor, video_inputs, stream_args, use_system=False):
    output_log = ''
    ##### patch pixel value
    dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
    dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
    dummy_inputs = processor(text=[dummy_text], videos=video_inputs, padding=True, return_tensors="pt", ).to(
        model.device)
    pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
    cur_video_embeds = model.visual(pixel_values_videos, grid_thw=dummy_inputs['video_grid_thw'])
    
    # print(video_inputs.shape, pixel_values_videos.shape, dummy_inputs['video_grid_thw'], flush= True)
    num_token_per_frame = torch.sum(dummy_inputs['input_ids'] == TOKEN_IDS['<|video_pad|>']).item() // \
                          dummy_inputs['video_grid_thw'][0, 0].item()
    if not hasattr(model, 'num_token_per_frame'):
        model.num_token_per_frame = num_token_per_frame
        model.visual_context_size = (model.context_len + 1) * num_token_per_frame + model.system_cache[0][0].shape[2]
        print('####[', model.video_name, '] Num token per frame :', model.num_token_per_frame,
              model.visual_context_size)
    else:
        assert model.num_token_per_frame == num_token_per_frame, 'mismatch frame resolution'

    cur_video_embeds = cur_video_embeds[-model.num_token_per_frame:]
    assert cur_video_embeds.shape[0] == num_token_per_frame, 'mismatch frame resolution'
    

    #### pad for stream_args
    video_pad = torch.ones((1, num_token_per_frame), device=stream_args.input_ids.device,
                           dtype=stream_args.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
    stream_args.input_ids = torch.cat((stream_args.input_ids, video_pad), dim=-1)
    if stream_args.video_grid_thw is None:
        assert dummy_inputs['video_grid_thw'][:, 0] == 1, 'only support one frame per time'
        stream_args.video_grid_thw = dummy_inputs['video_grid_thw']
    else:
        stream_args.video_grid_thw[:, 0] = stream_args.video_grid_thw[:, 0] + 1  # update one frame

    # print(stream_args.input_ids.device, cur_video_embeds.device, model.visual_cache.key_cache[0].device)
    output = model(inputs_embeds=cur_video_embeds.unsqueeze(0), past_key_values=model.visual_cache, use_cache=True,
                   streaming_args=stream_args.copy())
    model.visual_cache = output.past_key_values
    assert stream_args.input_ids.shape[1] == model.visual_cache[0][0].shape[2]

    if model.visual_cache[0][0].shape[2] > model.visual_context_size:
        output_log += 'Cache eviction. '
        to_remove = model.num_token_per_frame  # every frame has 196 tokens
        for i, (k_layer, v_layer) in enumerate(model.visual_cache):
            seq_len = k_layer.shape[2]
            indices_to_keep = list(range(model.system_cache[0][0].shape[2])) + list(
                range(model.system_cache[0][0].shape[2] + to_remove, seq_len))
            # indices_to_keep = list(range(to_remove, seq_len))
            indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
            model.visual_cache.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
            model.visual_cache.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)
        stream_args.video_grid_thw[:, 0] = stream_args.video_grid_thw[:, 0] - 1
        stream_args.input_ids = torch.index_select(stream_args.input_ids, 1, indices_tensor)
        assert stream_args.video_grid_thw[:, 0] == model.context_len + 1
        assert stream_args.input_ids.shape[1] == model.visual_context_size

        del k_layer, v_layer
        torch.cuda.empty_cache()

    assert stream_args.input_ids.shape[1] == model.visual_cache[0][0].shape[2]
    return model.visual_cache, stream_args, output_log


def prepare_system_cache(model, processor, stream_args):
    messages = [{"role": "user", "content": [{"type": "video", "video": '', }, ], }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = processor.tokenizer(text, padding=True, return_tensors="pt", )['input_ids'][0]
    split_index = (input_ids == model.config.video_token_id).nonzero(as_tuple=True)[0][0].item()
    input_ids = input_ids[:split_index].unsqueeze(0).to(model.device)

    output = model(input_ids, past_key_values=None, use_cache=True, streaming_args=stream_args.copy())
    stream_args.input_ids = torch.cat((stream_args.input_ids, input_ids), dim=-1)

    return output.past_key_values, stream_args


def qwen_baseline(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct', target_fps=1, context_len=32):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLSdpaAttention, Qwen2_5_VLModel
    from llava_ov.generate_qwen_op import (Qwen2_5_VLSdpaAttention_forward, Qwen2_5_VLForConditionalGeneration_forward,
                                           Qwen2_5_VLModel_forward, _get_initial_cache_position,
                                           prepare_inputs_for_generation,
                                           _update_model_kwargs_for_generation)
    Qwen2_5_VLSdpaAttention.forward = Qwen2_5_VLSdpaAttention_forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward
    Qwen2_5_VLForConditionalGeneration._get_initial_cache_position = _get_initial_cache_position
    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation
    Qwen2_5_VLForConditionalGeneration._update_model_kwargs_for_generation = _update_model_kwargs_for_generation

    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 448 * 448

    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time, type=f'stream{context_len}'))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}_stream{context_len}.jsonl"

    # Load task info
    task_df = pd.read_csv(TASK_CSV)

    requests = {}
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
            row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type

        # create doc
        doc_id = row[0]
        video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        time_stamp_sec = time_to_seconds(time_stamp)
        ctx = Qwen_prompt.format(question, '\n'.join(eval(options)))

        ## construct_requests
        arguments = (time_stamp_sec, ctx, doc_id,
                     (question_id, task_type, question, time_stamp, options, answer, frames_required,
                      temporal_clue_type))
        if video_path not in requests:
            requests[video_path] = []
        requests[video_path].append(arguments)

    for video_path in requests:
        requests[video_path].sort(key=lambda inst: inst[0])  # doc[3] = timestamp

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        ckpt_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Load model and processor from {ckpt_path}")

    stream_args_org = StreamingArgs(model.model.device)
    ### prepare system cache
    system_cache, stream_args_org = prepare_system_cache(model, processor, stream_args_org)
    model.system_cache = system_cache
    model.context_len = context_len

    # Inference
    start_time = time.time()
    output_dict = []
    kk = 0
    for vid in tqdm(requests, total=len(requests)):
        if hasattr(model, "num_token_per_frame"):
            delattr(model, "num_token_per_frame")
        model.video_name = vid
        model.visual_cache = clone_dynamic_cache(model.system_cache)  # each video, reset visual cache
        stream_args = stream_args_org.copy()

        #
        vr = VideoReader(vid, ctx=cpu(0))
        height, width, _ = vr.next().shape
        resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS,
                                                     max_pixels=MAX_PIXELS)

        temt_time = [aa[0] for aa in requests[vid]]
        max_time = max(temt_time)
        ##### read the whole video
        vr = VideoReader(vid, ctx=cpu(0))
        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        duration = len(vr) / vr.get_avg_fps()
        timestamps = np.arange(0, min(max_time + 1.0 / target_fps, duration), 1.0 / target_fps)
        frame_indices = (timestamps * orig_fps).astype(int)
        frame_indices = np.clip(frame_indices, 0, total_frame_num - 1)
        unique_indices = np.unique(frame_indices)

        for _ in range(5):
            try:
                frames = torch.from_numpy(vr.get_batch(unique_indices).asnumpy()).permute(0, 3, 1, 2)
                frames = transforms.functional.resize(frames, [resized_height, resized_width],
                                                  interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
                if frames is not None:
                    break
            except Exception as e:
                frames = None

        if frames is None:
            logger.error(f"Error in loading video {vid}")
            continue


        # test_compare(frames, processor, model)
        frame_ptr = 0
        for item in requests[vid]:
            query_time, ctx, doc_id, extra = item
            if frames is not None:
                # 1. Encode frames strictly before (or up to) query_time(only cache two frames)
                while frame_ptr + 1 < len(timestamps) and timestamps[frame_ptr + 1] <= query_time:
                    mem_log = report_gpu_memory(frame_ptr, print_ok=False)
                    to_frames = frames[frame_ptr: frame_ptr + 2]
                    model.visual_cache, stream_args, output_log = visual_cache_encode(model, processor, to_frames, use_system=False,
                                                                          stream_args=stream_args)
                    frame_ptr += 2
                    #print('[Streaming]  ', mem_log, ' | ', output_log, flush=True)

            ####  get to query response
            try:
                # model.to('cpu')
                question_id, task_type, question, time_stamp, options, answer, frames_required, temporal_clue_type = extra

                text_input = '<|vision_end|>' + ctx + '<|im_end|>\n<|im_start|>assistant\n'
                inputs = processor(text=[text_input], images=None, videos=None,
                                   padding=True, return_tensors='pt').to(model.device)

                past_key_values = clone_dynamic_cache(model.visual_cache)
                stream_tempt_args = stream_args.copy()

                ### Do temporal cache for extra frames that cannot be paired
                if frames is not None:
                    with torch.no_grad():
                        # for cases that only one frame left for the query, without being in cache
                        if frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                            # do not modify frame_ptr so that next time, it can use the left frame for new cache.
                            assert frame_ptr + 1 >= len(timestamps) or timestamps[frame_ptr + 1] > query_time
                            to_frames = frames[frame_ptr: frame_ptr + 1]
                            # print(f'\t <before> [{frame_ptr}]', past_key_values.key_cache[0].shape, model.visual_cache.key_cache[0].shape,
                            #       stream_tempt_args.input_ids.shape, flush=True)
                            ##### patch pixel value
                            dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
                            dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
                            dummy_inputs = processor(text=[dummy_text], videos=to_frames, padding=True, return_tensors="pt", ).to(model.device)
                            pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
                            cur_video_embeds = model.visual(pixel_values_videos, grid_thw=dummy_inputs['video_grid_thw'])
                            assert cur_video_embeds.shape[0] == model.num_token_per_frame, 'mismatch frame resolution'

                            #### pad for stream_args
                            video_pad = torch.ones((1, model.num_token_per_frame), device=stream_args.input_ids.device,
                                                   dtype=stream_args.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
                            stream_tempt_args.input_ids = torch.cat((stream_tempt_args.input_ids, video_pad), dim=-1)
                            if stream_tempt_args.video_grid_thw is None:
                                stream_tempt_args.video_grid_thw = dummy_inputs['video_grid_thw']
                                stream_tempt_args.video_grid_thw[:, 0] = 0
                            stream_tempt_args.video_grid_thw[:, 0] = stream_tempt_args.video_grid_thw[:, 0] + 1  # update one frame

                            # print(stream_args.input_ids.device, cur_video_embeds.device, model.visual_cache.key_cache[0].device)
                            output = model(inputs_embeds=cur_video_embeds.unsqueeze(0),
                                           past_key_values=clone_dynamic_cache(model.visual_cache),
                                           use_cache=True,
                                           streaming_args=stream_tempt_args.copy())
                            past_key_values = output.past_key_values
                            # print(f'\t <middle> [{frame_ptr}]', past_key_values.key_cache[0].shape,
                            #       model.visual_cache.key_cache[0].shape, stream_tempt_args.input_ids.shape, flush=True)
                            if past_key_values[0][0].shape[2] > model.visual_context_size:
                                to_remove = model.num_token_per_frame  # every frame has 196 tokens
                                for i, (k_layer, v_layer) in enumerate(past_key_values):
                                    seq_len = k_layer.shape[2]
                                    indices_to_keep = list(range(model.system_cache[0][0].shape[2])) + list(
                                        range(model.system_cache[0][0].shape[2] + to_remove, seq_len))
                                    indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
                                    past_key_values.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
                                    past_key_values.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)
                                stream_tempt_args.video_grid_thw[:, 0] = stream_tempt_args.video_grid_thw[:, 0] - 1
                                stream_tempt_args.input_ids = torch.index_select(stream_tempt_args.input_ids, 1, indices_tensor)
                                assert stream_tempt_args.video_grid_thw[:, 0] == model.context_len + 1
                                assert stream_tempt_args.input_ids.shape[1] == model.visual_context_size

                            # print(f'\t <after> [{frame_ptr}]', past_key_values.key_cache[0].shape,
                            #       model.visual_cache.key_cache[0].shape, stream_tempt_args.input_ids.shape, flush=True)

                with torch.inference_mode():
                    if hasattr(model, "num_token_per_frame"):
                        num_frame = (past_key_values.key_cache[0].shape[2] - model.system_cache[0][0].shape[2]) //model.num_token_per_frame
                        assert stream_tempt_args.video_grid_thw[:, 0] == num_frame, 'inconsistency before generation'
                    
                    generated_ids = model.generate(
                        input_ids=inputs['input_ids'],
                        max_new_tokens=16,
                        past_key_values=past_key_values,
                        use_cache=True,
                        streaming_args=stream_tempt_args.copy(),
                    )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = output_text[0]
                
                output_dict.append({
                    'question_id': question_id,
                    'task_type': task_type,
                    'question': question,
                    'time_stamp': time_stamp,
                    'answer': answer,
                    'options': eval(options),
                    'frames_required': frames_required,
                    'temporal_clue_type': temporal_clue_type,
                    'response': response
                })
                #print(question_id, answer, response)

            except Exception as e:
                logger.error(f"Error in processing {row}: {e}")

    with open(output_jsonl, "w") as f:
        json.dump(output_dict, f, indent=4)

    ## 3. calcuate results
    logger.info(f"Totally get {len(output_dict)} responses")
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    for item in output_dict:
        cnt_total['overall'] += 1
        cnt_total[item['task_type']] += 1
        if extract_characters_regex(item['response']) == item['answer']:
            cnt_correct['overall'] += 1
            cnt_correct[item['task_type']] += 1
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception',
                  'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning',
                  'Spatial Understanding', 'Action Perception', 'Counting']
    for task_type in task_types:
        if cnt_total[task_type] == 0:
            logger.info(f"- {task_type}: No question processed")
        else:
            logger.info(
                f"- {task_type}: {cnt_correct[task_type]}/{cnt_total[task_type]} = {100 * cnt_correct[task_type] / cnt_total[task_type]:.2f}%")
    if cnt_total['overall'] == 0:
        logger.info("No question processed")
    else:
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {100 * cnt_correct['overall'] / cnt_total['overall']:.2f}%")
    end_time = time.time()
    cost_time = int(end_time - start_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llava')
    parser.add_argument("--context_len", type=int, default=8)

    args = parser.parse_args()
    if 'llava' in args.model_name:
        llava_baseline(context_len=args.context_len)
    elif 'qwen' in args.model_name:
        qwen_baseline(context_len=args.context_len)
    else:
        raise NotImplementedError
