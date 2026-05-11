from importlib.metadata import metadata
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
import json, re, copy
from datetime import datetime
import argparse
import sys
from decord import VideoReader, cpu
import numpy as np

sys.path.append('ADD your project path to Llava folder here')

from llava_ov.builder import load_pretrained_model, Instance
from llava_ov.llava_onevision import Llava_OneVision
from llava.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

LOG_PATH = "log/{model}_{task}_{curr_time}_{type}.log"
TASK_CSV = "../datasets/OVO-Bench/ovo_bench_new.json"
VIDEO_DIR = "../datasets/OVO-Bench/videos/src_videos"

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]


def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""Question: {question}
Options:
{formatted_options}
Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response."""
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f"""You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {question}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response."""
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{step}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""

    elif task == "CRR":
        question = _anno_["question"]
        answer = _anno_["answer"]
        prompt = f"""You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{question}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response."""
    return prompt


def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)

        # Calculate Score for Every Result
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])

        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)

        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)

        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            # Calculate score for REC
            if result["task"] == "REC":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    # scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for SSR
            if result["task"] == "SSR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (
                            test_info_["response"] == "Y" and test_info_["type"] == 1):
                        # scores["SSR"].append(1)
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    # scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for CRR
            if result["task"] == "CRR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (
                            test_info_["response"] == "Y" and test_info_["type"] == 1):
                        # scores["CRR"].append(1)
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    # scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores

    backward_results = results["backward"]
    realtime_results = results["realtime"]
    forward_results = results["forward"]
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }

    if len(backward_results) > 0:
        # print("Evaluate Backward Tracing...")
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        # correct_backward, total_backward = 0, 0
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}, Total: {len(v)}")
            # correct_backward += sum(v)
            # total_backward += len(v)
            avg_scores["backward"].append(sum(v) / len(v))
        # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward']) / len(avg_scores['backward']):.2f}\n")
    else:
        # correct_backward = 0
        # total_backward = 0
        pass

    if len(realtime_results) > 0:
        # print("Evaluate Real-time Visual Perception...")
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        # correct_realtime, total_realtime = 0, 0
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}, Total: {len(v)}")
            # correct_realtime += sum(v)
            # total_realtime += len(v)
            avg_scores["realtime"].append(sum(v) / len(v))
        # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime']) / len(avg_scores['realtime']):.2f}\n")
    else:
        # correct_realtime = 0
        # total_realtime = 0
        pass

    if len(forward_results) > 0:
        # print("Evaluate Forward Active Responding...")
        forward_results, forward_scores = calculate_score_forward(forward_results)
        # correct_forward, total_forward = 0, 0
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v) / len(v):.2f}, Total: {len(v)}")
            # correct_forward += sum(v)
            # total_forward += len(v)
            avg_scores["forward"].append(sum(v) / len(v))
        # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward']) / len(avg_scores['forward']):.2f}\n")
    else:
        # correct_forward = 0
        # total_forward = 0
        pass

    # logger.info(f"Total Avg.: {100 * (sum(avg_scores['backward']) + sum(avg_scores['realtime']) + sum(avg_scores['forward'])) / (len(avg_scores['backward']) + len(avg_scores['realtime']) + len(avg_scores['forward'])):.2f}")
    logger.info(
        f"Total Avg.: {100 * (sum(avg_scores['backward']) / len(avg_scores['backward']) + sum(avg_scores['realtime']) / len(avg_scores['realtime']) + sum(avg_scores['forward']) / len(avg_scores['forward'])) / 3:.2f}")


def get_response(model, doc, frames):
    id, visual, task, context, time_stamp_sec, gt = doc

    gen_kwargs = {'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False}

    origin_image_aspect_ratio = getattr(model._config, "image_aspect_ratio", None)
    if origin_image_aspect_ratio is not None and model._config.image_aspect_ratio != origin_image_aspect_ratio:
        model._config.image_aspect_ratio = origin_image_aspect_ratio
        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

    question = "\n" + context + '<|im_end|>\n<|im_start|>assistant\n'
    # This is much safer for llama3, as we now have some object type in it
    if "llama_3" in model.conv_template:
        conv = copy.deepcopy(conv_templates[model.conv_template])
    else:
        conv = conv_templates[model.conv_template].copy()

    input_ids_list = [tokenizer_image_token(question, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
    pad_token_ids = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    input_ids = model.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(model.device)
    attention_masks = input_ids.ne(pad_token_ids).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, model.tokenizer, input_ids)
    gen_kwargs["modalities"] = ["video"]
    gen_kwargs["stopping_criteria"] = [stopping_criteria]
    model._config.mm_spatial_pool_stride = model.mm_spatial_pool_stride
    model._config.mm_spatial_pool_mode = model.mm_spatial_pool_mode

    if "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 1024

    if "image_aspect_ratio" in gen_kwargs.keys():
        gen_kwargs.pop("image_aspect_ratio")
    if not gen_kwargs.get("do_sample", False):
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)
        gen_kwargs.pop("top_k", None)

    past_key_values = model.clone_dynamic_cache(model.visual_cache)

    with torch.inference_mode():
        # new_line
        if frames:
            tempt_line = model.model.model.image_newline[None].to(model.model.device).unsqueeze(0)
            tempt_output = model.model(inputs_embeds=tempt_line, past_key_values=past_key_values, use_cache=True)
            past_key_values = tempt_output.past_key_values

        cont = model.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids,
                                    past_key_values=past_key_values, use_cache=model.use_cache, **gen_kwargs)

    text_outputs = model.tokenizer.batch_decode(cont, skip_special_tokens=True)

    text_outputs = [response.strip() for response in text_outputs]

    return text_outputs[0]


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

def llava_baseline(ckpt_path='lmms-lab/llava-onevision-qwen2-7b-ov', context_len=32):
    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Llava_ov', task=task, curr_time=curr_time, type=f'stream{context_len}'))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    target_fps = 1

    output_jsonl = f"results/llava_ov_{task}_{curr_time}_stream{context_len}.jsonl"

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    #
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

    # Load task info
    with open(TASK_CSV, 'r') as f:
        task_list = json.load(f)

    ## prepare video-clustered instances
    video_dict, total = {}, 0
    for item in tqdm(task_list):
        if item['task'] in backward_tasks or item['task'] in realtime_tasks:
            id, video, task, question, options, realtime, gt = \
                item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
            if video not in video_dict:
                video_dict[video] = []
            video_dict[video].append(item)
            total += 1
        elif item['task'] in forward_tasks:
            id, video, task, test_info = item['id'], item['video'], item['task'], item['test_info']
            total += len(test_info)
            for i in range(len(test_info)):
                new_item = item.copy()
                new_item['test_info'] = [test_info[i]]
                new_item['realtime'] = test_info[i]['realtime']
                new_item['index'] = i
                if video not in video_dict:
                    video_dict[video] = []
                video_dict[video].append(new_item)

        else:
            raise ValueError

    video_anno, total_num = {}, 0
    for vid, values in video_dict.items():
        # num_test_info = [1  if 'test_info' in v else 0 for v in values ]
        # print(vid, sum(num_test_info))
        values = sorted(values, key=lambda x: x['realtime'])
        video_anno[vid] = values
        total_num += len(values)

    start_time = time.time()
    kk = 0
    for vid in tqdm(video_anno.keys()):
        # for each video, reset
        model.visual_cache = model.clone_dynamic_cache(model.system_cache)
        video_path = osp.join(VIDEO_DIR, vid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        temt_time = [aa['realtime'] for aa in video_anno[vid]]
        max_time = max(temt_time)

        ##### read the whole video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        duration = len(vr) / vr.get_avg_fps()
        # timestamps = np.arange(0, duration, 1.0 / target_fps)
        timestamps = np.arange(0, min(max_time + 2.0 / target_fps, duration), 1.0 / target_fps)
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

        frame_ptr = 0
        for item in video_anno[vid]:
            query_time = int(item['realtime'] + 0.5)
            if frames is not None:
                # 1. Encode frames strictly before (or up to) query_time
                while frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                    mm_log = report_gpu_memory(frame_ptr, print_ok=False)
                    output_log = model.visual_cache_encode(frames[frame_ptr:frame_ptr + 1], use_system=False)
                    frame_ptr += 1
                    #print('[Stream]  ', mm_log, ' | ', output_log, flush=True)

            # 2. when reach to a timestamp, find all queries need to be answered
            try:
                if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                    id, video, task, question, options, realtime, gt = \
                        item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], \
                        item['gt']
                    # if id !=1127: continue
                    prompt = build_prompt(
                        task=task,
                        question=question,
                        options=options,
                        _anno_=None,
                        index=None,
                    )

                    doc = [id, video_path, task, prompt, realtime, gt]
                    response = get_response(model=model, doc=doc, frames=True if frames is not None else False)

                    output_dict = {
                        'id': id,
                        'video': video,
                        'task': task,
                        'question': question,
                        'response': response,
                        'ground_truth': chr(65 + gt),
                    }

                elif item['task'] in forward_tasks:
                    id, video, task, test_info = item['id'], item['video'], item['task'], item['test_info']
                    assert len(test_info) == 1, "only support one test"
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=0,
                    )

                    doc = [id, video_path, task, prompt, realtime, gt]
                    response = get_response(model=model, doc=doc, frames=True if frames is not None else False)
                    item['test_info'][0]['response'] = response

                    output_dict = item

                print(response)
                with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')

            except Exception as e:
                logger.error(f"Error in processing {item}: {e}")

    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results

    with open(output_jsonl, 'r') as f:
        lines = f.readlines()

    ### we need to map back to original anno format, as forward tasks are evalutae avg one instance, instead of overall
    final_dict = {}
    for line in lines:
        item = json.loads(line)
        id = item['id']
        if id not in final_dict:
            final_dict[id] = item
        else:
            assert item['task'] in forward_tasks
            assert len(item['test_info']) == 1
            final_dict[id]['test_info'].append(item['test_info'][0])

    items = [final_dict[j] for j in sorted(final_dict)]

    results = defaultdict(list)
    for item in items:
        # item = json.loads(line)
        if item['task'] in backward_tasks:
            results['backward'].append(item)
        elif item['task'] in realtime_tasks:
            results['realtime'].append(item)
        elif item['task'] in forward_tasks:
            results['forward'].append(item)
    score(results)


# ------------------------------------------------------------------------------------------------------------------------
from qwen_vl_utils.vision_process import (smart_nframes, smart_resize)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from llava_ov.utils import StreamingArgs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import numpy as np

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
    dummy_inputs = processor(text=[dummy_text], videos=video_inputs, padding=True, return_tensors="pt", ).to(model.device)
    pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
    num_token_per_frame = torch.sum(dummy_inputs['input_ids'] == TOKEN_IDS['<|video_pad|>']).item() // \
                          dummy_inputs['video_grid_thw'][0, 0].item()

    if not hasattr(model, 'num_token_per_frame'):
        model.num_token_per_frame = num_token_per_frame
        model.visual_context_size = (model.context_len + 1) * num_token_per_frame + model.system_cache[0][0].shape[2]
        print('####[', model.video_name, '] Num token per frame :', model.num_token_per_frame,
              model.visual_context_size)
    else:
        assert model.num_token_per_frame == num_token_per_frame, 'mismatch frame resolution'

    cur_video_embeds = model.visual(pixel_values_videos, grid_thw=dummy_inputs['video_grid_thw'])
    cur_video_embeds = cur_video_embeds[-model.num_token_per_frame:]

    #### pad for stream_args
    assert cur_video_embeds.shape[0] == num_token_per_frame, 'mismatch frame resolution'
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
    # print(output.past_key_values.key_cache[0].shape, flush=True)
    assert stream_args.input_ids.shape[1] == model.visual_cache[0][0].shape[2]

    if model.visual_cache[0][0].shape[2] > model.visual_context_size:
        output_log += 'Cache eviction. '
        to_remove = model.num_token_per_frame  # every frame has 196 tokens
        for i, (k_layer, v_layer) in enumerate(model.visual_cache):
            seq_len = k_layer.shape[2]
            indices_to_keep = list(range(model.system_cache[0][0].shape[2])) + list(
                range(model.system_cache[0][0].shape[2] + to_remove, seq_len))
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
    # print(text, output.past_key_values.key_cache[0].shape, flush=True)

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

    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time, type=f'stream{context_len}'))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}_{context_len}_stream{context_len}.jsonl"

    # Load task info
    with open(TASK_CSV, 'r') as f:
        task_list = json.load(f)

    ## prepare video-clustered instances
    video_dict, total = {}, 0
    for item in tqdm(task_list):
        if item['task'] in backward_tasks or item['task'] in realtime_tasks:
            id, video, task, question, options, realtime, gt = \
                item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
            if video not in video_dict:
                video_dict[video] = []
            video_dict[video].append(item)
            total += 1
        elif item['task'] in forward_tasks:
            id, video, task, test_info = item['id'], item['video'], item['task'], item['test_info']
            total += len(test_info)
            for i in range(len(test_info)):
                new_item = item.copy()
                new_item['test_info'] = [test_info[i]]
                new_item['realtime'] = test_info[i]['realtime']
                new_item['index'] = i
                if video not in video_dict:
                    video_dict[video] = []
                video_dict[video].append(new_item)

        else:
            raise ValueError

    video_anno, total_num = {}, 0
    for vid, values in video_dict.items():
        # num_test_info = [1  if 'test_info' in v else 0 for v in values ]
        # print(vid, sum(num_test_info))
        values = sorted(values, key=lambda x: x['realtime'])
        video_anno[vid] = values
        total_num += len(values)

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
    for vid in tqdm(video_anno.keys()):
        # each video may have different resolution. so reset context window length for each video
        if hasattr(model, "num_token_per_frame"):
            delattr(model, "num_token_per_frame")
        model.video_name = vid
        model.visual_cache = clone_dynamic_cache(model.system_cache)
        stream_args = stream_args_org.copy()
        # if '9eR7x93WqA8.mp4' not in vid: continue
        video_path = osp.join(VIDEO_DIR, vid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        temt_time = [aa['realtime'] for aa in video_anno[vid]]
        max_time = max(temt_time)
        #
        vr = VideoReader(video_path, ctx=cpu(0))
        try:
            height, width, _ = vr.next().shape
        except Exception:
            frame0 = vr[0]
            height, width, _ = frame0.shape
            print("H, W ", height, width, flush=True)
        resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS,
                                                     max_pixels=MAX_PIXELS)
        ##### read the whole video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        orig_fps = vr.get_avg_fps()
        duration = len(vr) / vr.get_avg_fps()
        # timestamps = np.arange(0, duration, 1.0 / target_fps)
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

        frame_ptr = 0
        for item in video_anno[vid]:
            query_time = int(item['realtime'] + 0.5)
            if frames is not None:
                # 1. Encode frames strictly before (or up to) query_time
                while frame_ptr + 1 < len(timestamps) and timestamps[frame_ptr + 1] <= query_time:
                    mem_log = report_gpu_memory(frame_ptr, print_ok=False)
                    to_frames = frames[frame_ptr: frame_ptr + 2]
                    model.visual_cache, stream_args, output_log = visual_cache_encode(model, processor, to_frames,
                                                                   use_system=False, stream_args=stream_args)
                    frame_ptr += 2
                    print('[Streaming]  ', mem_log, ' | ', output_log, flush=True)
            # 2. when reach to a timestamp, find all queries need to be answered
            try:
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
                            print(f'\t <before> [{frame_ptr}]', past_key_values.key_cache[0].shape, model.visual_cache.key_cache[0].shape,
                                  stream_tempt_args.input_ids.shape, flush=True)
                            ##### patch pixel value
                            dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
                            dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
                            dummy_inputs = processor(text=[dummy_text], videos=to_frames, padding=True,
                                                     return_tensors="pt", ).to(model.device)
                            pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
                            cur_video_embeds = model.visual(pixel_values_videos,  grid_thw=dummy_inputs['video_grid_thw'])
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
                            print(f'\t <middle> [{frame_ptr}]', past_key_values.key_cache[0].shape,
                                  model.visual_cache.key_cache[0].shape, stream_tempt_args.input_ids.shape, flush=True)
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
                                stream_tempt_args.input_ids = torch.index_select(stream_tempt_args.input_ids, 1,
                                                                                 indices_tensor)
                                assert stream_tempt_args.video_grid_thw[:, 0] == model.context_len + 1
                                assert stream_tempt_args.input_ids.shape[1] == model.visual_context_size

                            print(f'\t <after> [{frame_ptr}]', past_key_values.key_cache[0].shape,
                                  model.visual_cache.key_cache[0].shape, stream_tempt_args.input_ids.shape, flush=True)

                if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                    id, video, task, question, options, realtime, gt = \
                        item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], \
                        item['gt']
                    # if id !=1127: continue
                    prompt = build_prompt(
                        task=task,
                        question=question,
                        options=options,
                        _anno_=None,
                        index=None,
                    )

                    text_input = '<|vision_end|>' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
                    inputs = processor(text=[text_input], images=None, videos=None,
                                       padding=True, return_tensors='pt').to(model.device)

                    with torch.inference_mode():
                        generated_ids = model.generate(
                            input_ids=inputs['input_ids'],
                            max_new_tokens=16,
                            past_key_values=past_key_values,
                            use_cache=True,
                            streaming_args=stream_tempt_args.copy(),
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text[0]
                    print(chr(65 + gt), response)

                    output_dict = {
                        'id': id,
                        'video': video,
                        'task': task,
                        'question': question,
                        'response': response,
                        'ground_truth': chr(65 + gt),
                    }

                elif item['task'] in forward_tasks:
                    id, video, task, test_info = item['id'], item['video'], item['task'], item['test_info']
                    assert len(test_info) == 1, "only support one test"
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=0,
                    )

                    text_input = '<|vision_end|>' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
                    inputs = processor(text=[text_input], images=None, videos=None,
                                       padding=True, return_tensors='pt').to(model.device)


                    with torch.inference_mode():
                        generated_ids = model.generate(
                            input_ids=inputs['input_ids'],
                            max_new_tokens=16,
                            past_key_values=past_key_values,
                            use_cache=True,
                            streaming_args=stream_tempt_args.copy(),
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text[0]

                    item['test_info'][0]['response'] = response
                    output_dict = item

                with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')

            except Exception as e:
                logger.error(f"Error in processing {item}: {e}")

    end_time = time.time()
    cost_time = int(end_time - start_time)
    # Print results
    with open(output_jsonl, 'r') as f:
        lines = f.readlines()

    ### we need to map back to original anno format, as forward tasks are evalutae avg one instance, instead of overall
    final_dict = {}
    for line in lines:
        item = json.loads(line)
        id = item['id']
        if id not in final_dict:
            final_dict[id] = item
        else:
            assert item['task'] in forward_tasks
            assert len(item['test_info']) == 1
            final_dict[id]['test_info'].append(item['test_info'][0])

    items = [final_dict[j] for j in sorted(final_dict)]

    results = defaultdict(list)
    for item in items:
        # item = json.loads(line)
        if item['task'] in backward_tasks:
            results['backward'].append(item)
        elif item['task'] in realtime_tasks:
            results['realtime'].append(item)
        elif item['task'] in forward_tasks:
            results['forward'].append(item)
    score(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llava')
    parser.add_argument("--context_len", type=int, default=32)

    args = parser.parse_args()
    if 'llava' in args.model_name:
        llava_baseline(context_len=args.context_len)
    elif 'qwen' in args.model_name:
        qwen_baseline(context_len=args.context_len)
    else:
        raise NotImplementedError