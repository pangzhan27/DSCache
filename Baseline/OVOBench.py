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
sys.path.append('ADD your path to Llava folder here')

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

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import ffmpeg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

LOG_PATH = "log/{model}_{task}_{curr_time}.log"
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


def get_response(model, doc):
    id, visual, task, context, time_stamp_sec, gt, context_len = doc
    time_stamp_sec = int(time_stamp_sec+0.5)
    start_time_sec = 0
    fps = 1
    if time_stamp_sec > context_len:
        start_time_sec = time_stamp_sec - context_len

    gen_kwargs = {'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False}

    origin_image_aspect_ratio = getattr(model._config, "image_aspect_ratio", None)
    if origin_image_aspect_ratio is not None and model._config.image_aspect_ratio != origin_image_aspect_ratio:
        model._config.image_aspect_ratio = origin_image_aspect_ratio
        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

    assert type(visual) == str
    image_tensor = []
    try:
        if model.video_decode_backend == "decord":
            frames = model.load_video_span(visual, model.max_frames_num, start_time_sec, time_stamp_sec, fps=fps)
            # frames = self.load_video_ls(visual, start_time, end_time, long=32, fps=target_fps, l_fps=1/2)
        elif model.video_decode_backend == "pyav":
            raise ValueError('Unsupported backend')
            # frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
        frames = model._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(model.device)
        image_tensor.append(frames)
    except Exception as e:
        eval_logger.error(f"Error {e} in loading video")
        image_tensor = None

    task_type = "video"
    placeholder_count = len(frames) if model.token_strategy == "multiple" else 1

    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
        image_tokens = " ".join(image_tokens)
        question = image_tokens + "\n" + context #
    else:
        question = context

    # This is much safer for llama3, as we now have some object type in it
    if "llama_3" in model.conv_template:
        conv = copy.deepcopy(conv_templates[model.conv_template])
    else:
        conv = conv_templates[model.conv_template].copy()

    question_input = []
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    question_input.append(prompt_question)

    input_ids_list = [tokenizer_image_token(prompt, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for
                      prompt in question_input]
    pad_token_ids = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    input_ids = model.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(model.device)
    attention_masks = input_ids.ne(pad_token_ids).to(model.device)

    if task_type == "image":
        raise ValueError('not Supported')
    elif task_type == "video":
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

    with torch.inference_mode():
        cont = model.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids,
                                    images=image_tensor, use_cache=model.use_cache, **gen_kwargs)

    text_outputs = model.tokenizer.batch_decode(cont, skip_special_tokens=True)

    text_outputs = [response.strip() for response in text_outputs]

    return text_outputs[0]


def llava_baseline(ckpt_path='lmms-lab/llava-onevision-qwen2-7b-ov'):
    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Llava_ov', task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    context_len = 32

    output_jsonl = f"results/llava_ov_{task}_{curr_time}_{context_len}.jsonl"

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    #
    model = Llava_OneVision(pretrained=ckpt_path,
                            conv_template='qwen_1_5',
                            model_name='llava_qwen',
                            attn_implementation='sdpa',
                            device_map='cuda',
                            device='cuda',
                            )

    logger.info(f"Load model and processor from {ckpt_path}")

    # Load task info
    with open(TASK_CSV, 'r') as f:
        task_list = json.load(f)

    start_time = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item[
                        'gt']
                #if id !=1127: continue
                prompt = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )
                video_path = osp.join(VIDEO_DIR, video)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)
                
                doc = [id, video_path, task, prompt, realtime, gt, context_len]
                response = get_response(model=model, doc=doc)

                output_dict = {
                    'id': id,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                }

            elif item['task'] in forward_tasks:
                id, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                for i in range(len(test_info)):
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=i,
                    )
                    realtime = test_info[i]['realtime']
                    video_path = osp.join(VIDEO_DIR, video)
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(video_path)
                    doc = [id, video_path, task, prompt, realtime, gt, context_len]
                    response = get_response(model=model, doc=doc)
                    item['test_info'][i]['response'] = response

                output_dict = item

            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')

        except Exception as e:
            logger.error(f"Error in processing {item}: {e}")

    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results
    results = defaultdict(list)
    with open(output_jsonl, 'r') as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        if item['task'] in backward_tasks:
            results['backward'].append(item)
        elif item['task'] in realtime_tasks:
            results['realtime'].append(item)
        elif item['task'] in forward_tasks:
            results['forward'].append(item)
    score(results)


MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MIN_FRAMES = 2
MAX_FRAMES = 320

def get_response_qwen_standard(model, doc, processor):
    id, video_path, task, context, time_stamp_sec, gt, context_len = doc
    start_time_sec = 0
    fps = 1
    if time_stamp_sec > context_len:
        start_time_sec = time_stamp_sec - context_len

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "max_frames": MAX_FRAMES,
                    "min_frames": MIN_FRAMES,
                    "video_start": start_time_sec,
                    "video_end": time_stamp_sec,
                    "fps": fps
                },
                {
                    "type": "text",
                    "text": context
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(torch.device('cuda'))
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    return response


from qwen_vl_utils.vision_process import (smart_nframes, smart_resize)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import numpy as np

### use manually selected frame index
def get_response_qwen(model, doc, processor):
    id, video_path, task, context, time_stamp_sec, gt, context_len = doc
    start_time_sec = 0
    fps = 1
    if time_stamp_sec > context_len:
        start_time_sec = time_stamp_sec - context_len

    vr = VideoReader(video_path, ctx=cpu(0))
    height, width, _ = vr.next().shape
    resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS,
                                                 max_pixels=MAX_PIXELS)

    ##### read the whole video
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    orig_fps = vr.get_avg_fps()
    duration = len(vr) / vr.get_avg_fps()
    timestamps = np.arange(start_time_sec, min(time_stamp_sec + 1.0 / fps, duration), 1.0 / fps)
    frame_indices = (timestamps * orig_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frame_num - 1)
    unique_indices = np.unique(frame_indices)
    try:
        frames = torch.from_numpy(vr.get_batch(unique_indices).asnumpy()).permute(0, 3, 1, 2)
        frames = transforms.functional.resize(frames, [resized_height, resized_width],
                                              interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
    except Exception as e:
        eval_logger.error(f"Error {e} in loading video")
        frames = None


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "max_frames": MAX_FRAMES,
                    "min_frames": MIN_FRAMES,
                    "video_start": start_time_sec,
                    "video_end": time_stamp_sec,
                    "fps": fps
                },
                {
                    "type": "text",
                    "text": context
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames.to(model.device)] if frames is not None else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(torch.device('cuda'))
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    return response



def qwen_baseline(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct'):
    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    context_len = 32
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}_{context_len}.jsonl"

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

    # Load task info
    with open(TASK_CSV, 'r') as f:
        task_list = json.load(f)

    start_time = time.time()
    for item in tqdm(task_list):
        try:
            if item['task'] in backward_tasks or item['task'] in realtime_tasks:
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item[
                        'gt']
                # if id !=1127: continue
                prompt = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )
                video_path = osp.join(VIDEO_DIR, video)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)
                
                doc = [id, video_path, task, prompt, realtime, gt, context_len]
                response = get_response_qwen(model=model, doc=doc, processor=processor)

                output_dict = {
                    'id': id,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                }

            elif item['task'] in forward_tasks:
                id, video, task, test_info = \
                    item['id'], item['video'], item['task'], item['test_info']
                for i in range(len(test_info)):
                    prompt = build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=i,
                    )
                    realtime = test_info[i]['realtime']
                    video_path = osp.join(VIDEO_DIR, video)
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(video_path)
                    
                    doc = [id, video_path, task, prompt, realtime, gt, context_len]
                    response = get_response_qwen(model=model, doc=doc, processor=processor)
                    item['test_info'][i]['response'] = response

                output_dict = item

            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')

        except Exception as e:
            logger.error(f"Error in processing {item}: {e}")

    end_time = time.time()
    cost_time = int(end_time - start_time)

    # Print results
    results = defaultdict(list)
    with open(output_jsonl, 'r') as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
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

    args = parser.parse_args()
    if 'llava' in args.model_name:
        llava_baseline()
    elif 'qwen' in args.model_name:
        qwen_baseline()
    else:
        raise NotImplementedError