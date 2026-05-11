'''
The code here is for exploring PE effect.
(1)  relative PE.
'''
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
import json, re
from datetime import datetime
import argparse
import sys
sys.path.append('ADD your path to Llava folder here')

from llava_ov.builder import load_pretrained_model, Instance
from llava_ov.llava_onevision import Llava_OneVision

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import ffmpeg


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


LOG_PATH = "log/{model}_{task}_{curr_time}.log"
TASK_CSV="../datasets/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="../datasets/StreamingBench/Real-Time Visual Understanding"
PRE_PROMPT = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
POST_PROMPT = "Answer with the option's letter from the given choices directly."


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


def llava_baseline(ckpt_path='lmms-lab/llava-onevision-qwen2-7b-ov', context_len= 32):
    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model = 'Llava_ov',task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    output_jsonl = f"results/llava_ov_{task}_{curr_time}.jsonl"

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
    task_df = pd.read_csv(TASK_CSV)

    generation_kwargs = {'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False, 'until': ['\n\n']}
    split = 'test'
    ## 1. first build instances
    requests = collections.defaultdict(list)
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        #try:
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
            start_time_sec = 0
            ### test for short-term
            fps = 1
            if time_stamp_sec > context_len: # set to 0, means just one frame
                start_time_sec = time_stamp_sec - context_len
            doc_to_visual = ([video_path], start_time_sec, time_stamp_sec, fps)

            ## doc_2_text, pre_prompt + question + post prompt
            option = "\n".join([f"{opt}" for i, opt in enumerate(eval(options))])
            question = question + "\n" + option
            ctx = PRE_PROMPT + "\n" + question + "\n" + POST_PROMPT

            ## construct_requests
            arguments = (ctx, generation_kwargs, doc_to_visual, doc_id, task, split  )
            kwargs = {'metadata': {"task": task, "doc_id": doc_id, "repeats": 1, "split": split}}
            inst = Instance(request_type='generate_until', arguments=arguments, idx=0, doc=doc, **kwargs ) #
            reqtype = inst.request_type
            requests[reqtype].append(inst)

        # except Exception as e:
        #     logger.error(f"Error in processing {row}: {e}")


    ## 2. generate response
    output_dict = []
    start_time = time.time()
    for reqtype, reqs in requests.items():
        # reqs = reqs[:15]
        eval_logger.info(f"Processing {reqtype} with {len(reqs)} requests")
        eval_logger.info("Running {} requests".format(reqtype))

        responses = getattr(model, reqtype)(reqs)
        logger.info(f"Totally get {len(responses)} responses")

        for req, response in zip(reqs, responses):
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


#################################################### uniform sampling
def split_video(video_file, start_time, end_time):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = "split_video/streamingbench"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")
    if os.path.exists(output_file):
        logger.debug(f"Video file {output_file} already exists.")
        return output_file
    try:
        (
            ffmpeg
            .input(video_file, ss=int(start_time))
            .output(output_file, t=(int(end_time) - int(start_time)), vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
    logger.debug(f"Video: {output_file} splitting completed.")
    return output_file

Qwen_prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}

Answer with the option's letter from the given choices directly."""

### use split video
def qwen_baseline_org(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct'):
    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 448 * 448
    MIN_FRAMES = 4
    MAX_FRAMES = 1016

    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model = 'Qwen25_vl_split', task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_split_{task}_{curr_time}.jsonl"

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
    task_df = pd.read_csv(TASK_CSV)

    # Inference
    start_time = time.time()
    output_dict = []
    kkk = 0
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        # kkk += 1
        # if kkk > 5:
        #     break
        try:
            question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
            video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)
            start_time_sec = 0
            fps = 1
            if time_stamp_sec > 32:
                start_time_sec = time_stamp_sec - 32
                # start_time_sec = max(time_stamp_sec - 120, 0)
                # fps = 2
            video_path = split_video(video_path, start_time_sec, time_stamp_sec)
            
            # if 300 < time_stamp_sec <= 600:
            #     fps = 0.5
            # elif time_stamp_sec > 600:
            #     fps = 0.2
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
                            "fps": fps
                        },
                        {
                            "type": "text",
                            "text": Qwen_prompt.format(question, '\n'.join(eval(options)))
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
            # print(response, flush=True)

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

            # remove video
            files = [os.path.join('split_video/streamingbench', f) for f in os.listdir('split_video/streamingbench')]
            if len(files) > 3:
                for file in files[3:]:
                    os.remove(file)

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

def qwen_baseline_standard(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct'):
    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 448 * 448
    MIN_FRAMES = 2
    MAX_FRAMES = 32

    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}.jsonl"

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
    task_df = pd.read_csv(TASK_CSV)

    # Inference
    start_time = time.time()
    output_dict = []
    kkk = 0
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        # kkk += 1
        # if kkk > 10:
        #     break
        try:
            question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
            video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)
            start_time_sec = 0
            fps = 1
            # if time_stamp_sec > 1:
            #     start_time_sec = time_stamp_sec - 1

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
                            "text": Qwen_prompt.format(question, '\n'.join(eval(options)))
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ## TODO: This implementation is somehow problematic. e.g. video fps 25, start 1s, end 7s,
            ## it returns [75 ~125] with num frames floor2(175-25 +1)//25 = 6, [25, 58, 92, 125], but we would expect
            # [ 25, 50, 75, 100, 125, 150, 175]. But anyway, it will trimmed to 6 frames for 3D conv.
            image_inputs, video_inputs = process_vision_info(messages)
            # assert len(video_inputs) == 1, 'Only one video is allowed'
            # start_f, end_f = int(start_time_sec * video_kwargs['fps'][0] + 0.5), int(time_stamp_sec * video_kwargs['fps'][0] + 0.5)
            # video_inputs = [video_inputs[0][start_f:end_f+1]]

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
            # print(response, flush=True)

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


#################################### sliding window

from qwen_vl_utils.vision_process import (smart_nframes, smart_resize)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
import numpy as np

def qwen_baseline(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct', context_len=32):
    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 448 * 448
    MIN_FRAMES = 2
    MAX_FRAMES = 32

    task = os.path.basename(TASK_CSV).replace(".csv", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}.jsonl"

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
    task_df = pd.read_csv(TASK_CSV)

    # Inference
    start_time = time.time()
    output_dict = []
    kkk = 0
    for row in tqdm(task_df.itertuples(), total=len(task_df)):
        # kkk += 1
        # if kkk > 10:
        #     break
        try:
            question_id, task_type, question, time_stamp, answer, options, frames_required, temporal_clue_type = \
                row.question_id, row.task_type, row.question, row.time_stamp, row.answer, row.options, row.frames_required, row.temporal_clue_type
            video_path = osp.join(VIDEO_DIR, f"sample_{question_id.split('_')[-2]}", "video.mp4")
            time_stamp_sec = time_to_seconds(time_stamp)
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
                            "text": Qwen_prompt.format(question, '\n'.join(eval(options)))
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
            # print(response, flush=True)

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
    parser.add_argument("--model_name", type=str, default='llava_ov')

    args = parser.parse_args()
    if 'llava' in args.model_name:
        llava_baseline()
    elif 'qwen' in args.model_name:
        qwen_baseline()
    else:
        raise NotImplementedError


