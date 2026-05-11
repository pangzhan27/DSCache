# increase the last frame resolution
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

from llava_ov.helper import (time_to_seconds, extract_characters_regex, clone_dynamic_cache, cache_eviction,
                             report_gpu_memory, resume_used_streambench, prepare_system_cache, resume_used_ovobench,
                             build_prompt, score)
from llava_ov.long_function import (visual_cache_encode_llava_spatial, sample_long_cache, concat_dynamic_caches,
                                    visual_cache_encode, initialize_model_param, TOKEN_IDS)

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


def get_response(model, doc, frames, ltype, continuous):
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

    # new_line
    final_past_key_values = clone_dynamic_cache(model.system_cache)
    final_position_ids = torch.arange(final_past_key_values[0][0].shape[2] + input_ids.shape[1],
                                      device=model.device).unsqueeze(0)

    with torch.inference_mode():
        if frames:
            # 1. get short-term cache
            # 1.1 get the short-term(exclude the last frame)
            #print(frame_ptr, query_time)
            image_tensor = model.token_mem[:-1].flatten(0, 1)
            position_ids = torch.arange(model.system_cache[0][0].shape[2] + image_tensor.shape[0],
                                        device=model.device).unsqueeze(0)
            output = model.model(inputs_embeds=image_tensor.unsqueeze(0), position_ids=position_ids,
                                 past_key_values=clone_dynamic_cache(model.system_cache), use_cache=True)
            tempt_past_key_values = output.past_key_values

            # 1.2 include the last frame(with high resolution)
            last_tensor = model.last_frame.flatten(0, 1)
            tempt_line = model.model.model.image_newline[None].to(model.model.device).unsqueeze(0)
            last_tensor = torch.cat((last_tensor.unsqueeze(0), tempt_line), 1)
            spatial_num = last_tensor.shape[1]
            position_ids_new = torch.arange(tempt_past_key_values[0][0].shape[2] + last_tensor.shape[1],
                                        device=model.device).unsqueeze(0)
            output = model.model(inputs_embeds=last_tensor, position_ids=position_ids_new,
                                 past_key_values=tempt_past_key_values, use_cache=True)
            short_past_key_values = output.past_key_values

            # 2. prepare for the final cache and id
            if (ltype == 'default') or model.visual_cache[0][0].shape[2] <= model.system_cache[0][0].shape[2]:
                print('**Only use short term**', flush=True)
                final_past_key_values = short_past_key_values
                final_position_ids = torch.arange(final_past_key_values[0][0].shape[2] + input_ids.shape[1],
                                                  device=model.device).unsqueeze(0)
            else:
                print('**Long term is used**', flush=True)
                #### NOTE: does not support yet
                # 2.1. remove the system_cache from short-term
                short_past_key_values = cache_eviction(short_past_key_values, model.system_cache[0][0].shape[2])
                assert (short_past_key_values[0][0].shape[2] - spatial_num) % 196 == 0, 'should be dividable(short)'

                # 2.2. get long-term cache
                ## should not change self.visual_cache in case of later usage; visual_cache should have the same length every layer
                long_past_key_values = cache_eviction(clone_dynamic_cache(model.visual_cache),
                                                      model.system_cache[0][0].shape[2])
                assert long_past_key_values[0][0].shape[2] % 196 == 0, 'should be dividable(long)'
                l_sys, l_long = model.system_cache[0][0].shape[2], long_past_key_values[0][0].shape[2]
                l_short = short_past_key_values[0][0].shape[2]

                # 2.3. process long_cache
                if 'ls' in ltype:
                    sample_rate = int(ltype.split('_')[-1])
                    if sample_rate > 1:
                        long_past_key_values, long_pe = sample_long_cache(clone_dynamic_cache(long_past_key_values),
                                                                          sample_rate, 196)
                    else:  # using the whole long-term cache
                        long_pe = [torch.arange(l_long, device=model.device).unsqueeze(0) for i in
                                   range(len(model.system_cache.key_cache))]
                else:
                    pass

                # 2.4. concate final cache
                if long_past_key_values is None:
                    final_past_key_values = concat_dynamic_caches(
                        [clone_dynamic_cache(model.system_cache), short_past_key_values])
                    final_position_ids = torch.arange(final_past_key_values[0][0].shape[2] + input_ids.shape[1],
                                                      device=model.device).unsqueeze(0)
                else:
                    final_past_key_values = concat_dynamic_caches(
                        [clone_dynamic_cache(model.system_cache), long_past_key_values, short_past_key_values])

                    pos_sys = torch.arange(l_sys, device=model.device).unsqueeze(0)
                    ###### masked out to change PE
                    if continuous:
                        l_long = long_past_key_values[0][0].shape[2]
                        long_pe = [torch.arange(l_long, device=model.device).unsqueeze(0) for i in
                                   range(len(model.system_cache.key_cache))]
                    #####
                    pos_long = [p + l_sys for p in long_pe]
                    pos_short = torch.arange(l_sys + l_long, l_sys + l_long + l_short + input_ids.shape[1],
                                             device=model.device).unsqueeze(0)
                    final_position_ids = [torch.cat([pos_sys, p, pos_short], dim=1) for p in pos_long]
                    a = final_position_ids[0].cpu().numpy()

                    if final_position_ids[0][0, -1] == (final_position_ids[0].shape[1] - 1):
                        print('[Reassign continuous position ID]', flush=True)
                    else:
                        print('[Keep relative position ID]')

        cont = model.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids,
                                    position_ids=final_position_ids,
                                    past_key_values=final_past_key_values, use_cache=model.use_cache, **gen_kwargs)

    text_outputs = model.tokenizer.batch_decode(cont, skip_special_tokens=True)

    text_outputs = [response.strip() for response in text_outputs]

    return text_outputs[0]


def llava_baseline(ckpt_path='lmms-lab/llava-onevision-qwen2-7b-ov', context_len=32, frame_len=4, ltype='dynamic',
                   continuous=True, stride=2):
    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    post_fix = '_cont_pe' if continuous else ''
    to_type = f'{context_len}-{frame_len}_s{stride}_' + ltype + post_fix
    # Add file handler
    os.makedirs('log', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Llava_ov', task=task, curr_time=curr_time, type=to_type))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    target_fps = 1

    output_jsonl = f"results/llava_ov_{task}_{curr_time}_{to_type}.jsonl"
    print(output_jsonl, flush=True)

    # Load model and processor
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    #
    Llava_OneVision.visual_cache_encode = visual_cache_encode_llava_spatial
    model = Llava_OneVision(pretrained=ckpt_path,
                            conv_template='qwen_1_5',
                            model_name='llava_qwen',
                            attn_implementation='sdpa',
                            device_map='cuda',
                            device='cuda',
                            context_len=context_len,
                            )

    model.frame_len = frame_len + 1
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
    for vid in tqdm(video_anno.keys()):
        print(vid, 'Spatial', flush=True)
        video_path = osp.join(VIDEO_DIR, vid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        # for each query, reset
        model.visual_cache = clone_dynamic_cache(model.system_cache)
        model.token_mem = None  # reset frame tokens
        model.last_frame = None

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
            logger.error(f"Error in loading video {vid}")
            continue

        frame_ptr = 0
        for item in video_anno[vid]:
            query_time = int(item['realtime'] + 0.5)
            if frames is not None:
                # 1. Encode frames strictly before (or up to) query_time
                while frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                    mem_log = report_gpu_memory(frame_ptr, print_ok=False)
                    output_log = model.visual_cache_encode(frames[frame_ptr:frame_ptr + 1], stride=stride, use_system=False)
                    #print(mem_log, ' | ', output_log, flush=True)
                    frame_ptr += 1

            assert model.system_cache[0][0].shape[2] == 14, "system cached changes"
            assert model.last_frame is not None, 'should at least include frame_ptr = 0'
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
                    response = get_response(model=model, doc=doc, frames=True if frames is not None else False,
                                            ltype=ltype, continuous=continuous)

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
                    response = get_response(model=model, doc=doc, frames=True if frames is not None else False,
                                            ltype=ltype, continuous=continuous)
                    item['test_info'][0]['response'] = response

                    output_dict = item

                # print(id, response)
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
    score(results, logger)


# ------------------------------------------------------------------------------------------------------------------------
from qwen_vl_utils.vision_process import (smart_nframes, smart_resize)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from llava_ov.utils import StreamingArgs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu
import numpy as np


def qwen_baseline(ckpt_path='Qwen/Qwen2.5-VL-7B-Instruct', target_fps=1, context_len=32, frame_len=4, ltype='dynamic',
                  continuous=True, stride=2):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLSdpaAttention, Qwen2_5_VLModel
    from llava_ov.generate_qwen_op import (Qwen2_5_VLSdpaAttention_forward, Qwen2_5_VLForConditionalGeneration_forward,
                                           Qwen2_5_VLModel_forward, _get_initial_cache_position,
                                           prepare_inputs_for_generation, get_rope_index_multiple,
                                           _update_model_kwargs_for_generation)
    Qwen2_5_VLSdpaAttention.forward = Qwen2_5_VLSdpaAttention_forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward
    Qwen2_5_VLForConditionalGeneration._get_initial_cache_position = _get_initial_cache_position
    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation
    Qwen2_5_VLForConditionalGeneration._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
    Qwen2_5_VLForConditionalGeneration.get_rope_index = get_rope_index_multiple

    MIN_PIXELS = 448 * 448
    MAX_PIXELS = 448 * 448

    task = os.path.basename(TASK_CSV).replace(".json", "")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    post_fix = '_cont_pe' if continuous else ''
    to_type = f'{context_len}-{frame_len}_s{stride}_' + ltype + post_fix
    # Add file handler
    os.makedirs('log', exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH.format(model='Qwen25_vl', task=task, curr_time=curr_time, type=to_type))
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    output_jsonl = f"results/qwen25_vl_{task}_{curr_time}_{to_type}.jsonl"
    # print(output_jsonl, flush=True)

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
        values = sorted(values, key=lambda x: x['realtime'])
        video_anno[vid] = values
        total_num += len(values)

    #### resume from previous
    video_anno = resume_used_ovobench(video_anno, output_jsonl, file = 'results/qwen25_vl_ovo_bench_new_20260111_155129_12-2_s2.0_ls_2_cont_pe.jsonl')

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
    processor_last = AutoProcessor.from_pretrained(
        ckpt_path,
        min_pixels=MIN_PIXELS*stride*stride,
        max_pixels=MAX_PIXELS*stride*stride,
    )

    logger.info(f"Load model and processor from {ckpt_path}")

    stream_args_org = StreamingArgs(model.model.device)
    ### prepare system cache
    system_cache, stream_args_org = prepare_system_cache(model, processor, stream_args_org)
    model.system_cache = system_cache
    model.context_len = context_len
    model.frame_len = frame_len + 1

    # Inference
    start_time = time.time()
    k = 0
    for vid in tqdm(video_anno.keys()):
        print(vid, 'Spatial', flush=True)
        # if k < 502:
        #     k=k+1
        #     continue
        # each video may have different resolution. so reset context window length for each video
        if hasattr(model, "num_token_per_frame"):
            delattr(model, "num_token_per_frame")
        model.video_name = vid
        # for each query, reset
        model.visual_cache = clone_dynamic_cache(model.system_cache)
        model.token_mem = None  # reset frame tokens
        model.wait2cache = None
        model.last_frame = None
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
        #height, width, _ = vr.next().shape
        resized_height, resized_width = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS,
                                                     max_pixels=MAX_PIXELS)
        # resized_height_last, resized_width_last = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS*stride*stride,
        #                                              max_pixels=MAX_PIXELS*stride*stride)

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
                frames = torch.from_numpy(vr.get_batch(unique_indices).asnumpy()).permute(0, 3, 1, 2)
                if frames is not None:
                    break
            except Exception as e:
                frames = None

        if frames is None:
            logger.error(f"Error in loading video {vid}")
            continue

        if frames is not None:
            init_frame = transforms.functional.resize(frames[:2], [resized_height, resized_width],
                                         interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
            stream_args = initialize_model_param(model, processor, stream_args, init_frame)

        frame_ptr = 0
        # if frames is not None: visual_cache_encode(model, processor, frames[:2], use_system=False, stream_args=stream_args)
        for item in video_anno[vid]:
            query_time = int(item['realtime'] + 0.5)
            # print('use int end time + dual', flush=True)
            use_visual, semi_token_num = False, None
            if frames is not None:
                use_visual = True
                start = max(0, query_time - frame_len)
                if start > timestamps[-1]:
                    use_visual = False
                    print('--------------------- exceed --------------', vid, query_time, flush=True)
                elif query_time > timestamps[-1]:
                    semi_token_num = len(timestamps) - start
                    print('--------------------- semi_exceed --------------', vid, query_time, flush=True)

                offline_frames = transforms.functional.resize(frames[start:query_time + 1], [resized_height, resized_width],
                                                      interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
                # 1. Encode frames strictly before (or up to) query_time
                while frame_ptr < len(timestamps) and timestamps[frame_ptr] <= query_time:
                    mem_log = report_gpu_memory(frame_ptr, print_ok=False)
                    if frame_ptr + 1 < len(timestamps) and timestamps[frame_ptr + 1] <= query_time:
                        inc = 2
                        to_end = frame_ptr + 2
                    else:
                        inc = 1
                        to_end = frame_ptr + 1

                    model.last_frame = frames[to_end-1:to_end].float()
                    # model.last_frame = transforms.functional.resize(frames[to_end-1:to_end], [int(resized_height_last), int(resized_width_last)],
                    #                                               interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
                    to_frames = frames[frame_ptr: to_end]
                    to_frames = transforms.functional.resize(to_frames, [resized_height, resized_width],
                                                                  interpolation=InterpolationMode.BICUBIC, antialias=True, ).float()
                    model.visual_cache, stream_args, output_log = visual_cache_encode(model, processor, to_frames,
                                                                                      use_system=False, stream_args=stream_args)
                    #print(mem_log, ' | ', output_log, flush=True)
                    frame_ptr += inc

            # 2. when reach to a timestamp, find all queries need to be answered
            final_past_key_values = clone_dynamic_cache(model.system_cache)
            stream_final_args = stream_args_org.copy()
            second_per_grid_ts = None
            if (frames is not None) and (use_visual):
                with torch.no_grad():
                    # print(f'[{query_time}]', offline_frames.shape, model.token_mem.shape, torch.equal(offline_frames, model.token_mem) )
                    if semi_token_num is not None:
                        print('-------- ', vid, query_time, flush=True)
                        model.token_mem = model.token_mem[-semi_token_num:]
                    assert torch.equal(offline_frames, model.token_mem), "********************************short-term should match offline version"

                    # 1. get temporal short-term cache
                    # get feature
                    #print(frame_ptr, query_time)
                    dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
                    dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
                    prev_past_key_values = clone_dynamic_cache(model.system_cache)
                    stream_prev_args = stream_args_org.copy()
                    if len(model.token_mem) > 1:
                        dummy_inputs = processor(text=[dummy_text], videos=model.token_mem[:-1], padding=True, return_tensors="pt", ).to(model.device)
                        pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
                        cur_video_embeds = model.visual(pixel_values_videos, grid_thw=dummy_inputs['video_grid_thw'])
                        # get stream args
                        stream_prev_args = stream_args_org.copy()
                        num_frame = dummy_inputs['video_grid_thw'][0, 0]
                        video_pad = torch.ones((1, model.num_token_per_frame * num_frame), device=stream_args.input_ids.device,
                                           dtype=stream_args.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
                        stream_prev_args.input_ids = torch.cat((stream_prev_args.input_ids, video_pad), dim=-1)
                        stream_prev_args.video_grid_thw = dummy_inputs['video_grid_thw']
                        # get short-term cache
                        output = model(inputs_embeds=cur_video_embeds.unsqueeze(0),
                                   past_key_values=clone_dynamic_cache(model.system_cache),
                                   use_cache=True, streaming_args=stream_prev_args.copy())
                        prev_past_key_values = output.past_key_values
                    
                    
                    ## 1.2 prepare the lastest frame
                    last_inputs = processor_last(text=[dummy_text], videos=model.last_frame, padding=True, return_tensors="pt", ).to(model.device)
                    last_video_embeds = model.visual(last_inputs['pixel_values_videos'].type(model.visual.dtype),
                                                    grid_thw=last_inputs['video_grid_thw'])
                    # get stream args
                    spatial_num = last_video_embeds.shape[0]
                    stream_short_args = stream_args_org.copy()
                    last_video_pad = torch.ones((1, last_video_embeds.shape[0]), device=stream_args.input_ids.device, dtype=stream_args.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
                    stream_short_args.input_ids = torch.cat((stream_prev_args.input_ids, last_video_pad), dim=-1)
                    stream_short_args.video_grid_thw = last_inputs['video_grid_thw']
                    if len(model.token_mem) > 1:
                        stream_short_args.video_grid_thw = torch.cat((stream_prev_args.video_grid_thw, last_inputs['video_grid_thw']), dim=0)
                    # get short-term cache
                    output = model(inputs_embeds=last_video_embeds.unsqueeze(0), past_key_values=clone_dynamic_cache(prev_past_key_values),
                                   use_cache=True, streaming_args=stream_short_args.copy())
                    short_past_key_values = output.past_key_values

                    # 2. prepare for the final cache and id
                    if (ltype == 'default') or model.visual_cache[0][0].shape[2] <= model.system_cache[0][0].shape[2]:
                        print('Not use long!!', flush=True)
                        final_past_key_values = short_past_key_values
                        stream_final_args = stream_short_args.copy()
                        second_per_grid_ts = None
                    else:
                        print('Use long!!', flush=True)
                        #TODO: not implemented
                        # 2.1. remove the system_cache from short-term
                        short_past_key_values = cache_eviction(short_past_key_values, model.system_cache[0][0].shape[2])
                        num_short = ((short_past_key_values[0][0].shape[2] - spatial_num) // model.num_token_per_frame) + 1
                        assert (short_past_key_values[0][0].shape[2] - spatial_num) % model.num_token_per_frame == 0, 'should be dividable(short)'

                        # 2.2. get long-term cache
                        ## should not change self.visual_cache in case of later usage; visual_cache should have the same length every layer
                        long_past_key_values = cache_eviction(clone_dynamic_cache(model.visual_cache),
                                                              model.system_cache[0][0].shape[2])
                        assert long_past_key_values[0][0].shape[
                                   2] % model.num_token_per_frame == 0, 'should be dividable(long)'

                        # 2.3. process long_cache
                        if 'ls' in ltype:
                            sample_rate = int(ltype.split('_')[-1])
                            if sample_rate > 1:
                                long_past_key_values, _ = sample_long_cache(clone_dynamic_cache(long_past_key_values),
                                                                            sample_rate, model.num_token_per_frame)
                                # the concatenation part of short and long is the same length as sample rate
                                if long_past_key_values is not None:
                                    num_long = long_past_key_values[0][0].shape[2] // model.num_token_per_frame
                                    second_per_grid_ts = [sample_rate] * num_long
                                    assert num_short > 0, "when cache is available, means buffer is full"
                                    if num_short > 0:
                                        second_per_grid_ts = [sample_rate] * num_long + [1] * (num_short - 1)
                                else:
                                    print('not enough cache to sample', frame_ptr, model.visual_cache[0][0].shape,
                                          flush=True)

                            else:  # using the whole long-term cache
                                second_per_grid_ts = None
                        else:
                            pass

                        # 2.4. concate final cache
                        if long_past_key_values is None:
                            final_past_key_values = concat_dynamic_caches(
                                [clone_dynamic_cache(model.system_cache), short_past_key_values])
                            second_per_grid_ts = None
                        else:
                            final_past_key_values = concat_dynamic_caches(
                                [clone_dynamic_cache(model.system_cache), long_past_key_values, short_past_key_values])
                            ###### masked out to change PE
                            if continuous:
                                second_per_grid_ts = None

                        stream_final_args = stream_short_args.copy()
                        to_pad = final_past_key_values[0][0].shape[2] - model.system_cache[0][0].shape[2]
                        assert (to_pad - spatial_num) % model.num_token_per_frame == 0, 'should be dividable(final)'
                        num_final_frame = (to_pad - spatial_num) // model.num_token_per_frame
                        video_pad = torch.ones((1, to_pad), device=stream_args_org.input_ids.device,
                                               dtype=stream_args_org.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
                        stream_final_args.input_ids = torch.cat((stream_args_org.input_ids, video_pad), dim=-1)
                        stream_final_args.video_grid_thw[0, 0] = num_final_frame

            try:
                if second_per_grid_ts is not None:
                    print('[Keep relative position ID]')
                # sanity check
                assert final_past_key_values.key_cache[0].shape[2] == stream_final_args.input_ids.shape[1]

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
                            past_key_values=final_past_key_values,
                            use_cache=True,
                            streaming_args=stream_final_args.copy(),
                            second_per_grid_ts=second_per_grid_ts,
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text[0]
                    # print(prompt, response)

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
                            past_key_values=final_past_key_values,
                            use_cache=True,
                            streaming_args=stream_final_args.copy(),
                            second_per_grid_ts=second_per_grid_ts,
                        )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    response = output_text[0]

                    item['test_info'][0]['response'] = response
                    output_dict = item

                # print(id, response, flush=True)
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
    score(results, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llava')
    parser.add_argument("--context_len", type=int, default=32)  # long, if long < sample rate, then just short
    parser.add_argument("--frame_len", type=int, default=4)  # short
    parser.add_argument("--ltype", type=str, default='default')
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--stride", type=float, default=1.0)

    args = parser.parse_args()
    if 'llava' in args.model_name:
        llava_baseline(context_len=args.context_len, frame_len=args.frame_len, ltype=args.ltype,
                       continuous=args.continuous, stride = args.stride)
    elif 'qwen' in args.model_name:
        qwen_baseline(context_len=args.context_len, frame_len=args.frame_len, ltype=args.ltype,
                      continuous=args.continuous, stride = args.stride)
    else:
        raise NotImplementedError
