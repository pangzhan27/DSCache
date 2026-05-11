import os, json, torch, re
import copy
import os.path as osp
from datetime import datetime
##################################### Streaming Bench#################################################

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

##################################### OVOBench#################################################
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


def score(results, logger):
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

def prepare_system_cache(model, processor, stream_args):
    messages = [{"role": "user", "content": [{"type": "video", "video": '', }, ], }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = processor.tokenizer(text, padding=True, return_tensors="pt", )['input_ids'][0]
    split_index = (input_ids == model.config.video_token_id).nonzero(as_tuple=True)[0][0].item()
    input_ids = input_ids[:split_index].unsqueeze(0).to(model.device)
    #position_ids = torch.arange(input_ids.shape[1], device=model.device).unsqueeze(0) #, position_ids=position_ids
    output = model(input_ids, past_key_values=None, use_cache=True, streaming_args=stream_args.copy())
    stream_args.input_ids = torch.cat((stream_args.input_ids, input_ids), dim=-1)

    return output.past_key_values, stream_args



########################################## helper function ##################################
def report_gpu_memory(tag="", print_ok=True):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    if print_ok:
        print(f"[{tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB | peak={peak:.1f} MB", flush=True)
    return f"[{tag}] allocated={allocated:.1f} MB | reserved={reserved:.1f} MB | peak={peak:.1f} MB"

def resume_used_ovobench(video_anno, output_jsonl, file=''):
    ### file represents the jsonl file of previous results
    assert os.path.exists(file)
    with open(file, 'r') as f:
        lines = f.readlines()

    finished_videos = []
    for line in lines:
        item = json.loads(line)
        if item['video'] not in finished_videos:
            finished_videos.append(item['video'])
    # remove the last one(unsure if its is fully done)
    finished_videos = finished_videos[:-1]

    # rewrite finished results
    for line in lines:
        output_dict = json.loads(line)
        if output_dict['video'] in finished_videos:
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
    # remove the videos from dict
    left_anno = {}
    for vid in video_anno.keys():
        if vid not in finished_videos:
            left_anno[vid] = video_anno[vid]

    print('Finished videos:', len(finished_videos), 'Left:', len(left_anno))

    return left_anno


def resume_used_streambench(video_anno, output_jsonl, file=''):
    ### file represents the jsonl file of previous results
    assert os.path.exists(file)
    with open(file, 'r') as f:
        lines = f.readlines()

    finished_videos = []
    for line in lines:
        item = json.loads(line)
        video = f"sample_{item['question_id'].split('_')[-2]}"
        if video not in finished_videos:
            finished_videos.append(video)
    # remove the last one(unsure if its is fully done)
    finished_videos = finished_videos[:-1]

    # rewrite finished results
    for line in lines:
        output_dict = json.loads(line)
        video = f"sample_{output_dict['question_id'].split('_')[-2]}"
        if video in finished_videos:
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
    # remove the videos from dict
    left_anno = {}
    for vid in video_anno.keys():
        pure_vid = vid.split('/')[-2]
        if pure_vid not in finished_videos:
            left_anno[vid] = video_anno[vid]

    print('Finished videos:', len(finished_videos), 'Left:', len(left_anno))

    return left_anno


def resume_used_rvs(video_anno, output_jsonl, file=''):
    ### file represents the jsonl file of previous results
    assert os.path.exists(file)
    with open(file, 'r') as f:
        lines = f.readlines()

    finished_videos = []
    for line in lines:
        item = json.loads(line)
        if item['video_id'] not in finished_videos:
            finished_videos.append(item['video_id'])
    # remove the last one(unsure if its is fully done)
    finished_videos = finished_videos[:-1]
    finished_videos_full = ["../datasets/RVS/ego/videos/" + v for v in finished_videos]

    # rewrite finished results
    for line in lines:
        output_dict = json.loads(line)
        if output_dict['video_id'] in finished_videos:
            with open(output_jsonl, 'a' if osp.exists(output_jsonl) else 'w') as f:
                f.write(json.dumps(output_dict) + '\n')
    # remove the videos from dict
    left_anno = {}
    for vid in video_anno.keys():
        
        if vid not in finished_videos_full:
            left_anno[vid] = video_anno[vid]

    print('Finished videos:', len(finished_videos), 'Left:', len(left_anno))

    return left_anno


########################################### cache function ##################################
def contiguous_kv(past_key_values):
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        past_key_values.key_cache[i] = k_layer.contiguous()
        past_key_values.value_cache[i] = v_layer.contiguous()
    return past_key_values



def clone_dynamic_cache(cache):
    if cache is None:
        return None
    # shallow copy container
    new_cache = copy.copy(cache)
    # clone all key/value tensors
    new_cache.key_cache = [k.detach().clone() for k in cache.key_cache]
    new_cache.value_cache = [v.detach().clone() for v in cache.value_cache]
    return new_cache

def cache_eviction(past_key_values, start):
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        seq_len = k_layer.shape[2]
        indices_tensor = torch.tensor(list(range(start, seq_len)), device=k_layer.device)
        past_key_values.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
        past_key_values.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)
    del k_layer, v_layer
    torch.cuda.empty_cache()
    return past_key_values


