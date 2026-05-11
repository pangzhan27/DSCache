import torch
from helper import clone_dynamic_cache

@torch.no_grad()
def visual_cache_encode_llava(self, frames, use_system=False):
    output_log = ''
    # 1. get frame token feature
    frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)
    assert (frames.shape[-1]) == 384 and (frames.shape[-2] == 384)
    encoded_image_features = self.model.encode_images(frames)
    video_features = self.model.get_2dPool(encoded_image_features)
    video_features = video_features[-1:]
    assert video_features.shape[1] == 196

    # 2. feed into token memory (short)
    if self.token_mem is None:
        self.token_mem = video_features.clone()
    else:
        self.token_mem = torch.cat((self.token_mem, video_features.clone()), 0)

    # 3. decide whether evict token mem and create new cache for long term
    if self.token_mem.shape[0] > self.frame_len:
        output_log += 'Frame eviction. '
        evict_video_features = self.token_mem[:1]
        self.token_mem = self.token_mem[1:]
        assert self.token_mem.shape[0] == self.frame_len, "Once filled, the size should be consistent with frame_len"

        position_ids = torch.arange(self.visual_cache[0][0].shape[2] + evict_video_features.shape[1], device=self.device).unsqueeze(0)
        output = self.model(inputs_embeds=evict_video_features, past_key_values=clone_dynamic_cache(self.visual_cache),
                            use_cache=True, position_ids=position_ids)
        self.visual_cache = output.past_key_values

    # 4. decide wheter evict long-term cache
    if self.visual_cache[0][0].shape[2] > self.visual_context_size:
        output_log += 'Cache eviction. '
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
        assert self.visual_cache[0][0].shape[2] == self.visual_context_size, 'Cache size mismatch'

    ### the caching phase, all cache should be the same length. different cache length should only happen in generation,
    # and its process should not affect the existing cache and will be discarded once used
    visual_cache_len = [self.visual_cache[i][0].shape[2] for i in range(len(self.visual_cache))]
    assert len(set(visual_cache_len)) == 1, "Cache with different length"

    return output_log


### long_cache selection
def sample_long_cache(past_key_values, sample_rate, frame_token = 196):
    block_num = past_key_values[0][0].shape[-2] // frame_token
    pe_id = torch.arange(past_key_values[0][0].shape[-2],device=past_key_values[0][0].device)
    pe_id = pe_id.reshape(block_num, frame_token)
    if block_num < sample_rate:
        return None, None
    PE_ids = []
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        bz, head, length, dim = k_layer.shape
        k_layer_tmpt = k_layer.reshape(bz, head, block_num, frame_token, dim)
        v_layer_tmpt = v_layer.reshape(bz, head, block_num, frame_token, dim)
        start = (block_num - sample_rate) % sample_rate
        indices_tensor = torch.tensor(list(range(start, block_num, sample_rate)), device=k_layer.device)
        selected = indices_tensor.shape[0]
        past_key_values.key_cache[i] = torch.index_select(k_layer_tmpt, 2, indices_tensor).reshape(bz, head, selected * frame_token, dim)
        past_key_values.value_cache[i] = torch.index_select(v_layer_tmpt, 2, indices_tensor).reshape(bz, head, selected * frame_token, dim)
        PE_ids.append(torch.index_select(pe_id, 0, indices_tensor).reshape(1, selected * frame_token))
        a = PE_ids[0].cpu().numpy()
    del k_layer, v_layer, k_layer_tmpt, v_layer_tmpt
    torch.cuda.empty_cache()
    return past_key_values, PE_ids

def concat_dynamic_caches(caches):
    """
    caches: list of dynamicCache, each is a list of (k_layer, v_layer)
    returns: one dynamicCache
    """
    num_layers = len(caches[0].key_cache)
    new_cache = clone_dynamic_cache(caches[0])
    for layer_idx in range(num_layers):
        k_cat = torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=2)  # concat along sequence length
        v_cat = torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=2)
        new_cache.key_cache[layer_idx] = k_cat
        new_cache.value_cache[layer_idx] = v_cat
    return new_cache


#### for qwen
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

@torch.no_grad()
def visual_cache_encode(model, processor, video_inputs, stream_args, use_system=False):
    output_log = ''
    ##### here we save raw frame instead of feature because 2 frames  generate one feature. the order of frame affects which
    ##### adjacent frames will be used for feature extraction
    # 2. feed into token memory (short)
    if model.token_mem is None:
        model.token_mem = video_inputs.clone()
    else:
        model.token_mem = torch.cat((model.token_mem, video_inputs.clone()), 0)

    # 3. decide whether evict token mem and create new cache for long term
    if model.token_mem.shape[0] > model.frame_len:
        output_log += 'Frame eviction. '
        # 3.1 evict frame
        to_evict = model.token_mem.shape[0] - model.frame_len
        evict_video_features = model.token_mem[:to_evict]
        model.token_mem = model.token_mem[to_evict:]

        assert model.token_mem.shape[0] == model.frame_len, 'make sure every evicition is well-behaved'

        # 3.2 cache for the evicited frame for the earliest two frames.
        if model.wait2cache is None:
            output_log += '[first cache]'
            model.wait2cache = evict_video_features.clone()
        else:
            output_log += '[lag cache]'
            model.wait2cache = torch.cat([model.wait2cache, evict_video_features], 0)

        if model.wait2cache.shape[0] >= 2:
            to_cache = model.wait2cache[:2]
            model.wait2cache = model.wait2cache[2:]

            ## caching
            dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
            dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
            dummy_inputs = processor(text=[dummy_text], videos=to_cache, padding=True, return_tensors="pt", ).to(
                model.device)
            pixel_values_videos = dummy_inputs['pixel_values_videos'].type(model.visual.dtype)
            cur_video_embeds = model.visual(pixel_values_videos, grid_thw=dummy_inputs['video_grid_thw'])

            ## santity check
            num_token_per_frame = torch.sum(dummy_inputs['input_ids'] == TOKEN_IDS['<|video_pad|>']).item()
            assert model.num_token_per_frame == num_token_per_frame, 'mismatch frame resolution'
            assert cur_video_embeds.shape[0] == num_token_per_frame, 'mismatch frame resolution'

            ## prepare stream_args
            video_pad = torch.ones((1, model.num_token_per_frame), device=stream_args.input_ids.device,
                                   dtype=stream_args.input_ids.dtype) * TOKEN_IDS['<|video_pad|>']
            assert stream_args.input_ids.shape[1] == model.visual_cache[0][0].shape[2]
            assert stream_args.video_grid_thw[:, 0] == (
                        stream_args.input_ids.shape[1] - model.system_cache[0][0].shape[2]) // model.num_token_per_frame
            stream_args.input_ids = torch.cat((stream_args.input_ids, video_pad), dim=-1)
            stream_args.video_grid_thw[:, 0] = stream_args.video_grid_thw[:, 0] + 1  # update one frame

            ## get cache, if the frames is uniform samples, use the default second_per_grid_ts as None.
            output = model(inputs_embeds=cur_video_embeds.unsqueeze(0), past_key_values=model.visual_cache,
                           use_cache=True,  streaming_args=stream_args.copy())

            model.visual_cache = output.past_key_values

        output_log += f'{model.wait2cache.shape}'
        assert model.wait2cache.shape[0] < 2, "temporal frames should be less than 2"
        # create cache for evicted frame

    # print(stream_args.input_ids.shape, stream_args.video_grid_thw, flush=True)
    # 4. decide wheter evict long-term cache
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

    # print(stream_short_args.input_ids.shape[1], model.token_mem.shape, flush=True)
    assert stream_args.input_ids.shape[1] == model.visual_cache[0][0].shape[2]

    return model.visual_cache, stream_args, output_log

@torch.no_grad()
def initialize_model_param(model, processor, stream_args, video_inputs):
    if stream_args.video_grid_thw is None:
        ##### 1. get frame token features
        dummy_inputs = [{'role': 'user', 'content': [{'type': 'video', 'video': ''}]}]
        dummy_text = processor.apply_chat_template(dummy_inputs, tokenize=False, add_generation_prompt=False)
        dummy_inputs = processor(text=[dummy_text], videos=video_inputs, padding=True, return_tensors="pt", ).to(model.device)
        num_token_per_frame = torch.sum(dummy_inputs['input_ids'] == TOKEN_IDS['<|video_pad|>']).item() // \
                              dummy_inputs['video_grid_thw'][0, 0].item()

        model.num_token_per_frame = num_token_per_frame
        model.visual_context_size = (model.context_len + 1) * num_token_per_frame + model.system_cache[0][0].shape[2]
        print('[', model.video_name, '] Num token per frame :', model.num_token_per_frame, model.visual_context_size, flush=True)
        stream_args.video_grid_thw = dummy_inputs['video_grid_thw']
        stream_args.video_grid_thw[:, 0] = 0
    return stream_args


#########################################################for spatial ####################################################

@torch.no_grad()
def visual_cache_encode_llava_spatial(self, frames, stride, use_system=False):
    output_log = ''
    # 1. get frame token feature
    frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device) #
    assert (frames.shape[-1]) == 384 and (frames.shape[-2] == 384)
    encoded_image_features = self.model.encode_images(frames)
    self.last_frame = self.model.get_2dPool(encoded_image_features, stride=stride)
    video_features = self.model.get_2dPool(encoded_image_features)
    video_features = video_features[-1:]
    assert video_features.shape[1] == 196

    # 2. feed into token memory (short)
    if self.token_mem is None:
        self.token_mem = video_features.clone()
    else:
        self.token_mem = torch.cat((self.token_mem, video_features.clone()), 0)

    # 3. decide whether evict token mem and create new cache for long term
    if self.token_mem.shape[0] > self.frame_len:
        output_log += 'Frame eviction. '
        evict_video_features = self.token_mem[:1]
        self.token_mem = self.token_mem[1:]
        assert self.token_mem.shape[0] == self.frame_len, "Once filled, the size should be consistent with frame_len"

        position_ids = torch.arange(self.visual_cache[0][0].shape[2] + evict_video_features.shape[1], device=self.device).unsqueeze(0)
        output = self.model(inputs_embeds=evict_video_features, past_key_values=clone_dynamic_cache(self.visual_cache),
                            use_cache=True, position_ids=position_ids)
        self.visual_cache = output.past_key_values

    # 4. decide wheter evict long-term cache
    if self.visual_cache[0][0].shape[2] > self.visual_context_size:
        output_log += 'Cache eviction. '
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
        assert self.visual_cache[0][0].shape[2] == self.visual_context_size, 'Cache size mismatch'

    ### the caching phase, all cache should be the same length. different cache length should only happen in generation,
    # and its process should not affect the existing cache and will be discarded once used
    visual_cache_len = [self.visual_cache[i][0].shape[2] for i in range(len(self.visual_cache))]
    assert len(set(visual_cache_len)) == 1, "Cache with different length"

    return output_log
