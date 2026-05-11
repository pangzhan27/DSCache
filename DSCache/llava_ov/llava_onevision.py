import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import sys

sys.path.append('./DSCache/llava_ov')

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
from transformers.models.qwen2.modeling_qwen2 import DynamicCache

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
            truncate_context: Optional[bool] = False,
            # whether to truncate the context in generation, set it False for LLaVA-1.6
            customized_config: Optional[str] = None,  # ends in json
            max_frames_num: Optional[int] = 32,
            mm_spatial_pool_stride: Optional[int] = 2,
            mm_spatial_pool_mode: Optional[str] = "bilinear",
            token_strategy: Optional[str] = "single",
            # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
            video_decode_backend: str = "decord",
            context_len: int = 8,
    ) -> None:
        self._device = torch.device(device)
        self.device_map = device_map
        llava_model_args = {"multimodal": True}
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
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained,
                                                                                                          None,
                                                                                                          model_name,
                                                                                                          device_map=self.device_map,
                                                                                                          **llava_model_args)  # torch_dtype="float32",
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained,
                                                                                                          None,
                                                                                                          model_name,
                                                                                                          device_map=self.device_map,
                                                                                                          **llava_model_args)

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
        from transformers.generation.utils import GenerationMixin
        from generate_opt import (ReID_Qwen2Attention_forward, ReID_Qwen2Model_forward, GenerationMixin_prepare_inputs_for_generation,
                                  LlavaQwenForCausalLM_get_initial_cache_position, LlavaQwenForCausalLM_forward,
                                  GenerationMixin_update_model_kwargs_for_generation)
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, Qwen2Model, LlavaQwenForCausalLM

        Qwen2Attention.forward = ReID_Qwen2Attention_forward
        Qwen2Model.forward = ReID_Qwen2Model_forward
        LlavaQwenForCausalLM._get_initial_cache_position = LlavaQwenForCausalLM_get_initial_cache_position
        GenerationMixin.prepare_inputs_for_generation = GenerationMixin_prepare_inputs_for_generation
        GenerationMixin._update_model_kwargs_for_generation = GenerationMixin_update_model_kwargs_for_generation
        LlavaQwenForCausalLM.forward = LlavaQwenForCausalLM_forward

        ## cache the system prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        split_index = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
        input_ids = input_ids[:split_index]
        position_ids = torch.arange(input_ids.shape[0], device=self.device).unsqueeze(0)
        output = self.model(input_ids.unsqueeze(0).to(self.device), past_key_values=None, use_cache=True, position_ids=position_ids)
        # <|im_start|>system\n You are a helpful assistant.<|im_end|> \n <|im_start|>user \n
        self.system_cache = output.past_key_values
        # self.visual_cache = None
        self.visual_context_size = 196 * (context_len + 1) + self.system_cache[0][0].shape[2]
        self.context_len = context_len
        a = 1

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
        pass

    def generate_until(self, requests):
        pass
