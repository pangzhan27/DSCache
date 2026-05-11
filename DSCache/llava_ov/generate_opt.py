from torch import nn
import os, types
import torch
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.generation.utils import  is_torchdynamo_compiling, ModelOutput, ALL_CACHE_NAMES


from transformers.models.qwen2.modeling_qwen2 import (deprecate_kwarg, add_start_docstrings_to_model_forward, replace_return_docstrings,
                                                      QWEN2_INPUTS_DOCSTRING, CausalLMOutputWithPast,_CONFIG_FOR_DOC, Unpack, KwargsForCausalLM,
                                                      FlashAttentionKwargs, logger, DynamicCache, GenerationMixin, BaseModelOutputWithPast)
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation.utils import  Cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union





############################################ for full attention with truncated attention mask #####################
def LlavaQwenForCausalLM_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if inputs_embeds is None:
        assert images is None, "only support all image(via embedding) or all text(via ids), not mixed ones"

    return super(LlavaQwenForCausalLM, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def LlavaQwenForCausalLM_get_initial_cache_position(self, input_ids, model_kwargs):
    """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
    cumulate_length = [key.shape[2] for key in model_kwargs['past_key_values'].key_cache]  # self.config.cumulate_length
    # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
    if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
        new_len = model_kwargs["inputs_embeds"].shape[1]
        cache_position = [torch.arange(k + new_len, dtype=torch.int64, device=input_ids.device)
                          for k in cumulate_length]
        if not is_torchdynamo_compiling():
            cache_position = [cache_position[j][cumulate_length[j]:] for j in range(len(cumulate_length))]
        # cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
    elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
        cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        )
        raise NotImplementedError
    else:
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        raise NotImplementedError

    model_kwargs["cache_position"] = cache_position
    return model_kwargs



def GenerationMixin_prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
):
    """
    Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
    slicing inputs given the existing cache.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """

    # 1. Handle BC:
    model_inputs = {}
    # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    elif cache_position is None:
        raise ValueError('Cache position must be specified')

    ## cache position represents the length of input_ids, should be with same length
    cash_len = [k.shape[0] for k in cache_position]
    assert len(set(cash_len)) == 1

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position[0].shape[0]:]
        elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[0][-1] >= input_ids.shape[1])  # Exception 3
        ):
            input_ids = input_ids[:, -cache_position[0].shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            raise ValueError('not implemented')
            input_ids = input_ids[:, cache_position]

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and len(cache_position[0]) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    attention_mask = ( kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask)
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"

    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask
        #TODO: re id postions that are masked which is not happend in our case

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs


def GenerationMixin_update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
) -> Dict[str, Any]:
    # update past_key_values keeping its naming used in model code
    for possible_cache_name in ALL_CACHE_NAMES:
        if possible_cache_name in outputs:
            # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
            if possible_cache_name in ("past_buckets_states", "mems"):
                cache_name = "past_key_values"
            else:
                cache_name = possible_cache_name
            model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
            break

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = [ k[-1:] + num_new_tokens for k in model_kwargs["cache_position"]]
        if isinstance(model_kwargs["position_ids"], list):
            model_kwargs["position_ids"] = [ torch.cat([k, torch.tensor([[k[0, -1] + num_new_tokens]],
                                                                        dtype = k.dtype, device=k.device)], dim=-1)
                                             for k in model_kwargs["position_ids"]]
        else:
            k = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = torch.cat([k, torch.tensor([[k[0, -1] + num_new_tokens]],
                                                                      dtype = k.dtype, device=k.device)], dim=-1)
    else:
        raise ValueError('Not supported')
    return model_kwargs

############################################ for ReID #################################################################
def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        past_seen_tokens: int,
):
    if self.config._attn_implementation == "flash_attention_2":
        raise ValueError('Not supported')

    #past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    assert isinstance(past_key_values, DynamicCache), 'other type of cache is not supported'
    using_static_cache = False #isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = False #isinstance(past_key_values, SlidingWindowCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    # SlidingWindowCache or StaticCache
    if using_sliding_window_cache or using_static_cache:
        raise ValueError('not supported')
        # target_length = past_key_values.get_max_cache_shape()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        config=self.config,
        past_key_values=past_key_values,
    )

    if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask




@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def ReID_Qwen2Model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    assert position_ids is not None, "MUST provide position_ids"
    if not isinstance(position_ids, list):
        position_ids = [position_ids for _ in range(self.config.num_hidden_layers)]
    ## build necessities for each layer
    cache_position, causal_mask, position_embeddings = [], [], []
    for layer_idx in range(self.config.num_hidden_layers):
        is_empty_layer = (
                len(past_key_values.key_cache) == 0  # no cache in any layer
                or len(past_key_values.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
                or len(past_key_values.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = past_key_values.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        cache_pos = torch.arange(layer_seq_length, layer_seq_length + inputs_embeds.shape[1], device=inputs_embeds.device)
        attent_mask = torch.ones(layer_seq_length + inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        c_mask = _update_causal_mask(self, attent_mask, inputs_embeds, cache_pos, past_key_values, output_attentions, layer_seq_length)
        # create position embeddings to be shared across the decoder layers
        position_embed = self.rotary_emb(inputs_embeds, position_ids[layer_idx])

        cache_position.append(cache_pos)
        causal_mask.append(c_mask)
        position_embeddings.append(position_embed)


    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()

from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward, ALL_ATTENTION_FUNCTIONS, rotate_half

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    if q.shape[2] != k.shape[2]:
        q_cos = cos[:, :, -q.shape[2]:]
        q_sin = sin[:, :, -q.shape[2]:]
        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    return q_embed, k_embed

def ReID_Qwen2Attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings[self.layer_idx] #@P
        #print('save before adding PE!!!!!', flush=True)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position[self.layer_idx]}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask[self.layer_idx],
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights