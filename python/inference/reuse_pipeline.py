import math
import torch
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache, StaticCache
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from rope_reposition import rotate_k_cache_rope, rotate_yarn_position_encoding
from compress_kv import compress_kv, QueryStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_for_reuse(
    tokenizer: AutoTokenizer,
    texts: List[str],
    keep_bos: bool = False,
    role: str | None = None,
    add_generation_prompt: bool = False
) -> Dict[str, torch.Tensor]:
    if role is not None: # 使用chat template进行tokenize
        texts = [
            tokenizer.apply_chat_template(
                [{"role": role, "content": text}], tokenize=False, 
                add_generation_prompt=add_generation_prompt, enable_thinking=False
            ) for text in texts
        ]
        if tokenizer.bos_token is not None:
            texts = [text[len(tokenizer.bos_token):] for text in texts]
    # tokenize
    kwargs = {"padding": True, "return_tensors": "pt", "return_attention_mask": True}
    try:
        inputs = tokenizer(texts, **kwargs)
    except Exception as e:
        texts = [text.encode("utf-8", errors="ignore").decode("utf-8") for text in texts]
        inputs = tokenizer(texts, **kwargs)
    # 去除 begin_of_sentence 特殊 Token
    if not keep_bos and inputs.input_ids.shape[1] > 1 and int(inputs.input_ids[0, 0]) == tokenizer.bos_token_id:
        inputs["input_ids"] = inputs.input_ids[:, 1:]
        inputs["attention_mask"] = inputs.attention_mask[:, 1:]
    return inputs


def prefill_kv_cache(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
) -> DynamicCache:
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    return outputs.past_key_values


@dataclass
class BatchedKVInstance:
    input_ids: List[torch.LongTensor]
    """
    Shape: Tuple(layer_num) * Tuple(2)(key or value) * List(batch_size) * Tensor(head_num, seq_len, head_size)
    """
    past_key_values: Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], ...]

    original_lengths: List[int]

    def unpack(self):
        return self.input_ids, self.past_key_values
    
    def stack(self, other):
        self.input_ids.extend(other.input_ids)
        for layer, other_layer in zip(self.past_key_values, other.past_key_values):
            for korv, other_korv in zip(layer, other_layer):
                korv.extend(other_korv)
        self.original_lengths.extend(other.original_lengths)
        return self
    
    def to(self, device: str) -> None:
        self.input_ids = [ids.to(device) for ids in self.input_ids]
        past_key_values = []
        for keys, values in self.past_key_values:
            keys = [k.to(device) for k in keys]
            values = [v.to(device) for v in values]
            past_key_values.append((keys, values))
        self.past_key_values = past_key_values


class LLMInference:
    def __init__(
        self, 
        model_name_or_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attn_impl: str = "flash_attention_2",
    ):
        """
        初始化模型和分词器
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 设备类型 (cuda/cpu)
        """
        self.attn_impl = attn_impl
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = self.model.config
        self.rope_theta = config.rope_theta
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

    def get_prefill_kv_cache(self, 
        texts: List[str],
        keep_bos: bool,
        role: str | None = None,
        compress_method: str | None = None,
    ) -> BatchedKVInstance:
        """
        获取一批文本prefill后的键值缓存
        
        Args:
            texts: 文本列表
            keep_bos: 是否保留begin_of_sentence特殊Token
            
        Returns:
            past_key_values: 包含所有层键值缓存的元组
        """
        enable_compress = compress_method is not None

        inputs = tokenize_for_reuse(self.tokenizer, texts, keep_bos=keep_bos, role=role).to(self.device)
        with QueryStorage(self.model, enabled=enable_compress) as query_storage:
            past_key_values = prefill_kv_cache(self.model, inputs)
            queries = query_storage.get_all_queries()

        seq_len = inputs.attention_mask.sum(dim=1).tolist()
        input_ids = [ids[:seql] for ids, seql in zip(inputs.input_ids, seq_len)]

        full_kv = []
        for layer_i, (key_states, value_states) in enumerate(past_key_values):
            layer_keys, layer_values = [], []
            batch_indices = []
            for seq_i, seq_l in enumerate(seq_len):
                key, value = key_states[seq_i, :, :seq_l], value_states[seq_i, :, :seq_l]
                if enable_compress:
                    compressed_len = math.ceil(seq_l / 4.0) # compress rate 4:1
                    key, value, indices = compress_kv(
                        compress_method, compressed_len,
                        queries[layer_i][seq_i, :seq_l], key, value,
                    )
                    batch_indices.append(indices)
                layer_keys.append(key)
                layer_values.append(value)
            full_kv.append((layer_keys, layer_values))
        if enable_compress:
            input_ids = [ids[indices] for ids, indices in zip(input_ids, batch_indices)]
        return BatchedKVInstance(input_ids, tuple(full_kv), seq_len)

    def decode_with_past_kv(
        self,
        query_text: str,
        system_prompt_kv: Optional[BatchedKVInstance] = None,
        precomputed_kv: Optional[BatchedKVInstance] = None,
        max_new_tokens: int = 512,
        return_kv: bool = False,
        role: str | None = "user",
    ) -> Union[str, Tuple[str, Tuple[Tuple[torch.Tensor, ...], ...]]]:
        """
        使用已有的键值缓存进行解码
        """
        # Unpack system_prompt_kv and precomputed_kv
        system_prompt_ids = None
        if system_prompt_kv is not None:
            system_prompt_ids, system_prompt_kv = system_prompt_kv.unpack()
        precomputed_ids, original_lengths = None, None
        if precomputed_kv is not None:
            original_lengths = precomputed_kv.original_lengths
            precomputed_ids, precomputed_kv = precomputed_kv.unpack()

        # 合并所有past key values
        past_key_values = self._merge_kv_caches(system_prompt_kv, precomputed_kv, original_lengths)
        if past_key_values is not None:
            past_ids = []
            if system_prompt_ids is not None:
                past_ids.extend(system_prompt_ids)
            if precomputed_ids is not None:
                past_ids.extend(precomputed_ids)
            past_ids = torch.cat(past_ids).unsqueeze(0)
            # we don't check this for the compressed KV case
            # assert past_ids.shape[1] == past_key_values[0][0][0].shape[1]
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        
        # 编码查询文本
        query_inputs = tokenize_for_reuse(
            self.tokenizer, [query_text], keep_bos=True, role=role, add_generation_prompt=True
        ).to(self.device)

        if past_key_values is not None:
            if query_inputs.input_ids[0, 0] == self.tokenizer.bos_token_id:
                query_inputs["input_ids"] = query_inputs.input_ids[:, 1:]
            query_inputs["input_ids"] = torch.cat([past_ids, query_inputs["input_ids"]], dim=1)
            query_inputs["attention_mask"] = torch.ones_like(query_inputs["input_ids"], dtype=torch.bool)
        
        # 获取查询文本的attention mask
        # if past_key_values is not None:
        #     attention_mask = self._prepare_attention_mask(past_key_values, query_inputs)
        #     query_inputs['attention_mask'] = attention_mask
        
        if past_key_values is None:
            past_key_values = DynamicCache()

        # 生成文本
        generated_outputs = self.model.generate(
            **query_inputs,
            past_key_values=past_key_values,
            # attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # do_sample=False,
            # temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        
        # 解码生成的文本（跳过查询部分）
        generated_tokens = generated_outputs[0][len(query_inputs.input_ids[0]):]
        generated_tokens = generated_tokens[:max_new_tokens] # force limited output
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if return_kv:
            return generated_text, past_key_values
        return generated_text

    def _merge_kv_caches(
        self,
        system_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        precomputed_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        original_lengths: List[int],
    ) -> Optional[Tuple[Tuple[torch.Tensor, ...], ...]]:
        """
        合并多个键值缓存
        
        Args:
            system_kv: 系统提示词KV缓存
            precomputed_kv: 预计算KV缓存
            
        Returns:
            合并后的KV缓存
        """
        if system_kv is None and precomputed_kv is None:
            return None
        
        merged_kv = []
        if precomputed_kv is None:
            for key, value in system_kv:
                merged_kv.append((key[0].unsqueeze(0), value[0].unsqueeze(0)))
            return tuple(merged_kv)
        
        # 合并两个KV缓存（在序列长度维度上拼接）
        # print("model rope theta", self.model.config.rope_theta)
        for system_layer, recomp_layer in zip(system_kv, precomputed_kv):
            layer_full_kv = []
            for kv_i in range(2):
                layer_kv = [system_layer[kv_i][0]]
                cumulative_kv_len = int(system_layer[kv_i][0].shape[1])
                for seqlen, recomp_layer_kv in zip(original_lengths, recomp_layer[kv_i]):
                    if kv_i == 0: # Rotate key cache to have correct RoPE
                        # print(f"cumulative_kv_len: {cumulative_kv_len}")
                        recomp_layer_kv = rotate_k_cache_rope(
                            recomp_layer_kv, 
                            cumulative_kv_len,
                            self.rope_theta,
                            self.rope_type,
                        )
                    cumulative_kv_len += seqlen
                    layer_kv.append(recomp_layer_kv)
                layer_full_kv.append(torch.cat(layer_kv, dim=1).unsqueeze(0))
            merged_kv.append(tuple(layer_full_kv))
        
        return tuple(merged_kv)

    def _prepare_attention_mask(
        self, 
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        current_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        准备attention mask，考虑past key values的长度
        
        Args:
            past_key_values: 过去的键值缓存
            current_inputs: 当前输入的字典
            
        Returns:
            完整的attention mask
        """
        current_attention_mask = current_inputs.get("attention_mask", None)
        
        if past_key_values is None:
            return current_attention_mask
        
        # 获取past key values的序列长度
        past_seq_length = past_key_values[0][0][0].shape[1]  # [batch, num_heads, seq_len, head_dim]
        
        # 创建past部分的attention mask（全1）
        batch_size = current_attention_mask.shape[0]
        past_attention_mask = torch.ones(
            (batch_size, past_seq_length), 
            dtype=current_attention_mask.dtype, 
            device=self.device
        )
        
        # 拼接past和current的attention mask
        full_attention_mask = torch.cat([past_attention_mask, current_attention_mask], dim=1)
        
        return full_attention_mask

    def _merge_selected_kv_layer(
        self,
        system_kv_layer: torch.Tensor, # (head_num, seq_len, head_size)
        precomputed_kv_layer: List[torch.Tensor], # [batch_size] * (head_num, seq_len, head_size)
        retained_masks: List[torch.BoolTensor],
        rotate: bool,
        original_lengths: List[int],
    ) -> torch.Tensor:
        retained_kv_list = [system_kv_layer]
        retained_length = int(system_kv_layer.shape[1])
        prefix_length = retained_length
        for seqlen, kv, mask in zip(original_lengths, precomputed_kv_layer, retained_masks):
            retained_kv = kv[:, mask]
            if rotate:
                retained_kv = rotate_k_cache_rope(retained_kv, prefix_length, self.rope_theta, self.rope_type)
            retained_kv_list.append(retained_kv)
            retained_length += int(torch.sum(mask))
            prefix_length += seqlen
        merged_kv = torch.cat(retained_kv_list, dim=1)
        assert retained_length == merged_kv.shape[1], f"{retained_length} != merged_kv length {merged_kv.shape[1]}"
        return merged_kv.unsqueeze(0) # (batch_size=1, head_num, seq_len, head_size)
    
    def _gen_vdiff_recompute_mask(
        self,
        recompute_ratio: float,
        system_kv: BatchedKVInstance,
        precomputed_kv: BatchedKVInstance,
        full_ids: torch.Tensor,
        compare_layer: int = 1,
    ) -> List[torch.Tensor]:
        # values shape (num_heads, seq_len, head_size)
        system_length: int = len(system_kv.input_ids[0])
        full_length: int = len(full_ids)
        context_length = full_length - system_length
        # get reference and reused values
        ref_values = self.model.model(
            full_ids.unsqueeze(0), use_cache=True
        ).past_key_values[compare_layer][1][0, :, system_length:]
        reused_values = self._merge_kv_caches(
            system_kv.past_key_values, precomputed_kv.past_key_values, precomputed_kv.original_lengths
        )[compare_layer][1][0, :, system_length:]
        # get values diff and top k indices
        values_diff = torch.mean((ref_values - reused_values) ** 2, [0, 2])
        num_selected_tokens = int(recompute_ratio * context_length)
        top_indices = torch.topk(values_diff, num_selected_tokens).indices
        # get recompute masks
        recompute_mask_full = torch.zeros(context_length, dtype=torch.bool, device=ref_values.device)
        recompute_mask_full[top_indices] = True
        recompute_masks = []
        for input_ids in precomputed_kv.input_ids:
            recompute_masks.append(recompute_mask_full[:len(input_ids)])
            recompute_mask_full = recompute_mask_full[len(input_ids):]
        return recompute_masks
    
    def selective_recompute(
        self,
        system_kv: BatchedKVInstance,
        precomputed_kv: BatchedKVInstance,
        recompute_method: str,
        discard_kv: bool = False
    ) -> BatchedKVInstance:
        # 0. 检查传入Tensor形状是否匹配
        full_input_ids = system_kv.input_ids + precomputed_kv.input_ids
        full_input_ids = torch.cat(full_input_ids, dim=0)
        full_kv_length = len(full_input_ids)

        if recompute_method.startswith("vdiff"):
            recompute_ratio = float(recompute_method.split('-')[1])
            recompute_masks = self._gen_vdiff_recompute_mask(recompute_ratio, system_kv, precomputed_kv, full_input_ids)
        else:
            recompute_masks = gen_recompute_mask(self.tokenizer, precomputed_kv, recompute_method)
        device = system_kv.past_key_values[0][0][0].device
        assert len(precomputed_kv.past_key_values[0][0]) == len(recompute_masks)

        recompute_masks = [mask.bool() for mask in recompute_masks]
        retained_masks = [~mask for mask in recompute_masks]

        # 1. 从precomputed_kv中去除重算部分，再拼接
        retained_kv = []
        for (sys_key, sys_val), (pre_key, pre_val) in zip(system_kv.past_key_values, precomputed_kv.past_key_values):
            retained_kv.append((
                self._merge_selected_kv_layer(sys_key[0], pre_key, retained_masks, True, precomputed_kv.original_lengths),
                self._merge_selected_kv_layer(sys_val[0], pre_val, retained_masks, False, precomputed_kv.original_lengths),
            ))

        if discard_kv:
            system_kv.past_key_values = ()
            precomputed_kv.past_key_values = ()

        retained_kv_length = int(retained_kv[0][0].shape[2])
        retained_kv = DynamicCache.from_legacy_cache(tuple(retained_kv))

        # 2. Concatenate ids to recompute + prepare customized attention_mask
        recompute_length = full_kv_length - retained_kv_length
        prefix_length = len(system_kv.input_ids[0])
        position_counter = prefix_length
        rearranged_index = torch.arange(full_kv_length, dtype=torch.long, device=device)
        # print(f"prefix_length: {prefix_length}, full_kv_length: {full_kv_length}, recompute_length: {recompute_length}")
        attention_mask_past = torch.zeros((recompute_length, retained_kv_length), dtype=torch.long, device=device)
        attention_mask_tril = torch.tril(torch.ones((recompute_length, recompute_length), dtype=torch.long, device=device))
        
        recompute_ids = []
        recompute_counter = 0
        for ids, mask in zip(precomputed_kv.input_ids, recompute_masks):
            recompute_ids.append(ids[mask])
            for masked in mask:
                if masked:
                    rearranged_index[retained_kv_length + recompute_counter] = position_counter
                    attention_mask_past[recompute_counter, :prefix_length] = 1
                    recompute_counter += 1
                else:
                    rearranged_index[prefix_length] = position_counter
                    prefix_length += 1
                position_counter += 1

        assert recompute_counter == recompute_length, \
            f"recompute_counter {recompute_counter} != recompute_length {recompute_length}"
        assert prefix_length == retained_kv_length
        recompute_ids = torch.cat(recompute_ids, dim=0)
        attention_mask = torch.cat([attention_mask_past, attention_mask_tril], dim=-1).bool()

        # 3. 调整输入形状，准备推理
        recompute_ids = recompute_ids.unsqueeze(0) # (batch_size, query_len)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # (batch_size, head_size, query_len, kv_length)
        position_ids = rearranged_index[-recompute_length:].unsqueeze(0) # (batch_size, query_len)

        # 4. 推理
        self.model.model.config._attn_implementation = "sdpa"
        output = self.model(
            recompute_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=retained_kv,
            use_cache=True,
        )
        self.model.model.config._attn_implementation = self.attn_impl

        # 5. 重新排列KV Cache
        assert isinstance(output.past_key_values, DynamicCache)
        rearranged_index = rearranged_index.squeeze(0)
        ret_kv = []
        for output_key, output_value in output.past_key_values:
            output_key, output_value = output_key[0], output_value[0] # only 1 sequence
            key = torch.empty_like(output_key)
            value = torch.empty_like(output_value)
            ret_kv.append((
                [key.index_copy_(1, rearranged_index, output_key)],
                [value.index_copy_(1, rearranged_index, output_value)]
            ))
        
        original_lengths = len(system_kv.original_lengths + precomputed_kv.original_lengths)
        return BatchedKVInstance(
            input_ids=[full_input_ids],
            past_key_values=tuple(ret_kv),
            original_lengths=[original_lengths]
        )


def gen_recompute_mask(
    tokenizer: AutoTokenizer,
    batch: BatchedKVInstance, 
    recompute_type: str,
) -> List[torch.Tensor]:
    recompute_masks = []
    if recompute_type.startswith("leading"):
        leading_num = int(recompute_type.split('-')[1])
        for cache in batch.past_key_values[0][0]:
            mask = torch.zeros((cache.shape[1],), dtype=torch.bool, device=cache.device)
            mask[:leading_num] = 1
            recompute_masks.append(mask)
    elif recompute_type.startswith("pctg"):
        leading_num = int(recompute_type.split('-')[1])
        for cache in batch.past_key_values[0][0]:
            prefix_length = max(4, int(cache.shape[1] * leading_num / 100))
            mask = torch.zeros((cache.shape[1],), dtype=torch.bool, device=cache.device)
            mask[:prefix_length] = 1
            recompute_masks.append(mask)
    elif recompute_type.startswith("punc"):
        leading_num = int(recompute_type.split('-')[1])
        punctuations = set([ch for ch in string.punctuation])
        for input_ids in batch.input_ids:
            mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
            mask[:leading_num] = 1
            for idx, token_id in enumerate(input_ids):
                token = tokenizer.decode(token_id)
                if token[0] in punctuations or token[-1] in punctuations:
                    mask[idx] = 1
            recompute_masks.append(mask)
    else:
        raise ValueError(f"Unrecognized recompute type {recompute_type}")
    return recompute_masks
