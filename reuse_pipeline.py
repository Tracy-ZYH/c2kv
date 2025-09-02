import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache, StaticCache
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from rope_reposition import rotate_k_cache_rope, rotate_yarn_position_encoding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchedKVInstance:
    input_ids: List[torch.LongTensor]
    """
    Shape: Tuple(layer_num) * Tuple(2)(key or value) * List(batch_size) * Tensor(head_num, seq_len, head_size)
    """
    past_key_values: Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], ...]

    def unpack(self):
        return self.input_ids, self.past_key_values
    
    def stack(self, other):
        self.input_ids.extend(other.input_ids)
        for layer, other_layer in zip(self.past_key_values, other.past_key_values):
            for korv, other_korv in zip(layer, other_layer):
                korv.extend(other_korv)
        return self


class LLMInference:
    def __init__(self, model_name_or_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化模型和分词器
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 设备类型 (cuda/cpu)
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"模型加载完成，设备: {device}")

    def get_prefill_kv_cache(self, 
        texts: List[str],
        keep_bos: bool,
    ) -> BatchedKVInstance:
        """
        获取一批文本prefill后的键值缓存
        
        Args:
            texts: 文本列表
            keep_bos: 是否保留begin_of_sentence特殊Token
            
        Returns:
            past_key_values: 包含所有层键值缓存的元组
        """
        # 编码文本
        try:
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                return_tensors="pt", 
                return_attention_mask=True
            ).to(self.device)
        except Exception as e:
            inputs = self.tokenizer(
                [text.encode("utf-8", errors="ignore").decode("utf-8") for text in texts], 
                padding=True, 
                return_tensors="pt", 
                return_attention_mask=True
            ).to(self.device)

        if not keep_bos and inputs.input_ids[0, 0] == self.tokenizer.bos_token_id:
            inputs["input_ids"] = inputs.input_ids[:, 1:]
            inputs["attention_mask"] = inputs.attention_mask[:, 1:]
        
        # 前向传播获取键值缓存
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True
            )
        
        # 返回past_key_values (包含所有层的key和value)
        seq_len = [sum(seq_attn.cpu().tolist()) for seq_attn in inputs.attention_mask]
        input_ids = [ids[:seql] for ids, seql in zip(inputs.input_ids, seq_len)]

        full_kv = []
        for layer in outputs.past_key_values:
            layer_kv = []
            for kv_i in range(2):
                korv = []
                for seq_i, seq_l in enumerate(seq_len):
                    korv.append(layer[kv_i][seq_i][:, :seq_l])
                layer_kv.append(korv)
            full_kv.append(tuple(layer_kv))
        # return tuple(full_kv), input_ids
        return BatchedKVInstance(input_ids, tuple(full_kv))

    def decode_with_past_kv(
        self,
        query_text: str,
        system_prompt_kv: Optional[BatchedKVInstance] = None,
        precomputed_kv: Optional[BatchedKVInstance] = None,
        max_new_tokens: int = 512,
        return_kv: bool = False,
    ) -> Union[str, Tuple[str, Tuple[Tuple[torch.Tensor, ...], ...]]]:
        """
        使用已有的键值缓存进行解码
        """
        # Unpack system_prompt_kv and precomputed_kv
        system_prompt_ids = None
        if system_prompt_kv is not None:
            system_prompt_ids, system_prompt_kv = system_prompt_kv.unpack()
        precomputed_ids = None
        if precomputed_kv is not None:
            precomputed_ids, precomputed_kv = precomputed_kv.unpack()

        # 合并所有past key values
        past_key_values = self._merge_kv_caches(system_prompt_kv, precomputed_kv)
        if past_key_values is not None:
            past_ids = []
            if system_prompt_ids is not None:
                past_ids.extend(system_prompt_ids)
            if precomputed_ids is not None:
                past_ids.extend(precomputed_ids)
            past_ids = torch.cat(past_ids).unsqueeze(0)
            assert past_ids.shape[1] == past_key_values[0][0][0].shape[1]
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        else:
            past_key_values = None
        
        # 编码查询文本
        query_inputs = self.tokenizer(
            query_text, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(self.device)

        if past_key_values is not None:
            if query_inputs.input_ids[0, 0] == self.tokenizer.bos_token_id:
                query_inputs.input_ids = query_inputs.input_ids[:, 1:]
            query_inputs["input_ids"] = torch.cat([past_ids, query_inputs.input_ids], dim=1)
            query_inputs["attention_mask"] = torch.ones_like(query_inputs["input_ids"])
        
        # 获取查询文本的attention mask
        # if past_key_values is not None:
        #     attention_mask = self._prepare_attention_mask(past_key_values, query_inputs)
        #     query_inputs['attention_mask'] = attention_mask
        
        if past_key_values is None:
            past_key_values = DynamicCache()

        # 生成文本
        with torch.no_grad():
            generated_outputs = self.model.generate(
                **query_inputs,
                past_key_values=past_key_values,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=0.1,
                # top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # 解码生成的文本（跳过查询部分）
        generated_tokens = generated_outputs[0][len(query_inputs.input_ids[0]):]
        generated_tokens = generated_tokens[:max_new_tokens] # force limited output
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if return_kv:
            return generated_text, past_key_values.to_legacy_cache()
        return generated_text

    def _merge_kv_caches(
        self, 
        system_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
        precomputed_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]
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
                for recomp_layer_kv in recomp_layer[kv_i]:
                    if kv_i == 0: # Rotate key cache to have correct RoPE
                        # print(f"cumulative_kv_len: {cumulative_kv_len}")
                        recomp_layer_kv = rotate_k_cache_rope(
                            recomp_layer_kv, 
                            cumulative_kv_len,
                            self.model.config.rope_theta,
                        )
                    cumulative_kv_len += int(recomp_layer_kv.shape[1])
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
    ) -> torch.Tensor:
        retained_kv_list = [system_kv_layer]
        retained_length = int(system_kv_layer.shape[1])
        prefix_length = retained_length
        for kv, mask in zip(precomputed_kv_layer, retained_masks):
            retained_kv = kv[:, mask]
            if rotate:
                retained_kv = rotate_k_cache_rope(retained_kv, prefix_length, self.model.config.rope_theta)
            retained_kv_list.append(retained_kv)
            retained_length += int(torch.sum(mask))
            prefix_length += int(kv.shape[1])
        merged_kv = torch.cat(retained_kv_list, dim=1)
        assert retained_length == merged_kv.shape[1], f"{retained_length} != merged_kv length {merged_kv.shape[1]}"
        return merged_kv.unsqueeze(0) # (batch_size=1, head_num, seq_len, head_size)
    
    def selective_recompute(
        self,
        system_kv: BatchedKVInstance,
        precomputed_kv: BatchedKVInstance,
        recompute_masks: List[torch.BoolTensor],
    ) -> BatchedKVInstance:
        # 0. 检查传入Tensor形状是否匹配
        device = system_kv.past_key_values[0][0][0].device
        assert len(precomputed_kv.past_key_values[0][0]) == len(recompute_masks)

        full_input_ids = system_kv.input_ids + precomputed_kv.input_ids
        full_input_ids = torch.cat(full_input_ids, dim=0)
        full_kv_length = len(full_input_ids)

        recompute_masks = [mask.bool() for mask in recompute_masks]
        retained_masks = [~mask for mask in recompute_masks]

        # print(system_kv.input_ids)
        # print(precomputed_kv.input_ids)

        # 1. 从precomputed_kv中去除重算部分，再拼接
        past_kv = []
        for (sys_key, sys_val), (pre_key, pre_val) in zip(system_kv.past_key_values, precomputed_kv.past_key_values):
            past_kv.append((
                self._merge_selected_kv_layer(sys_key[0], pre_key, retained_masks, True),
                self._merge_selected_kv_layer(sys_val[0], pre_val, retained_masks, False),
            ))

        retained_kv_length = int(past_kv[0][0].shape[2])
        retained_kv = DynamicCache.from_legacy_cache(tuple(past_kv))

        # 2. Concatenate ids to recompute + prepare customized attention_mask
        recompute_length = full_kv_length - retained_kv_length
        prefix_length = len(system_kv.input_ids[0])
        position_counter = prefix_length
        rearranged_index = torch.arange(full_kv_length, dtype=torch.long, device=device)
        print(f"prefix_length: {prefix_length}, full_kv_length: {full_kv_length}, recompute_length: {recompute_length}")
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
        # print(torch.sum(attention_mask[:, :-10], dim=-1))
        # print(attention_mask[:, -15:])
        # print(recompute_ids)
        # print(rearranged_index)

        # 3. 调整输入形状，准备推理
        recompute_ids = recompute_ids.unsqueeze(0) # (batch_size, query_len)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # (batch_size, head_size, query_len, kv_length)
        position_ids = rearranged_index[-recompute_length:].unsqueeze(0) # (batch_size, query_len)

        # 4. 推理
        output = self.model(
            recompute_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=retained_kv,
            use_cache=True,
        )

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
        
        return BatchedKVInstance(
            input_ids=[full_input_ids],
            past_key_values=tuple(ret_kv),
        )
