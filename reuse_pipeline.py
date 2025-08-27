import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from typing import List, Tuple, Dict, Any, Optional
import logging


from rope_reposition import rotate_k_cache_rope, rotate_yarn_position_encoding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            # device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"模型加载完成，设备: {device}")

    def get_prefill_kv_cache(self, 
        texts: List[str],
        keep_begin_of_text: bool=True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取一批文本prefill后的键值缓存
        
        Args:
            texts: 文本列表
            
        Returns:
            past_key_values: 包含所有层键值缓存的元组
        """
        # 编码文本
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to(self.device)
        # print("get_prefill_kv_cache input_ids")
        # print(inputs.input_ids[:, :10])
        if not keep_begin_of_text and inputs.input_ids[0, 0] == self.tokenizer.bos_token_id:
            inputs["input_ids"] = inputs.input_ids[:, 1:]
            inputs["attention_mask"] = inputs.attention_mask[:, 1:]
        print("get_prefill input_ids")
        print(inputs.input_ids[:, :10])
        
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
                layer_kv.append(tuple(korv))
            full_kv.append(tuple(layer_kv))
        return tuple(full_kv), input_ids

    def decode_with_past_kv(
        self,
        query_text: str,
        system_prompt_ids: Optional[List[torch.Tensor]] = None,
        system_prompt_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        precomputed_ids: Optional[List[torch.Tensor]] = None,
        precomputed_kv: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        max_new_tokens: int = 512,
        return_kv: bool = False,
    ) -> str:
        """
        使用已有的键值缓存进行解码
        
        Args:
            system_prompt_kv: 系统提示词的键值缓存
            precomputed_kv: 预先计算好的键值缓存
            query_text: 查询文本
            max_new_tokens: 最大生成token数
            
        Returns:
            generated_text: 生成的文本
        """
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
        # print(query_inputs)
        print("input len", query_inputs.input_ids.shape)
        print("input ids")
        print(query_inputs.input_ids[0, 30:40])
        # breakpoint()
        
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
        
        if precomputed_kv is None:
            return system_kv
        
        # 合并两个KV缓存（在序列长度维度上拼接）
        merged_kv = []
        print("model rope theta", self.model.config.rope_theta)
        for system_layer, recomp_layer in zip(system_kv, precomputed_kv):
            layer_full_kv = []
            for kv_i in range(2):
                layer_kv = [system_layer[kv_i][0]]
                cumulative_kv_len = int(system_layer[kv_i][0].shape[1])
                for recomp_layer_kv in recomp_layer[kv_i]:
                    # if kv_i == 0: # Rotate key cache to have correct RoPE
                    #     # print(f"cumulative_kv_len: {cumulative_kv_len}")
                    #     recomp_layer_kv = rotate_k_cache_rope(
                    #         recomp_layer_kv, 
                    #         cumulative_kv_len,
                    #         self.model.config.rope_theta,
                    #     )
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

    def save_kv_cache(self, kv_cache: Tuple[Tuple[torch.Tensor, ...], ...], filepath: str):
        """
        保存键值缓存到文件
        
        Args:
            kv_cache: 要保存的键值缓存
            filepath: 文件路径
        """
        torch.save(kv_cache, filepath)
        logger.info(f"KV缓存已保存到: {filepath}")

    def load_kv_cache(self, filepath: str) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        从文件加载键值缓存
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的键值缓存
        """
        kv_cache = torch.load(filepath, map_location=self.device)
        logger.info(f"KV缓存已从 {filepath} 加载")
        return kv_cache

# 使用示例
if __name__ == "__main__":
    # 初始化推理器
    inference = LLMInference("Qwen/Qwen2.5-7B-Instruct")
    
    # 示例1: 获取prefill KV缓存
    sys_prompt = ['你是一个人工智能，请按照用户提出的问题准确无误且专业地回答问题：']
    sys_cache = inference.get_prefill_kv_cache(sys_prompt)
    print(len(sys_cache), len(sys_cache[0]), len(sys_cache[0][0]), sys_cache[0][0][0].shape)

    texts = ["Hello, What about your last job. You are not looking well. ", "It was frustrating."]
    kv_cache = inference.get_prefill_kv_cache(texts)
    print(f"获取到 {len(texts)} 个文本的KV缓存")
    print(len(kv_cache), len(kv_cache[0]), len(kv_cache[0][0]), kv_cache[0][0][0].shape)

    # 保存KV缓存（可选）
    inference.save_kv_cache(kv_cache, "precomputed_kv.pth")
    
    # 示例2: 使用KV缓存进行解码
    query = "请继续上面的对话："
    query_all = "".join(sys_prompt + texts) + query
    generated_text = inference.decode_with_past_kv(
        system_prompt_kv=sys_cache,
        precomputed_kv=kv_cache,
        query_text=query_all,
        max_new_tokens=100
    )
    
    print(f"生成的文本: {generated_text}")