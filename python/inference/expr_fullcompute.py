import os
import argparse
import json
import math
import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from compress_kv import compress_kv, QueryStorage
from mdocdataset import load_mdoc_dataset, AbstractMDQADataset
from expr_timer import DataRecorder, ExprTimer


def prepare_example_with_template(example, system_prompt, tokenizer):
    """处理单个示例，使用chat template"""
    if 'system_prompt' in example:
        system_prompt = example['system_prompt']
    
    messages = [{"role": "system", "content": system_prompt},]
    for document in example['documents']:
        messages.append({"role": "user", "content": document})
    
    context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example['question']},],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    
    return context, prompt, example['answer'], example['qid']


class MDQAEvaluator:
    def __init__(
        self, 
        model_name: str, 
        compression_method: str | None = None,
        device: str = "cuda", 
        torch_dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True, padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.compression_method = compression_method
    
    @torch.inference_mode()
    def generate_answer(self, context: str, prompt: str, max_new_tokens: int, record: DataRecorder) -> str:
        """生成答案，prefill prompt后再generation"""
        # Prefill: tokenize context
        context_inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        context_length = context_inputs.input_ids.shape[1]
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        # 计算prompt的position ids
        position_ids = torch.arange(context_length, context_length + prompt_length, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Prefill阶段：获取最后一个token的hidden state
        enable_compress = self.compression_method is not None
        with QueryStorage(self.model, enable_compress) as query_storage:
            with record.record('extract'):
                past_key_values = self.model(**context_inputs, return_dict=True, use_cache=True).past_key_values
                queries = query_storage.get_all_queries()
        
        if enable_compress:
            capacity = math.ceil(context_length / 4.0)
            compressed_key_values = []
            for query, layer in zip(queries, past_key_values.layers):
                keys, values, _ = compress_kv(
                    self.compression_method, capacity, query[0], layer.keys[0], layer.values[0]
                )
                compressed_key_values.append((keys.unsqueeze(0), values.unsqueeze(0)))
            past_key_values = DynamicCache(compressed_key_values)

        if record.enable:
            cache_cpu = [(key.to(device="cpu"), value.to(device="cpu")) for key, value in past_key_values]
            with record.record('offload'):
                for layer_i in range(len(cache_cpu)):
                    k, v = cache_cpu[layer_i]
                    cache_cpu[layer_i] = (k.to(self.device), v.to(self.device))
            del cache_cpu
        
        mock_input_ids = prompt_inputs.input_ids.new_zeros((1, 1)).expand((-1, past_key_values.get_seq_length()))
        input_ids = torch.cat([mock_input_ids, prompt_inputs.input_ids], dim=1)
        attention_mask = context_inputs.attention_mask.new_ones(input_ids.shape)
        with record.record('generate'):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 解码并移除prompt部分
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = full_text[len(prompt_text):].strip()

        return answer


def evaluate_model_on_dataset(
    dataset: AbstractMDQADataset,
    args: argparse.Namespace,
):
    max_examples = args.max_examples
    output_file = args.output_file

    """评估模型"""
    evaluator = MDQAEvaluator(args.model, args.compress, device=args.device)
    system_prompt = dataset.system_prompt
    timer = ExprTimer("fr", args.profile)
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    
    results = []
    em_scores = []
    
    for i in tqdm(range(num_examples), desc="Evaluating"):
        example = dataset[i]
        record = timer.record(example['qid'])
        context, prompt, gt, qid = prepare_example_with_template(example, system_prompt, evaluator.tokenizer)
        
        max_new_tokens = example['max_new_tokens'] if timer.enable and 'max_new_tokens' in example else dataset.max_new_tokens
        pred = evaluator.generate_answer(context, prompt, max_new_tokens, record)
        em_score = dataset.metric(pred, gt)
        em_scores.append(em_score)
        
        results.append({
            'qid': qid,
            'prediction': pred,
            'ground_truth': gt,
            'em_score': em_score
        })
        if timer.enable:
            record.phases['generate'] /= len(evaluator.tokenizer.encode(pred))
            results[-1]['timer'] = record.summary()
            del results[-1]['prediction']
    
    # 计算指标
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0.0
    
    # 保存结果
    if output_file:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        summary = {
            'model': args.model,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
            **timer.statistics(),
        }
        
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return {'exact_match': exact_match, 'num_examples': len(results)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on MDQA datasets")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--max_examples", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--output_file", type=str, default=None, help="Output file")
    parser.add_argument("--only_supporting", action="store_true", help="Use only supporting paragraphs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--cot", action="store_true", default=False, help="Use cot prompt")
    parser.add_argument("--compress", type=str, default=None, help="KV Cache compression type")
    parser.add_argument("--profile", action="store_true", default=False, help="Profile model")
    
    args = parser.parse_args()
    
    dataset = load_mdoc_dataset(
        args.dataset, args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    results = evaluate_model_on_dataset(dataset=dataset, args=args)
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Examples evaluated: {results['num_examples']}")
    print(f"Exact Match: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
