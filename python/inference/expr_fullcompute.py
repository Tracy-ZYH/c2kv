import argparse
import json
import string
import regex
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import datasets
import concurrent.futures

from mdocdataset import load_mdoc_dataset, AbstractMDQADataset


def prepare_example_with_template(example, system_prompt, tokenizer):
    """处理单个示例的辅助函数，使用chat template"""
    if 'system_prompt' in example:
        system_prompt = example['system_prompt']
    
    # 构建对话格式的消息列表
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for document in example['documents']:
        messages.append({"role": "user", "content": document})
    messages.append({"role": "user", "content": example['question']})
    
    # 使用chat template处理消息
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt, example['answer'], example['qid']

def prepare_prompts_parallel(dataset, max_examples=None, max_workers=None, tokenizer=None):
    """使用多线程准备prompts，使用chat template"""
    prompts = []
    ground_truths = []
    qids = []
    
    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    
    # 获取系统提示（假设对所有示例都相同）
    system_prompt = dataset.system_prompt
    
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(prepare_example_with_template, dataset[i], system_prompt, tokenizer): i 
            for i in range(num_examples)
        }
        
        # 使用tqdm显示进度
        with tqdm(total=num_examples, desc="Preparing prompts") as pbar:
            # 按完成顺序处理结果
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    prompt, answer, qid = future.result()
                    prompts.append(prompt)
                    ground_truths.append(answer)
                    qids.append(qid)
                    pbar.update(1)
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"Error processing example {idx}: {e}")
                    pbar.update(1)
    
    return prompts, ground_truths, qids


class MDQAEvaluator:
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True, padding_side='left',
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def generate_answer(self, prompt: str, generation_config: Optional[GenerationConfig]=None) -> str:
        """Generate answer for a single prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                # generation_config=generation_config,
                max_new_tokens=generation_config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated part (after the prompt)
        prompt_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        answer = generated_text[prompt_length:].strip()
        
        return answer
    
    def batch_generate_answers(self, prompts: List[str], batch_size: int = 1, 
                             generation_config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate answers for a batch of prompts"""
        answers = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize batch - now using the properly formatted prompts
            try:
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            except Exception as e:
                batch_prompts = [prompt.encode("utf-8", errors="ignore").decode("utf-8") for prompt in batch_prompts]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device),
                    # generation_config=generation_config,
                    max_new_tokens=generation_config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # Decode each output in the batch
            for j in range(len(batch_prompts)):
                generated_text = self.tokenizer.decode(outputs[j], skip_special_tokens=True)
                prompt_length = len(self.tokenizer.decode(inputs.input_ids[j], skip_special_tokens=True))
                answer = generated_text[prompt_length:].strip()
                answers.append(answer)
        
        return answers


def evaluate_model_on_dataset(
    model_name: str,
    dataset: AbstractMDQADataset,
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate a model on a MDQA dataset"""
    
    # Initialize evaluator to get tokenizer
    evaluator = MDQAEvaluator(model_name)
    
    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=dataset.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=evaluator.tokenizer.pad_token_id
    )
    
    # Prepare prompts using chat template
    prompts, ground_truths, qids = prepare_prompts_parallel(
        dataset, max_examples, max_workers=8, tokenizer=evaluator.tokenizer
    )
    
    # Generate answers
    predictions = evaluator.batch_generate_answers(prompts, batch_size, generation_config)
    
    # Calculate metrics
    em_scores = []
    results = []
    
    for pred, gt, qid, prompt in zip(predictions, ground_truths, qids, prompts):
        em_score = dataset.metric(pred, gt)
        em_scores.append(em_score)
        
        results.append({
            'qid': qid,
            'prediction': pred,
            'ground_truth': gt,
            'em_score': em_score
        })
    
    # Calculate overall metrics
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            for result in results:
                if result:  # 只写入非空结果
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Also save a summary
        summary = {
            'model': model_name,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(results),
            'exact_match': exact_match,
            'batch_size': batch_size
        }
        
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return {
        'exact_match': exact_match,
        'num_examples': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoregressive models on MDQA datasets")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name or path (e.g., 'Qwen/Qwen2-7B-Instruct', 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Type of dataset")
    
    # Optional arguments
    parser.add_argument("--dataset_path", type=str, default=None, 
                       help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for inference")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save results")
    parser.add_argument("--only_supporting", action="store_true",
                       help="For Musique dataset, use only supporting paragraphs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--cot", action="store_true", default=False,
                       help="Use cot prompt")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_mdoc_dataset(
        args.dataset, 
        args.dataset_path, 
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")
    
    # Evaluate model
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset=dataset,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        output_file=args.output_file
    )
    
    print(f"\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
