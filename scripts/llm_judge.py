# PYTHONPATH=`pwd`/python/inference:$PYTHONPATH
import argparse
import json
import os
from typing import Optional, Dict
import asyncio
import aiohttp
from mdocdataset import load_mdoc_dataset, AbstractMDQADataset
from tqdm import tqdm


def extract_question_parts(dataset_name: str, item: Dict[str, str]) -> str:
    """
    Extract useful parts from question for different datasets
    """
    # Add more dataset-specific extractors as needed
    if dataset_name == "amap":
        return item['system_prompt'].split("\n<用户请求>\n")[1].split("</用户请求>")[0].strip()
    return item['question'].strip()


def load_prediction_json(file_path: str):
    """Load prediction JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_prompt_template(file_path: str):
    """Load prompt template from JSON file with system and user parts"""
    with open(file_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    if 'system' not in template or 'user' not in template:
        raise ValueError("Prompt template JSON must contain 'system' and 'user' keys")
    
    return template['system'], template['user']


def create_qid_to_question_map(dataset, dataset_name: str):
    """Create a map from qid to processed question"""
    qid_to_question = {}
    for item in dataset:
        qid = item.get('qid', item.get('id', ''))
        if qid:
            qid_to_question[qid] = extract_question_parts(dataset_name, item)
    return qid_to_question


def format_chat_template(system_msg: str, user_msg: str):
    """Format messages into chat template"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


async def call_llm_judge_async(session, api_url: str, headers: dict, messages: list, model: str = "gpt-4"):
    """Asynchronously call the LLM judge with the prepared messages"""
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1000
    }
    
    async with session.post(f"{api_url}/chat/completions", headers=headers, json=data) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API call failed with status {response.status}: {error_text}")
        
        response_json = await response.json()
        return response_json["choices"][0]["message"]["content"]


async def process_single_judgment(session, semaphore, api_url, headers, system_prompt, user_prompt_template,
                                 qid, prediction, ground_truth, question, original_data, model):
    """Process a single judgment request with semaphore"""
    async with semaphore:  # Limit concurrent requests
        # Format the user prompt with question, prediction, and ground truth
        formatted_user_prompt = user_prompt_template.format(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth
        )
        
        # Prepare messages for chat completion
        messages = format_chat_template(system_prompt, formatted_user_prompt)
        
        # Call LLM judge
        try:
            llm_judge_result = await call_llm_judge_async(
                session=session,
                api_url=api_url,
                headers=headers,
                messages=messages,
                model=model,
            )
        except Exception as e:
            print(f"Error calling LLM judge for qid {qid}: {str(e)}")
            llm_judge_result = "Error occurred during evaluation"
        
        # Prepare output data
        output_data = {
            "qid": qid,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "llm_judge_response": llm_judge_result,
            # "original_prediction_data": original_data
        }
        
        return output_data


async def main_async(dataset, dataset_name, predictions, api_url, headers, system_prompt, user_prompt_template, 
                     output_file_path, max_concurrent, model="gpt-4"):
    """Main async function to process all judgments"""
    # Create qid to question map
    print("Creating question mapping...")
    qid_to_question = create_qid_to_question_map(dataset, dataset_name)
    print(f"Created mapping for {len(qid_to_question)} questions")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Open output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Create async session
        async with aiohttp.ClientSession() as session:
            # Prepare tasks for all predictions
            tasks = []
            for pred_item in predictions:
                qid = pred_item['qid']
                
                # Get question from mapping
                if qid not in qid_to_question:
                    print(f"Warning: Question with qid {qid} not found in dataset")
                    continue
                
                question = qid_to_question[qid]
                prediction = pred_item['prediction']
                ground_truth = pred_item['ground_truth'][0] if pred_item['ground_truth'] else ""
                
                task = process_single_judgment(
                    session=session,
                    semaphore=semaphore,
                    api_url=api_url,
                    headers=headers,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    qid=qid,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    question=question,
                    original_data=pred_item,
                    model=model
                )
                tasks.append(task)
            
            # Create tqdm progress bar
            pbar = tqdm(total=len(tasks), desc="Processing LLM Judgments", unit="item")
            
            # Process all tasks with progress bar
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_file.flush()
                    pbar.update(1)
            
            pbar.close()
    
    print(f"LLM Judge evaluation completed. Results saved to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description='LLM Judge script for evaluating predictions')
    parser.add_argument('--model', type=str, default="gpt-4", help='LLM model to use (default: gpt-4)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--dataset_cot', action='store_true', help='Use cot prompt')
    parser.add_argument('--prediction', type=str, required=True, help='Prediction JSON file path')
    parser.add_argument('--api_url', type=str, required=True, help='OpenAI API URL')
    parser.add_argument('--prompt_file', type=str, required=True, help='LLM Judge Prompt JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--max_concurrent', type=int, default=5, help='Maximum number of concurrent API calls (default: 5)')
    
    args = parser.parse_args()
    
    # Prepare headers with API key if provided
    headers = {
        "Content-Type": "application/json",
    }
    
    # Load prompt template
    system_prompt, user_prompt_template = load_prompt_template(args.prompt_file)
    
    # Load prediction data
    predictions = load_prediction_json(args.prediction)
    
    # Load dataset
    dataset = load_mdoc_dataset(args.dataset, enable_cot=args.dataset_cot)
    
    # Run async main function
    asyncio.run(main_async(
        dataset=dataset,
        dataset_name=args.dataset,
        predictions=predictions,
        api_url=args.api_url,
        headers=headers,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        output_file_path=args.output,
        max_concurrent=args.max_concurrent,
        model=args.model
    ))


if __name__ == "__main__":
    main()
