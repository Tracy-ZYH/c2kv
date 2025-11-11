import json
import os
import re
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def extract_rating_from_response(response):
    if not isinstance(response, str):
        return None
    pattern = r'评分[：:]\s*(\d)/5'
    matches = re.findall(pattern, response)
    if matches:
        return int(matches[-1])
    return None

def process_jsonl_file(file_path):
    ratings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 检查是否存在"llm_judge_response"键
                if "llm_judge_response" in item:
                    response = item["llm_judge_response"]
                    rating = extract_rating_from_response(response)
                    if rating is not None:
                        ratings.append(rating) 
            except json.JSONDecodeError:
                # 跳过无法解析的行
                continue
    return np.array(ratings, dtype=int)

def process_directory(directory_path):
    directory = Path(directory_path)
    results = {}
    jsonl_files = list(directory.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name}...")
        ratings_array = process_jsonl_file(jsonl_file)
        results[jsonl_file.name] = ratings_array
    return results

def save_to_npz(results, output_path):
    # 准备保存的数据
    save_dict = {}
    for filename, ratings_array in results.items():
        # 使用文件名作为key，但替换不合法字符
        key = '_'.join(filename.split('.')[:-1]).replace('-', '_')
        save_dict[key] = ratings_array
        # 同时保存文件名映射
        save_dict[f"{key}_filename"] = filename
    np.savez(output_path, **save_dict)

def main(directory_path, output_path="results.npz"):
    print(f"Processing directory: {directory_path}")
    results = process_directory(directory_path)
    for filename, ratings in results.items():
        print(f"{filename}: {len(ratings)} ratings found")
        if len(ratings) > 0:
            print(f"  Ratings range: {ratings.min()} - {ratings.max()}")
    save_to_npz(results, output_path)
    print(f"Results saved to {output_path}")

    return results

# 使用示例
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--score_dir", type=str, default="./results/amap/llm_judge")
    parser.add_argument("--output_path", type=str, default="ratings_results.npz")
    args = parser.parse_args()

    results = main(args.score_dir, args.output_path)
