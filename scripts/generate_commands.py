import os
import argparse
from pathlib import Path
from tarfile import data_filter

def find_checkpoints(roots):
    """从多个根路径查找所有以 checkpoint- 开头的子目录"""
    ckpt_dirs = []
    
    for root in roots:
        root = root.strip()
        if not os.path.exists(root):
            print(f"[WARNING] Checkpoints root path does not exist: {root}")
            continue
            
        for sub_root, dirs, _ in os.walk(root):
            for d in dirs:
                if d.startswith("checkpoint-"):
                    full_path = os.path.join(sub_root, d)
                    ckpt_dirs.append(full_path)
    
    return sorted(ckpt_dirs)

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation commands for multiple checkpoints.")
    parser.add_argument('--add_ckpt', action='append', required=True, dest='checkpoints_roots', help='Additional checkpoint directory to evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., musique)')
    parser.add_argument('--max_examples', type=int, default=100, help='Max number of examples to test.')
    parser.add_argument('--output_file', required=True, help='Output TXT file to save commands')
    args = parser.parse_args()

    # 查找所有checkpoints
    checkpoints = find_checkpoints(args.checkpoints_roots)
    if not checkpoints:
        print("[WARNING] No checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # 生成命令并写入文件
    with open(args.output_file, 'a') as f:
        for checkpoint in checkpoints:
            cmd = [
                'python', 'python/inference/expr_gistmodel.py',
                '--model', checkpoint,
                '--dataset', args.dataset,
                # '--max_examples', str(args.max_examples),
                '--output_file', f'saved_kv/tmp/{args.dataset}/{os.path.basename(checkpoint)}.jsonl'
            ]
            f.write(' '.join(cmd) + '\n')
    
    print(f"Commands saved to {args.output_file}")

if __name__ == '__main__':
    main()
