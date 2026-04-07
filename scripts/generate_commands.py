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
    parser.add_argument('--datasets', default='wikimqa,musique,hotpotqa,multinews,samsum', help='Dataset names separated by comma (e.g., A,B,C,D)')
    parser.add_argument('--max_examples', type=int, default=None, help='Max number of examples to test.')
    parser.add_argument('--output_file', required=True, help='Output file to save commands')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite the output file instead of appending')
    args, extra_args = parser.parse_known_args()

    # 解析多个dataset
    datasets = [ds.strip() for ds in args.datasets.split(',')]
    
    # 查找所有checkpoints
    checkpoints = find_checkpoints(args.checkpoints_roots)
    if not checkpoints:
        print("[WARNING] No checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    print(f"Will evaluate on {len(datasets)} datasets: {datasets}")
    if extra_args:
        print(f"Extra arguments to append: {' '.join(extra_args)}")
    
    # 生成命令并写入文件
    file_mode = 'w' if args.overwrite else 'a'
    with open(args.output_file, file_mode) as f:
        for checkpoint in checkpoints:
            method = checkpoint.split('/')[-2]
            model = checkpoint.split('/')[-3]
            for dataset in datasets:
                cmd = [
                    'python', 'python/inference/expr_c2kv.py',
                    '--model', checkpoint,
                    '--output_file', f'results/gist/{model}/{method}/{dataset}/{os.path.basename(checkpoint)}.jsonl',
                    '--dataset', dataset,
                ]
                if args.max_examples:
                    cmd.extend(['--max_examples', str(args.max_examples)])
                if extra_args:
                    cmd.extend(extra_args)
                f.write(' '.join(cmd) + '\n')
    
    print(f"Commands {'written' if args.overwrite else 'appended'} to {args.output_file}")
    print(f"Total commands generated: {len(checkpoints) * len(datasets)}")

if __name__ == '__main__':
    main()
