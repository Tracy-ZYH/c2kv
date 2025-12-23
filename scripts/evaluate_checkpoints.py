import os
import subprocess
import json
import argparse
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 全局锁用于线程安全的日志记录
print_lock = threading.Lock()

def log_message(message):
    """线程安全的日志输出"""
    with print_lock:
        print(message)

def find_checkpoints(roots):
    """从多个根路径查找所有以 checkpoint- 开头的子目录"""
    ckpt_dirs = []
    
    for root in roots:
        root = root.strip()
        if not os.path.exists(root):
            log_message(f"[WARNING] Checkpoints root path does not exist: {root}")
            continue
            
        for sub_root, dirs, _ in os.walk(root):
            for d in dirs:
                if d.startswith("checkpoint-"):
                    full_path = os.path.join(sub_root, d)
                    ckpt_dirs.append(full_path)
    
    return sorted(ckpt_dirs)

def run_evaluation(gpu_id, model_path, dataset, max_examples, output_dir, results_list, lock, progress_counter, total_checkpoints):
    """在指定GPU上运行单个评估任务"""
    
    # 创建唯一临时文件名
    timestamp = int(time.time() * 1000000) % 1000000
    output_file = os.path.join(output_dir, f"output_{gpu_id}_{timestamp}.json")
    
    cmd = [
        'python', 'python/inference/expr_gistmodel.py',
        '--model', model_path,
        '--dataset', dataset,
        '--max_examples', str(max_examples),
        '--output_file', output_file
    ]
    
    # 设置GPU环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_message(f"[GPU {gpu_id}] Running evaluation: {' '.join(cmd)}")
    
    try:
        # 直接执行，不重定向stdout和stderr
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(
                cmd,
                stdout=devnull,
                stderr=devnull,
                env=env,
                timeout=3600  # 1小时超时
            )
        
        if result.returncode != 0:
            log_message(f"[ERROR][GPU {gpu_id}] Evaluation failed for {model_path}")
            return
        
        # 构造 summary 文件路径
        base_name = os.path.splitext(output_file)[0]
        summary_file = f"{base_name}_summary.json"
        
        if not os.path.exists(summary_file):
            log_message(f"[WARNING][GPU {gpu_id}] Summary file not found: {summary_file}")
            return
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            exact_match = summary.get('exact_match')
            num_examples = summary.get('num_examples')
            
            result_entry = {
                "model": model_path,
                "dataset": dataset,
                "num_examples": num_examples,
                "score": exact_match,
            }
            
            with lock:
                results_list.append(result_entry)
                progress_counter[0] += 1
                completed = progress_counter[0]
                
            log_message(f"[GPU {gpu_id}] Completed {model_path} | EM: {exact_match:.4f} | Progress: {completed}/{total_checkpoints} ({completed/total_checkpoints*100:.1f}%)")
            
        except Exception as e:
            log_message(f"[ERROR][GPU {gpu_id}] Failed to parse summary file {summary_file}: {e}")
            
    except subprocess.TimeoutExpired:
        log_message(f"[ERROR][GPU {gpu_id}] Evaluation timeout for {model_path}")
    except Exception as e:
        log_message(f"[ERROR][GPU {gpu_id}] Unexpected error for {model_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints on multiple GPUs.")
    parser.add_argument('--add_ckpt', action='append', required=True, dest='checkpoints_roots', help='Additional checkpoint directory to evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., musique)')
    parser.add_argument('--max_examples', type=int, default=100, help='Max number of examples to test.')
    parser.add_argument('--gpus', required=True, help='GPU IDs to use, separated by commas (e.g., 0,1,2,3)')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save results')
    args = parser.parse_args()

    # 解析GPU列表
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    if not gpu_ids:
        raise ValueError("No valid GPU IDs provided")
    
    log_message(f"Using GPUs: {gpu_ids}")
    
    # 创建临时目录用于存储中间输出
    temp_output_dir = tempfile.mkdtemp(prefix="eval_output_")
    log_message(f"Using temporary output directory: {temp_output_dir}")
    
    # 查找所有checkpoints
    checkpoints = find_checkpoints(args.checkpoints_roots)
    if not checkpoints:
        log_message("[WARNING] No checkpoints found.")
        return
    
    log_message(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # 结果收集
    results = []
    results_lock = threading.Lock()
    
    # 进度计数器（使用列表以便在闭包中修改）
    progress_counter = [0]
    total_checkpoints = len(checkpoints)
    
    # 创建GPU任务队列
    gpu_queue = gpu_ids.copy()
    gpu_queue_lock = threading.Lock()
    
    def get_available_gpu():
        with gpu_queue_lock:
            if gpu_queue:
                return gpu_queue.pop(0)
            return None
    
    def release_gpu(gpu_id):
        with gpu_queue_lock:
            gpu_queue.append(gpu_id)
    
    # GPU任务包装器
    def evaluate_checkpoint(model_path):
        gpu_id = None
        while gpu_id is None:
            gpu_id = get_available_gpu()
            if gpu_id is None:
                time.sleep(1)  # 等待GPU释放
        
        try:
            run_evaluation(
                gpu_id=gpu_id,
                model_path=model_path,
                dataset=args.dataset,
                max_examples=args.max_examples,
                output_dir=temp_output_dir,
                results_list=results,
                lock=results_lock,
                progress_counter=progress_counter,
                total_checkpoints=total_checkpoints
            )
        finally:
            release_gpu(gpu_id)
    
    # 使用线程池并发执行
    max_workers = len(gpu_ids)
    log_message(f"Starting evaluation with {max_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_model = {executor.submit(evaluate_checkpoint, ckpt): ckpt for ckpt in checkpoints}
        
        # 等待所有任务完成
        for future in as_completed(future_to_model):
            model_path = future_to_model[future]
            try:
                future.result()
            except Exception as exc:
                log_message(f'[ERROR] Model {model_path} generated an exception: {exc}')
    
    # 保存结果到文件
    try:
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        log_message(f"\nAll evaluations completed. Results saved to {args.output_file}")
    except Exception as e:
        log_message(f"[ERROR] Failed to save results to {args.output_file}: {e}")
        # 作为备选，保存到当前目录
        backup_file = "evaluation_results_backup.json"
        with open(backup_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        log_message(f"Results saved to backup file: {backup_file}")
    
    # 打印汇总结果
    log_message("\n=== EVALUATION RESULTS ===")
    for r in results:
        log_message(f"{r['model']} | EM: {r['score']:.4f}")

if __name__ == '__main__':
    main()
