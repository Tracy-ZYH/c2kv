import os
import subprocess
import json
import argparse
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# 全局锁用于线程安全的日志记录
print_lock = threading.Lock()

def log_message(message):
    """线程安全的日志输出到stdout"""
    with print_lock:
        print(message, flush=True)

def log_error(message):
    """线程安全的错误输出到stderr"""
    with print_lock:
        print(message, file=sys.stderr, flush=True)

def run_evaluation_command(gpu_id, cmd_line, results_list, lock, progress_counter, total_commands, output_dir):
    """在指定GPU上运行单个命令"""
    
    cmd = cmd_line.split()
    model_path = None
    
    # 从命令行中提取模型路径
    for i, arg in enumerate(cmd):
        if arg == '--model' and i + 1 < len(cmd):
            model_path = cmd[i + 1]
            break
    
    if not model_path:
        log_error(f"[ERROR][GPU {gpu_id}] Could not extract model path from command: {cmd_line}")
        return
    
    # 设置GPU环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_message(f"[GPU {gpu_id}] Running evaluation: {cmd_line}")

    try:
        # 执行命令，捕获输出
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode != 0:
            log_error(f"[ERROR][GPU {gpu_id}] Evaluation failed for {model_path}")
            log_error(f"[STDERR][GPU {gpu_id}] {result.stderr.decode('utf-8', errors='replace')}")
            return
        
        # 尝试解析输出文件获取结果
        output_file = None
        for i, arg in enumerate(cmd):
            if arg == '--output_file' and i + 1 < len(cmd):
                output_file = cmd[i + 1]
                break
        
        if not output_file:
            log_error(f"[ERROR][GPU {gpu_id}] Could not find output file in command: {cmd_line}")
            # 如果没有明确的输出文件，创建一个默认的
            output_file = os.path.join(output_dir, f"output_{gpu_id}_{int(time.time())}.json")
        
        # 构造 summary 文件路径
        base_name = os.path.splitext(output_file)[0]
        summary_file = f"{base_name}_summary.json"
        
        if not os.path.exists(summary_file):
            log_error(f"[WARNING][GPU {gpu_id}] Summary file not found: {summary_file}")
            # 检查是否有其他可能的summary文件
            output_dir_check = os.path.dirname(output_file)
            for f in os.listdir(output_dir_check) if os.path.exists(output_dir_check) else []:
                if f.startswith(os.path.basename(base_name)) and f.endswith('_summary.json'):
                    summary_file = os.path.join(output_dir_check, f)
                    break
        
        exact_match = None
        num_examples = None
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                exact_match = summary.get('exact_match')
                num_examples = summary.get('num_examples')
            except Exception as e:
                log_error(f"[ERROR][GPU {gpu_id}] Failed to parse summary file {summary_file}: {e}")
        else:
            log_message(f"[INFO][GPU {gpu_id}] No summary file found for {model_path}, assuming success")
        
        result_entry = {
            "model": model_path,
            "dataset": [cmd[i + 1] for i, arg in enumerate(cmd) if arg == '--dataset' and i + 1 < len(cmd)][0] if '--dataset' in cmd else 'unknown',
            "num_examples": num_examples,
            "score": exact_match,
        }
        
        with lock:
            results_list.append(result_entry)
            progress_counter[0] += 1
            completed = progress_counter[0]
            
        log_message(f"[GPU {gpu_id}] Completed {model_path} | EM: {exact_match} | Progress: {completed}/{total_commands} ({completed/total_commands*100:.1f}%)")
        
    except subprocess.TimeoutExpired:
        log_error(f"[ERROR][GPU {gpu_id}] Evaluation timeout for {model_path} in command: {cmd_line}")
    except Exception as e:
        log_error(f"[ERROR][GPU {gpu_id}] Unexpected error for {model_path}: {e}")
        log_error(f"[ERROR][GPU {gpu_id}] Command: {cmd_line}")

class GPUResourcePool:
    """GPU资源池，动态分配GPU资源"""
    
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids
        self.available_gpus = set(gpu_ids)  # 可用GPU集合
        self.gpu_lock = threading.Lock()  # 保护GPU资源的锁
        self.gpu_condition = threading.Condition(self.gpu_lock)  # 条件变量用于等待GPU
    
    def acquire_gpu(self):
        """获取一个可用的GPU ID，如果没有可用GPU则等待"""
        with self.gpu_condition:
            while not self.available_gpus:
                # 没有可用GPU，等待
                self.gpu_condition.wait()
            # 获取一个GPU
            gpu_id = self.available_gpus.pop()
            return gpu_id
    
    def release_gpu(self, gpu_id):
        """释放GPU资源"""
        with self.gpu_condition:
            self.available_gpus.add(gpu_id)
            # 通知等待的线程
            self.gpu_condition.notify_all()

def main():
    parser = argparse.ArgumentParser(description="Execute evaluation commands in parallel on multiple GPUs.")
    parser.add_argument('--commands_file', required=True, help='TXT file containing commands to execute')
    parser.add_argument('--gpus', default="0,1,2,3,4,5,6,7", help='GPU IDs to use, separated by commas (e.g., 0,1,2,3)')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save results')
    args = parser.parse_args()

    # 解析GPU列表
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    if not gpu_ids:
        raise ValueError("No valid GPU IDs provided")
    
    log_message(f"Using GPUs: {gpu_ids}")
    
    # 创建GPU资源池
    gpu_pool = GPUResourcePool(gpu_ids)
    
    # 创建临时目录用于存储中间输出
    temp_output_dir = tempfile.mkdtemp(prefix="eval_output_")
    log_message(f"Using temporary output directory: {temp_output_dir}")
    
    # 读取命令文件
    if not os.path.exists(args.commands_file):
        log_error(f"[ERROR] Commands file does not exist: {args.commands_file}")
        return
    
    with open(args.commands_file, 'r') as f:
        commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not commands:
        log_message("[WARNING] No commands found in the file.")
        return
    
    log_message(f"Found {len(commands)} commands to execute")
    
    # 结果收集
    results = []
    results_lock = threading.Lock()
    
    # 进度计数器（使用列表以便在闭包中修改）
    progress_counter = [0]
    total_commands = len(commands)
    
    def execute_task(cmd):
        """执行单个任务，动态获取GPU资源"""
        gpu_id = gpu_pool.acquire_gpu()
        try:
            run_evaluation_command(
                gpu_id,
                cmd,
                results,
                results_lock,
                progress_counter,
                total_commands,
                temp_output_dir
            )
        finally:
            gpu_pool.release_gpu(gpu_id)
    
    # 使用线程池执行任务，线程数可以远大于GPU数，因为GPU分配是动态的
    max_workers = min(len(commands), len(gpu_ids) * 2)  # 不超过命令数，但可以是GPU数的倍数
    log_message(f"Starting execution with {max_workers} concurrent workers and {len(gpu_ids)} GPUs")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_cmd = {executor.submit(execute_task, cmd): cmd for cmd in commands}
        
        # 等待所有任务完成
        for future in as_completed(future_to_cmd):
            cmd = future_to_cmd[future]
            try:
                future.result()
            except Exception as exc:
                log_error(f'[ERROR] Command generated an exception: {exc}')
                log_error(f'[ERROR] Command: {cmd}')
    
    # 保存结果到文件
    try:
        with open(args.output_file, 'a') as f:
            for result in results:
                if result:  # 只写入非空结果
                    f.write(json.dumps(result) + '\n')
        log_message(f"\nAll evaluations completed. Results saved to {args.output_file}")
    except Exception as e:
        log_error(f"[ERROR] Failed to save results to {args.output_file}: {e}")
        # 作为备选，保存到当前目录
        backup_file = "evaluation_results_backup.json"
        try:
            with open(backup_file, 'w') as f:
                for result in results:
                    if result:
                        f.write(json.dumps(result) + '\n')
            log_message(f"Results saved to backup file: {backup_file}")
        except Exception as backup_e:
            log_error(f"[ERROR] Failed to save backup results: {backup_e}")
    
    # 打印汇总结果
    log_message("\n=== EVALUATION RESULTS ===")
    for r in results:
        if r:
            score_str = f"{r['score']:.4f}" if r['score'] is not None else "N/A"
            log_message(f"{r['model']} | EM: {score_str}")

if __name__ == '__main__':
    main()
