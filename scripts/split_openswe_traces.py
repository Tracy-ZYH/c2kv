#!/usr/bin/env python3
"""
Scientific train-test split for Open-SWE-Traces dataset.
Split by instance_id (task/session) to avoid data leakage.
Optimized version with parallel processing.
"""

import os
import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict
import random
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration
SOURCE_DIR = "/mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces"
TRAIN_DIR = "/mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces--train"
TEST_DIR = "/mnt/nas1/duchuheng/datasets/nvidia--Open-SWE-Traces--test"
TEST_RATIO = 0.1  # 10% for test set
RANDOM_SEED = 42  # For reproducibility

SUBDIRS = [
    "minimax_m25_openhands_trajectories",
    "minimax_m25_sweagent_trajectories",
    "qwen35_openhands_trajectories",
    "qwen35_sweagent_trajectories"
]

def collect_instance_ids_from_file(file_path):
    """Collect instance_ids from a single parquet file."""
    try:
        table = pq.read_table(file_path, columns=['instance_id'])
        return set(table['instance_id'].to_pylist())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def get_all_instance_ids_parallel(base_dir, subdirs, max_workers=8):
    """Collect all unique instance_ids across all subdirectories using parallel processing."""
    all_files = []
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, "data", subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} does not exist, skipping")
            continue
        
        files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet')])
        all_files.extend(files)
    
    print(f"Found {len(all_files)} parquet files to process")
    
    all_instance_ids = set()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect_instance_ids_from_file, f): f for f in all_files}
        for future in as_completed(futures):
            instance_ids = future.result()
            all_instance_ids.update(instance_ids)
    
    return all_instance_ids

def deterministic_split(instance_ids, test_ratio, seed):
    """
    Deterministic split using hash-based assignment.
    This ensures the same instance_id always goes to the same split,
    regardless of processing order.
    """
    train_ids = set()
    test_ids = set()
    
    for instance_id in instance_ids:
        # Use hash for deterministic assignment
        hash_val = int(hashlib.md5(f"{seed}_{instance_id}".encode()).hexdigest(), 16)
        if (hash_val % 100) < (test_ratio * 100):
            test_ids.add(instance_id)
        else:
            train_ids.add(instance_id)
    
    return train_ids, test_ids

def split_and_save_parquet(source_dir, target_dir, subdir, instance_id_set, split_name):
    """Read parquet files, filter by instance_id set, and save to target directory."""
    source_path = os.path.join(source_dir, "data", subdir)
    target_path = os.path.join(target_dir, "data", subdir)
    
    if not os.path.exists(source_path):
        print(f"Warning: {source_path} does not exist, skipping")
        return
    
    os.makedirs(target_path, exist_ok=True)
    
    files = sorted([f for f in os.listdir(source_path) if f.endswith('.parquet')])
    total_rows_written = 0
    
    for file_idx, filename in enumerate(files):
        source_file = os.path.join(source_path, filename)
        target_file = os.path.join(target_path, filename)
        
        # Read the entire table
        table = pq.read_table(source_file)
        
        # Filter rows where instance_id is in the target set
        instance_ids_col = table['instance_id'].to_pylist()
        mask = [iid in instance_id_set for iid in instance_ids_col]
        
        if sum(mask) == 0:
            continue
        
        # Create filtered table
        filtered_table = table.filter(pa.array(mask))
        
        # Write to target
        pq.write_table(filtered_table, target_file)
        total_rows_written += filtered_table.num_rows
        
        print(f"  [{split_name}] {subdir}/{filename}: {filtered_table.num_rows} rows written")
    
    print(f"  [{split_name}] {subdir}: Total {total_rows_written} rows written")

def copy_metadata_files(source_dir, target_dir):
    """Copy non-data files (README, tools json, etc.) to target directories."""
    files_to_copy = ['README.md', 'openhands_tools.json', 'sweagent_tools.json', '.gitattributes']
    
    for filename in files_to_copy:
        source_file = os.path.join(source_dir, filename)
        if os.path.exists(source_file):
            for target_dir_path in [target_dir]:
                import shutil
                shutil.copy2(source_file, os.path.join(target_dir_path, filename))
                print(f"Copied {filename} to {target_dir_path}")

def process_subdir_for_split(args):
    """Process a single subdirectory for a specific split (train or test)."""
    source_dir, target_dir, subdir, instance_id_set, split_name = args
    
    source_path = os.path.join(source_dir, "data", subdir)
    target_path = os.path.join(target_dir, "data", subdir)
    
    if not os.path.exists(source_path):
        return f"Warning: {source_path} does not exist, skipping"
    
    os.makedirs(target_path, exist_ok=True)
    
    files = sorted([f for f in os.listdir(source_path) if f.endswith('.parquet')])
    total_rows_written = 0
    messages = []
    
    for filename in files:
        source_file = os.path.join(source_path, filename)
        target_file = os.path.join(target_path, filename)
        
        try:
            # Read the entire table
            table = pq.read_table(source_file)
            
            # Filter rows where instance_id is in the target set
            instance_ids_col = table['instance_id'].to_pylist()
            mask = [iid in instance_id_set for iid in instance_ids_col]
            
            if sum(mask) == 0:
                continue
            
            # Create filtered table
            filtered_table = table.filter(pa.array(mask))
            
            # Write to target
            pq.write_table(filtered_table, target_file)
            total_rows_written += filtered_table.num_rows
            
            messages.append(f"  [{split_name}] {subdir}/{filename}: {filtered_table.num_rows} rows")
        except Exception as e:
            messages.append(f"  [{split_name}] Error processing {filename}: {e}")
    
    messages.append(f"  [{split_name}] {subdir}: Total {total_rows_written} rows written")
    return "\n".join(messages)

def main():
    print("=" * 80)
    print("Open-SWE-Traces Train-Test Split")
    print("=" * 80)
    
    # Step 1: Collect all unique instance_ids (parallel)
    print("\n[Step 1] Collecting all unique instance_ids (parallel)...")
    all_instance_ids = get_all_instance_ids_parallel(SOURCE_DIR, SUBDIRS, max_workers=8)
    print(f"Total unique instance_ids: {len(all_instance_ids)}")
    
    # Step 2: Perform deterministic split
    print(f"\n[Step 2] Performing deterministic split (test_ratio={TEST_RATIO}, seed={RANDOM_SEED})...")
    train_ids, test_ids = deterministic_split(all_instance_ids, TEST_RATIO, RANDOM_SEED)
    print(f"Train set: {len(train_ids)} instance_ids ({len(train_ids)/len(all_instance_ids)*100:.1f}%)")
    print(f"Test set: {len(test_ids)} instance_ids ({len(test_ids)/len(all_instance_ids)*100:.1f}%)")
    
    # Verify no overlap
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping instance_ids!"
    print("✓ No overlap between train and test sets")
    
    # Step 3: Create output directories
    print("\n[Step 3] Creating output directories...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Step 4: Split and save parquet files (parallel processing per subdirectory)
    print("\n[Step 4] Splitting and saving parquet files...")
    
    # Prepare tasks for parallel processing
    tasks = []
    for subdir in SUBDIRS:
        tasks.append((SOURCE_DIR, TRAIN_DIR, subdir, train_ids, "TRAIN"))
        tasks.append((SOURCE_DIR, TEST_DIR, subdir, test_ids, "TEST"))
    
    # Process in parallel (but limit to avoid memory issues)
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_subdir_for_split, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            print(result)
    
    # Step 5: Copy metadata files
    print("\n[Step 5] Copying metadata files...")
    copy_metadata_files(SOURCE_DIR, TRAIN_DIR)
    copy_metadata_files(SOURCE_DIR, TEST_DIR)
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("Split completed successfully!")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
