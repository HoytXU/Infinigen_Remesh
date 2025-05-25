import os
import subprocess
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 8
CACHE_FILE = "relative_file_list.txt"
MAX_RETRIES = 3  # Maximum number of retry attempts for failed tasks

# Progress tracking files
def get_progress_files():
    rank = int(os.getenv("RANK", "0"))
    return {
        "completed": f"completed_tasks_rank{rank}.json",
        "failed": f"failed_tasks_rank{rank}.json"
    }

def load_progress():
    """Load previously completed and failed tasks"""
    progress_files = get_progress_files()
    completed_tasks = []
    failed_tasks = {}
    
    if os.path.exists(progress_files["completed"]):
        try:
            with open(progress_files["completed"], "r") as f:
                completed_tasks = json.load(f)
            print(f"📂 Loaded {len(completed_tasks)} completed tasks from {progress_files['completed']}")
        except json.JSONDecodeError:
            print(f"⚠️ Error loading completed tasks file. Starting with empty list.")
    
    if os.path.exists(progress_files["failed"]):
        try:
            with open(progress_files["failed"], "r") as f:
                failed_tasks = json.load(f)
            print(f"📂 Loaded {len(failed_tasks)} failed tasks from {progress_files['failed']}")
        except json.JSONDecodeError:
            print(f"⚠️ Error loading failed tasks file. Starting with empty list.")
    
    return completed_tasks, failed_tasks

def save_progress(completed_tasks, failed_tasks):
    """Save progress to disk"""
    progress_files = get_progress_files()
    
    with open(progress_files["completed"], "w") as f:
        json.dump(completed_tasks, f, indent=2)
    
    with open(progress_files["failed"], "w") as f:
        json.dump(failed_tasks, f, indent=2)

def get_tasks(limit=None, resume=True):
    if os.path.exists(CACHE_FILE):
        print(f"\U0001F4C4 Loading relative file list from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r") as f:
            relative_paths = [line.strip() for line in f if line.strip()]
    else:
        raise RuntimeError(f"{CACHE_FILE} not found. Please generate it beforehand.")

    # 根据环境变量自动划分任务（DLC 多节点）
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    chunk_size = (len(relative_paths) + world_size - 1) // world_size
    start = rank * chunk_size
    end = min((rank + 1) * chunk_size, len(relative_paths))
    relative_paths = relative_paths[start:end]

    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(relative_paths)} files)")

    REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
    all_tasks = [
        (
            os.path.join(BASE_DIR, rel_path),
            os.path.join(REMESH_DIR, "remeshes", rel_path)
        )
        for rel_path in relative_paths
    ]

    # Handle resuming from previous run if needed
    if resume:
        completed_tasks, failed_tasks = load_progress()
        
        # Filter out completed tasks
        pending_tasks = []
        for task in all_tasks:
            input_path, output_path = task
            if input_path in completed_tasks:
                print(f"⏭️ Skipping already completed: {input_path}")
                continue
                
            # Check if output already exists (as an additional check)
            if os.path.exists(output_path):
                if input_path not in completed_tasks:
                    completed_tasks.append(input_path)
                print(f"⏭️ Output exists, skipping: {input_path}")
                continue
                
            # Add failed tasks with their retry count
            if input_path in failed_tasks:
                if failed_tasks[input_path] >= MAX_RETRIES:
                    print(f"⚠️ Skipping max retries exceeded: {input_path} ({failed_tasks[input_path]} attempts)")
                    continue
                print(f"🔄 Retrying failed task: {input_path} (attempt {failed_tasks[input_path] + 1})")
            
            pending_tasks.append(task)
        
        # Save updated progress
        save_progress(completed_tasks, failed_tasks)
        tasks = pending_tasks
        print(f"📊 Resume mode: {len(all_tasks) - len(tasks)} skipped, {len(tasks)} to process")
    else:
        tasks = all_tasks
        # Initialize empty progress files
        save_progress([], {})
        print("🔄 Fresh start: previous progress ignored")

    if limit:
        tasks = tasks[:limit]
        print(f"\U0001F9EA Limiting to first {limit} tasks")

    print(f"✅ Total tasks to process: {len(tasks)}\n")
    return tasks

def run_blender_remesh(task):
    input_path, output_path = task
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 [START] {input_path}")
    cmd = [
        "blender", "--background",
        "--python", "remesh_worker.py",
        "--", input_path, output_path, str(VOXEL_SIZE)
    ]
    
    # Load current progress
    completed_tasks, failed_tasks = load_progress()
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ [DONE]  {output_path}")
        
        # Track successful completion
        if input_path not in completed_tasks:
            completed_tasks.append(input_path)
            
        # Remove from failed tasks if it was there
        if input_path in failed_tasks:
            del failed_tasks[input_path]
            
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL]  {input_path}")
        print(f"    ↳ stderr: {e.stderr.decode(errors='ignore')[:200]}...")  # 部分报错信息
        
        # Track failure and retry count
        if input_path in failed_tasks:
            failed_tasks[input_path] += 1
        else:
            failed_tasks[input_path] = 1
    
    # Save updated progress
    save_progress(completed_tasks, failed_tasks)
    
    return input_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, help="Limit number of files to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore previous progress and start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry previously failed tasks")
    args = parser.parse_args()

    print("📋 Preparing remesh job list...\n")
    
    # Handle retry-failed option
    if args.retry_failed:
        progress_files = get_progress_files()
        if not os.path.exists(progress_files["failed"]):
            print("❌ No failed tasks record found. Exiting.")
            exit(1)
            
        with open(progress_files["failed"], "r") as f:
            failed_tasks = json.load(f)
            
        if not failed_tasks:
            print("✅ No failed tasks to retry. Exiting.")
            exit(0)
            
        print(f"🔄 Retrying {len(failed_tasks)} previously failed tasks")
        
        # Reconstruct tasks from failed tasks
        REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
        tasks = []
        for input_path in failed_tasks:
            rel_path = os.path.relpath(input_path, BASE_DIR)
            output_path = os.path.join(REMESH_DIR, "remeshes", rel_path)
            tasks.append((input_path, output_path))
    else:
        # Normal mode with optional resume
        tasks = get_tasks(limit=args.max, resume=not args.no_resume)

    if not tasks:
        print("✅ No tasks to process. All tasks may have already completed. Exiting.")
        exit(0)

    print(f"🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")

    start_time = time.time()
    completed_count = 0
    failed_count = 0
    
    try:
        with Pool(processes=MAX_PROCESSES) as pool:
            results = pool.map(run_blender_remesh, tasks)
            
        # Count completed and failed tasks
        progress_files = get_progress_files()
        if os.path.exists(progress_files["completed"]):
            with open(progress_files["completed"], "r") as f:
                completed_count = len(json.load(f))
                
        if os.path.exists(progress_files["failed"]):
            with open(progress_files["failed"], "r") as f:
                failed_count = len(json.load(f))
                
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All tasks processed in {elapsed_time:.2f} seconds.")
        print(f"📊 Summary: {completed_count} completed, {failed_count} failed")
        
        if failed_count > 0:
            print(f"⚠️ Some tasks failed. Run with --retry-failed to retry them.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Progress has been saved and can be resumed.")
        print("   Run the script again to resume from where you left off.")

