import os
import subprocess
import argparse
from multiprocessing import Pool
import json
from pathlib import Path

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 8
CACHE_FILE = "relative_file_list.txt"
COMPLETED_FILE = "completed_tasks.json"

def get_completed_tasks():
    """Load the set of completed tasks from the completed tasks file."""
    if os.path.exists(COMPLETED_FILE):
        try:
            with open(COMPLETED_FILE, 'r') as f:
                completed_tasks = set(json.load(f))
                print(f"📂 Loaded {len(completed_tasks)} completed tasks from tracking file")
                return completed_tasks
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Warning: Could not read completed tasks file: {e}")
            return set()
    else:
        print("👤 First run - no completed tasks file found")
    return set()

def save_completed_task(input_path):
    """Mark a task as completed by adding it to the completed tasks file."""
    completed = get_completed_tasks()
    # Add the relative path to the completed set
    rel_path = os.path.relpath(input_path, BASE_DIR)
    completed.add(rel_path)
    
    # Save the updated set
    try:
        with open(COMPLETED_FILE, 'w') as f:
            json.dump(list(completed), f)
    except IOError as e:
        print(f"⚠️ Warning: Could not update completed tasks file: {e}")

def get_tasks(limit=None, force_rerun=False):
    if os.path.exists(CACHE_FILE):
        print(f"\U0001F4C4 Loading relative file list from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r") as f:
            relative_paths = [line.strip() for line in f if line.strip()]
    else:
        # Use os.walk to generate the file list more efficiently
        print(f"\U0001F4C4 Generating relative file list using os.walk...")
        relative_paths = []
        for root, dirs, files in os.walk(BASE_DIR):
            for file in files:
                if file.endswith('.obj') or file.endswith('.ply'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, BASE_DIR)
                    relative_paths.append(rel_path)
        
        # Cache the results for future runs
        with open(CACHE_FILE, "w") as f:
            f.write("\n".join(relative_paths))
        print(f"\U0001F4C4 Saved {len(relative_paths)} paths to '{CACHE_FILE}'")

    # 根据环境变量自动划分任务（DLC 多节点）
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Use round-robin allocation instead of chunking
    relative_paths = [p for i, p in enumerate(relative_paths) if i % world_size == rank]

    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(relative_paths)} files)")
    
    # Filter out completed tasks unless force_rerun is True
    if not force_rerun:
        completed_tasks = get_completed_tasks()
        original_count = len(relative_paths)
        relative_paths = [p for p in relative_paths if p not in completed_tasks]
        skipped = original_count - len(relative_paths)
        if skipped > 0:
            print(f"⏭️ Skipping {skipped} already completed tasks ({len(completed_tasks)} total completed)")
    elif os.path.exists(COMPLETED_FILE):
        print(f"🔄 Force rerun requested - ignoring {COMPLETED_FILE}")

    REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
    tasks = [
        (
            os.path.join(BASE_DIR, rel_path),
            os.path.join(REMESH_DIR, "remeshes_v3", rel_path)
        )
        for rel_path in relative_paths
    ]

    if limit:
        tasks = tasks[:limit]
        print(f"\U0001F9EA Limiting to first {limit} tasks")

    print(f"✅ Total collected tasks: {len(tasks)}\n")
    return tasks

def run_blender_remesh(task):
    input_path, output_path = task
    
    # No need to check file existence - completed tasks were already filtered out
    # during task list generation in get_tasks()
    print(f"🚀 [START] {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "blender", "--background",
        "--python", "remesh_worker.py",
        "--", input_path, output_path, str(VOXEL_SIZE)
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ [DONE]  {output_path}")
        # Mark task as completed
        save_completed_task(input_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL]  {input_path}")
        print(f"    ↳ stderr: {e.stderr.decode(errors='ignore')[:200]}...")  # 部分报错信息

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, help="Limit number of files to process")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files, ignoring completed tasks")
    args = parser.parse_args()

    print("📋 Preparing remesh job list...\n")
    tasks = get_tasks(limit=args.max, force_rerun=args.force)

    if not tasks:
        print("❌ No tasks to process. Exiting.")
        exit(1)

    print(f"🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")

    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(run_blender_remesh, tasks)

    print("\n🎉 All tasks completed.")

