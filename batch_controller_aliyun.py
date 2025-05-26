import os
import subprocess
import argparse
import threading
import time
from multiprocessing import Pool, Manager

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 8
CACHE_FILE = "relative_file_list.txt"

# Define the remesh directory and done file path
REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
REMESH_OUTPUT_DIR = os.path.join(REMESH_DIR, "remeshes_v3")
DONE_FILE = os.path.join(REMESH_OUTPUT_DIR, "done.txt")

# Flush interval in seconds (5 minutes)
FLUSH_INTERVAL = 5 * 60

# Global done list
done_paths = set()

def read_done_file():
    """Read the done.txt file and return a set of completed paths"""
    if not os.path.exists(DONE_FILE):
        return set()
        
    completed = set()
    try:
        with open(DONE_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    completed.add(line)
        print(f"📂 Loaded {len(completed)} completed tasks from {DONE_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not read done file: {e}")
    
    return completed

def flush_done_paths(stop_event=None):
    """Flush the global done_paths to the done file"""
    global done_paths
    
    if not os.path.exists(os.path.dirname(DONE_FILE)):
        os.makedirs(os.path.dirname(DONE_FILE), exist_ok=True)
    
    # Initial wait before first flush
    if stop_event:
        stop_event.wait(10)  # Wait 10 seconds before first flush
    
    while True:
        current_done = done_paths.copy()  # Copy to avoid modifying while iterating
        
        if current_done:
            # Read existing done items to avoid duplicates
            existing_done = read_done_file()
            new_items = current_done - existing_done
            
            if new_items:
                try:
                    rank = os.getenv("RANK", "0")
                    print(f"💾 [RANK {rank}] Flushing {len(new_items)} completed tasks to {DONE_FILE}")
                    
                    with open(DONE_FILE, 'a') as f:
                        for item in new_items:
                            f.write(f"{item}\n")
                except Exception as e:
                    print(f"⚠️ Error flushing done paths: {e}")
        
        # Break if we're shutting down and not in daemon mode
        if stop_event and stop_event.is_set():
            break
            
        # Wait for next flush interval
        if stop_event:
            stop_event.wait(FLUSH_INTERVAL)
        else:
            # One-time flush when called directly
            break

def get_tasks(limit=None):
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
        
        # Sort paths for consistent ordering
        relative_paths.sort()
        # Cache the results for future runs
        with open(CACHE_FILE, "w") as f:
            f.write("\n".join(relative_paths))
        print(f"\U0001F4C4 Saved {len(relative_paths)} paths to '{CACHE_FILE}'")

    # 根据环境变量自动划分任务（DLC 多节点）
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Load the done file first
    global done_paths
    done_paths = read_done_file()
    
    # Filter out completed tasks
    original_count = len(relative_paths)
    relative_paths = [p for p in relative_paths if p not in done_paths]
    skipped = original_count - len(relative_paths)
    if skipped > 0:
        print(f"⏭️ Skipping {skipped} already completed tasks")
    
    # Use round-robin allocation instead of chunking
    relative_paths = [p for i, p in enumerate(relative_paths) if i % world_size == rank]

    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(relative_paths)} files)")

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
    
    # Check if the output already exists to avoid redundant work
    if os.path.exists(output_path):
        # Mark as done
        rel_path = os.path.relpath(input_path, BASE_DIR)
        done_paths.add(rel_path)
        print(f"⏩ [SKIP]  {output_path} (already exists)")
        return
        
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
        
        # Mark as done
        rel_path = os.path.relpath(input_path, BASE_DIR)
        done_paths.add(rel_path)
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL]  {input_path}")
        print(f"    ↳ stderr: {e.stderr.decode(errors='ignore')[:200]}...")  # 部分报错信息

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, help="Limit number of files to process")
    args = parser.parse_args()

    print("📋 Preparing remesh job list...\n")
    tasks = get_tasks(limit=args.max)

    if not tasks:
        print("❌ No tasks to process. Exiting.")
        exit(1)
        
    # Start the flush thread
    stop_event = threading.Event()
    flush_thread = threading.Thread(
        target=flush_done_paths,
        args=(stop_event,),
        daemon=True
    )
    flush_thread.start()

    print(f"🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")

    try:
        with Pool(processes=MAX_PROCESSES) as pool:
            pool.map(run_blender_remesh, tasks)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Waiting for final flush...")
    finally:
        # Signal the flush thread to stop and do a final flush
        stop_event.set()
        flush_thread.join(timeout=30)  # Wait up to 30 seconds for final flush
        # Do one more flush directly to ensure everything is saved
        flush_done_paths()

    print("\n🎉 All tasks completed.")

