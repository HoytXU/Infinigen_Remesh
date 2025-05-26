import os
import subprocess
import argparse
import threading
import time
import multiprocessing
from multiprocessing import Pool, Manager
from functools import lru_cache

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
# Dynamically adjust MAX_PROCESSES based on CPU cores, but keep within reasonable limits
MAX_PROCESSES = min(multiprocessing.cpu_count(), 8)
CACHE_FILE = "relative_file_list.txt"

# Define the remesh directory and done file path
REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
REMESH_OUTPUT_DIR = os.path.join(REMESH_DIR, "remeshes_v3")
DONE_FILE = os.path.join(REMESH_OUTPUT_DIR, "done.txt")

# Flush interval in seconds (5 minutes)
FLUSH_INTERVAL = 5 * 60

# Global done list and cache of existing files
done_paths = set()
existing_output_files = set()

# Cache for dirname calls
@lru_cache(maxsize=10000)
def cached_dirname(path):
    return os.path.dirname(path)

def read_done_file():
    """Read the done.txt file and return a set of completed paths"""
    if not os.path.exists(DONE_FILE):
        return set()
        
    completed = set()
    try:
        # Use a more efficient file reading approach
        with open(DONE_FILE, 'r') as f:
            # Read in larger chunks for better I/O performance
            chunk_size = 8192  # 8KB chunks
            buffer = ""
            chunk = f.read(chunk_size)
            
            while chunk:
                buffer += chunk
                lines = buffer.split('\n')
                # Process all complete lines except possibly the last one
                for line in lines[:-1]:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        completed.add(line)
                        
                # Keep the last potentially incomplete line for the next iteration
                buffer = lines[-1]
                chunk = f.read(chunk_size)
                
            # Don't forget the last line if it's not empty
            if buffer.strip() and not buffer.strip().startswith('#'):
                completed.add(buffer.strip())
                
        print(f"📂 Loaded {len(completed)} completed tasks from {DONE_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: Could not read done file: {e}")
    
    return completed

def flush_done_paths(stop_event=None):
    """Flush the global done_paths to the done file"""
    global done_paths
    
    # Create directory once at the beginning
    try:
        os.makedirs(os.path.dirname(DONE_FILE), exist_ok=True)
    except Exception as e:
        print(f"⚠️ Error creating done file directory: {e}")
    
    # Initial wait before first flush
    if stop_event:
        stop_event.wait(10)  # Wait 10 seconds before first flush
    
    # Read existing done items only once at the start
    existing_done = read_done_file() if os.path.exists(DONE_FILE) else set()
    last_flushed = set(existing_done)  # Track what we've already written
    
    while True:
        current_done = done_paths.copy()  # Copy to avoid modifying while iterating
        
        if current_done:
            # Only write new items since last flush (avoid redundant file reads)
            new_items = current_done - last_flushed
            
            if new_items:
                try:
                    rank = os.getenv("RANK", "0")
                    print(f"💾 [RANK {rank}] Flushing {len(new_items)} completed tasks to {DONE_FILE}")
                    
                    # Batch write for better performance
                    with open(DONE_FILE, 'a') as f:
                        f.write('\n'.join(new_items) + '\n')
                    
                    # Update what we've flushed
                    last_flushed.update(new_items)
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
    global REMESH_DIR, REMESH_OUTPUT_DIR, done_paths, existing_output_files
    if os.path.exists(CACHE_FILE):
        print(f"\U0001F4C4 Loading relative file list from '{CACHE_FILE}'...")
        # More efficient file reading
        relative_paths = []
        with open(CACHE_FILE, "r") as f:
            # Read in chunks for better performance
            for chunk in iter(lambda: f.read(65536), ''):
                relative_paths.extend(line.strip() for line in chunk.splitlines() if line.strip())
    else:
        # Use os.walk to generate the file list more efficiently
        print(f"\U0001F4C4 Generating relative file list using os.walk...")
        relative_paths = []
        
        # Filter for valid extensions once
        valid_extensions = ('.obj', '.ply')
        
        # Use a more efficient loop
        for root, dirs, files in os.walk(BASE_DIR):
            # Skip hidden directories for efficiency
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith(valid_extensions):
                    # Calculate paths more efficiently
                    rel_path = os.path.relpath(os.path.join(root, file), BASE_DIR)
                    relative_paths.append(rel_path)
        
        # Sort paths for consistent ordering
        relative_paths.sort()
        
        # Cache the results for future runs - ensure parent directory exists
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        except (OSError, IOError):
            pass  # Ignore if we can't create the directory
            
        # Write in a single operation for better performance
        with open(CACHE_FILE, "w") as f:
            f.write("\n".join(relative_paths))
        print(f"\U0001F4C4 Saved {len(relative_paths)} paths to '{CACHE_FILE}'")

    # 根据环境变量自动划分任务（DLC 多节点）
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Load the done file first
    done_paths = read_done_file()
    
    # Cache existing output files to avoid expensive os.path.exists checks later
    output_dir = os.path.join(REMESH_DIR, "remeshes_v3")
    if os.path.exists(output_dir):
        print(f"🔍 Scanning output directory for existing files...")
        for root, dirs, files in os.walk(output_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith(('.obj', '.ply')):  # Same extensions as input
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    parent_dir = os.path.dirname(rel_path)
                    # Add both the file and its parent directory to our cache
                    existing_output_files.add(rel_path)
        print(f"💾 Found {len(existing_output_files)} existing output files")
    
    # Filter out completed tasks - use done_paths which is faster than file checks
    original_count = len(relative_paths)
    relative_paths = [p for p in relative_paths if p not in done_paths]
    skipped = original_count - len(relative_paths)
    if skipped > 0:
        print(f"⏭️ Skipping {skipped} already completed tasks")
    
    # Use round-robin allocation instead of chunking
    relative_paths = [p for i, p in enumerate(relative_paths) if i % world_size == rank]

    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(relative_paths)} files)")

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
    
    # Get relative path once and reuse
    rel_path = os.path.relpath(input_path, BASE_DIR)
    
    # Check if the output already exists using our cached data instead of filesystem
    rel_output = os.path.relpath(output_path, os.path.join(REMESH_DIR, "remeshes_v3"))
    if rel_output in existing_output_files or os.path.exists(output_path):
        # Mark as done
        done_paths.add(rel_path)
        print(f"⏩ [SKIP]  {output_path} (already exists)")
        return
        
    print(f"🚀 [START] {input_path}")
    
    # Create output directory if it doesn't exist - use cached dirname
    output_dir = cached_dirname(output_path)
    try:
        # Direct try/except is faster than checking first
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        # Already exists or other error we can ignore
        pass
    
    cmd = [
        "blender", "--background",
        "--python", "remesh_worker.py",
        "--", input_path, output_path, str(VOXEL_SIZE)
    ]
    try:
        # We already calculated rel_path at the start of the function
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ [DONE]  {output_path}")
        
        # Mark as done - using already calculated rel_path
        done_paths.add(rel_path)
        
        # Also add to existing files cache to avoid future filesystem checks
        rel_output = os.path.relpath(output_path, os.path.join(REMESH_DIR, "remeshes_v3"))
        existing_output_files.add(rel_output)
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
        
    # Pre-create all needed output directories in one pass
    print("💾 Pre-creating output directories...")
    needed_dirs = set()
    for _, output_path in tasks:
        needed_dirs.add(cached_dirname(output_path))
    
    # Create directories in batches
    for dir_path in needed_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            # Just log and continue if we can't create a directory
            print(f"⚠️ Could not create directory {dir_path}: {e}")
        
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
