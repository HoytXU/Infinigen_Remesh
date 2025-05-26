import os
import subprocess
import argparse
import json
import time
import sys
from multiprocessing import Pool
from pathlib import Path

# Try importing tqdm for progress bars, provide fallback if not installed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not installed, basic progress reporting will be used.")
    print("   You can install it with: pip install tqdm")

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 8
CACHE_FILE = "relative_file_list.txt"
MAX_RETRIES = 3  # Maximum number of retry attempts for failed tasks

# Default directory for progress tracking files (can be overridden via command-line)
# Will be updated from command-line arguments
PROGRESS_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/remeshes_v2/progress"  # Current directory by default

# if path not exist, create it
os.makedirs(PROGRESS_DIR, exist_ok=True)    

# Progress tracking files
def get_progress_files():
    """Get paths to progress tracking files based on rank and progress directory"""
    global PROGRESS_DIR
    rank = int(os.getenv("RANK", "0"))
    
    # Ensure progress directory exists
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    return {
        "completed": os.path.join(PROGRESS_DIR, f"completed_tasks_rank{rank}.json"),
        "failed": os.path.join(PROGRESS_DIR, f"failed_tasks_rank{rank}.json"),
        "stats": os.path.join(PROGRESS_DIR, f"task_stats_rank{rank}.json")
    }

def load_progress(auto_detect=False):
    """Load previously completed and failed tasks
    
    Args:
        auto_detect (bool): If True, automatically detect completed tasks by scanning for output files
    """
    progress_files = get_progress_files()
    completed_tasks = []
    failed_tasks = {}
    
    # Load from progress files if they exist
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
    
    # Auto-detect previously completed tasks if requested
    if auto_detect and not completed_tasks:
        print("🔍 Auto-detecting previously completed tasks...")
        auto_detected = detect_completed_tasks()
        if auto_detected:
            completed_tasks = auto_detected
            print(f"💾 Auto-detected {len(completed_tasks)} completed tasks from existing output files")
            # Save the auto-detected list
            save_progress(completed_tasks, failed_tasks)
    
    return completed_tasks, failed_tasks

def detect_completed_tasks():
    """Auto-detect completed tasks by scanning for existing output files (optimized version)"""
    # Get all possible input files
    with open(CACHE_FILE, "r") as f:
        all_relative_paths = [line.strip() for line in f if line.strip()]
    
    REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
    output_base_dir = os.path.join(REMESH_DIR, "remeshes_v2")
    completed = []
    
    # Check if output base directory exists before scanning
    if not os.path.exists(output_base_dir):
        print(f"⚠️ Output directory {output_base_dir} does not exist yet. No completed files to detect.")
        return completed
    
    print(f"🔍 Scanning output directory for existing files...")
    start_time = time.time()
    
    # Get a set of all existing output files (much faster than checking one by one)
    existing_outputs = set()
    try:
        # Use os.walk to efficiently scan the directory tree
        for root, dirs, files in os.walk(output_base_dir):
            # Get relative path from output base directory
            rel_root = os.path.relpath(root, output_base_dir)
            
            # Skip the root directory in the path calculation
            if rel_root == '.':
                rel_root = ''
                
            # Add all files with their relative paths
            for file in files:
                rel_path = os.path.join(rel_root, file)
                existing_outputs.add(rel_path)
                
        print(f"📊 Found {len(existing_outputs)} files in output directory (scan took {time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"⚠️ Error scanning output directory: {str(e)}")
        return completed
    
    # Match input files with existing output files
    match_count = 0
    print("🔄 Matching with input files...")
    match_start = time.time()
    
    for rel_path in all_relative_paths:
        input_path = os.path.join(BASE_DIR, rel_path)
        
        # Check if this file exists in our set of output files
        if rel_path in existing_outputs:
            completed.append(input_path)
            match_count += 1
    
    print(f"✅ Matched {match_count} completed files (matching took {time.time() - match_start:.2f}s)")
    return completed

def save_progress(completed_tasks, failed_tasks):
    """Save progress to disk"""
    progress_files = get_progress_files()
    
    with open(progress_files["completed"], "w") as f:
        json.dump(completed_tasks, f, indent=2)
    
    with open(progress_files["failed"], "w") as f:
        json.dump(failed_tasks, f, indent=2)

def get_file_size(file_path):
    """Get file size in bytes, returns 0 if file doesn't exist"""
    try:
        return os.path.getsize(file_path)
    except (FileNotFoundError, OSError):
        return 0

def get_tasks(limit=None, resume=True, balance_by_size=False):
    """Get tasks to process with optional size-based distribution"""
    if os.path.exists(CACHE_FILE):
        print(f"\U0001F4C4 Loading relative file list from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r") as f:
            relative_paths = [line.strip() for line in f if line.strip()]
    else:
        raise RuntimeError(f"{CACHE_FILE} not found. Please generate it beforehand.")

    # 根据环境变量自动划分任务（DLC 多节点）
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    # Generate the full paths for all relative paths
    REMESH_DIR = BASE_DIR.rsplit("meshes", 1)[0]
    all_paths = []
    for rel_path in relative_paths:
        input_path = os.path.join(BASE_DIR, rel_path)
        output_path = os.path.join(REMESH_DIR, "remeshes_v2", rel_path)
        all_paths.append((input_path, output_path, rel_path))
    
    if balance_by_size and world_size > 1:
        # Size-based load balancing
        print("⚖️ Using file size-based load balancing...")
        
        # Get file sizes and sort paths by size (largest first)
        sized_paths = [(get_file_size(input_path), input_path, output_path, rel_path) 
                      for input_path, output_path, rel_path in all_paths]
        sized_paths.sort(reverse=True)  # Sort by size, largest first
        
        # Distribute tasks in a round-robin fashion by size
        node_assignments = [[] for _ in range(world_size)]
        for i, (size, input_path, output_path, rel_path) in enumerate(sized_paths):
            # Assign to node with least total size
            target_node = i % world_size
            node_assignments[target_node].append((input_path, output_path, rel_path))
        
        # Get paths for this rank
        assigned_paths = node_assignments[rank]
        print(f"📊 Size-balanced assignment: {len(assigned_paths)} files for rank {rank}")
        
        # Extract just input and output paths
        all_paths = [(input_path, output_path, rel_path) for input_path, output_path, rel_path in assigned_paths]
    else:
        # Standard index-based partitioning
        chunk_size = (len(all_paths) + world_size - 1) // world_size
        start = rank * chunk_size
        end = min((rank + 1) * chunk_size, len(all_paths))
        all_paths = all_paths[start:end]
    
    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(all_paths)} files)")
    
    # Convert to task format (input_path, output_path)
    all_tasks = [(input_path, output_path) for input_path, output_path, _ in all_paths]
    
    print(f"\U0001F7A9 Assigned task slice for RANK={rank} (total {len(all_tasks)} files)")

    # Handle resuming from previous run if needed
    if resume:
        # Auto-detect completed files if this is the first run with resumable feature
        completed_tasks, failed_tasks = load_progress(auto_detect=True)
        
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
    
    # Don't print here if using tqdm to avoid breaking progress bar
    if not TQDM_AVAILABLE or 'progress_bar' not in globals():
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
        
        # Don't print here if using tqdm to avoid breaking progress bar
        if not TQDM_AVAILABLE or 'progress_bar' not in globals():
            print(f"✅ [DONE]  {output_path}")
        
        # Track successful completion
        if input_path not in completed_tasks:
            completed_tasks.append(input_path)
            
        # Remove from failed tasks if it was there
        if input_path in failed_tasks:
            del failed_tasks[input_path]
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode(errors='ignore')[:200] + "..." if len(e.stderr) > 200 else e.stderr.decode(errors='ignore')
        
        # Don't print here if using tqdm to avoid breaking progress bar
        if not TQDM_AVAILABLE or 'progress_bar' not in globals():
            print(f"❌ [FAIL]  {input_path}")
            print(f"    ↳ stderr: {error_msg}")  # 部分报错信息
        
        # Log errors to a dedicated file for better debugging
        try:
            with open(os.path.join(PROGRESS_DIR, "error_log.txt"), "a") as f:
                f.write(f"ERROR [{time.strftime('%Y-%m-%d %H:%M:%S')}] - {input_path}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Error: {error_msg}\n\n")
        except Exception:
            pass  # Silently fail if error logging fails
        
        # Track failure and retry count
        if input_path in failed_tasks:
            failed_tasks[input_path] += 1
        else:
            failed_tasks[input_path] = 1
    
    # Save updated progress
    save_progress(completed_tasks, failed_tasks)
    
    return input_path

# Function to run blender remesh using dynamic queue for better load balancing
def process_tasks_dynamic(tasks, num_processes=MAX_PROCESSES):
    """Process tasks using a dynamic work stealing approach for better load balancing"""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Create a task queue
    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    
    # Tracking variables
    completed = []
    failed = {}
    lock = threading.Lock()
    active_count = threading.Semaphore(0)
    total_tasks = len(tasks)
    processed_count = 0
    
    # Create progress bar
    if TQDM_AVAILABLE:
        progress_bar = tqdm(total=total_tasks, desc="Processing", unit="file", 
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    # Progress tracking
    def update_progress(input_path, success):
        nonlocal processed_count
        processed_count += 1
        
        # Update the progress bar if available
        if TQDM_AVAILABLE:
            status = "✅" if success else "❌"
            progress_bar.set_postfix_str(f"{status} {os.path.basename(input_path)}")
            progress_bar.update(1)
        else:
            # Fallback to simple progress indicator
            status = "Done" if success else "Failed"
            print(f"\r🔄 Progress: {processed_count}/{total_tasks} ({processed_count/total_tasks*100:.1f}%) - {status}: {os.path.basename(input_path)}", end="", flush=True)
    
    # Worker function
    def worker():
        while True:
            try:
                # Get a task from the queue
                task = task_queue.get(block=False)
                active_count.release()  # Signal that a task is being processed
                
                # Process the task
                input_path, output_path = task
                success = False
                
                try:
                    # Run the actual processing
                    run_blender_remesh(task)
                    success = True
                except Exception as e:
                    if TQDM_AVAILABLE:
                        progress_bar.write(f"🔥 Unexpected error processing {input_path}: {str(e)}")
                    else:
                        print(f"\n🔥 Unexpected error processing {input_path}: {str(e)}")
                
                # Update tracking
                with lock:
                    if success:
                        completed.append(input_path)
                    else:
                        if input_path in failed:
                            failed[input_path] += 1
                        else:
                            failed[input_path] = 1
                
                # Update progress
                update_progress(input_path, success)
                
                # Mark as done
                task_queue.task_done()
                active_count.acquire()  # Signal that a task is finished
                
            except queue.Empty:
                # No more tasks
                break
    
    print(f"🧩 Starting dynamic task processing with {num_processes} workers...")
    start_time = time.time()
    
    # Create and start worker threads
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        workers = [executor.submit(worker) for _ in range(num_processes)]
        
        try:
            # Wait for all tasks to complete
            while not task_queue.empty() or active_count._value > 0:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user. Saving progress...")
        finally:
            # Save progress
            progress_files = get_progress_files()
            with open(progress_files["completed"], "w") as f:
                json.dump(completed, f, indent=2)
            with open(progress_files["failed"], "w") as f:
                json.dump(failed, f, indent=2)
    
    # Close progress bar if using tqdm
    if TQDM_AVAILABLE:
        progress_bar.close()
        
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Processing completed in {elapsed_time:.2f} seconds")
    print(f"📈 Results: {len(completed)} completed, {len(failed)} failed")
    
    return completed, failed

def log_system_info():
    """Log system information to help with debugging"""
    import platform
    
    print("📃 System Information:")
    print(f"   - OS: {platform.system()} {platform.version()}")
    print(f"   - Python: {platform.python_version()}")
    print(f"   - CPU Cores: {os.cpu_count()}")
    print(f"   - Progress tracking: {'tqdm' if TQDM_AVAILABLE else 'basic'}")
    print()

def main():
    global PROGRESS_DIR  # Properly declare global before assignment
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, help="Limit number of files to process")
    parser.add_argument("--no-resume", action="store_true", help="Ignore previous progress and start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry previously failed tasks")
    parser.add_argument("--progress-dir", type=str, default=".", help="Directory to store progress tracking files")
    parser.add_argument("--balance-by-size", action="store_true", help="Balance workload by file size instead of count")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic work stealing for better CPU utilization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, help="Path to log file (default: progress_dir/batch_log.txt)")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect completed tasks from existing output files")
    args = parser.parse_args()
    
    # Update progress directory from command line
    PROGRESS_DIR = args.progress_dir
    
    # Set up logging to file if requested
    if args.log_file or args.verbose:
        log_file = args.log_file or os.path.join(PROGRESS_DIR, "batch_log.txt")
        print(f"💾 Logging to {log_file}")
        
        # Create a tee-like function to log to both file and stdout
        original_stdout = sys.stdout
        log_file_handle = open(log_file, 'a')
        
        class TeeLogger:
            def write(self, message):
                original_stdout.write(message)
                log_file_handle.write(message)
                log_file_handle.flush()
            def flush(self):
                original_stdout.flush()
                log_file_handle.flush()
        
        sys.stdout = TeeLogger()
    
    # Log system info if verbose
    if args.verbose:
        log_system_info()

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
            output_path = os.path.join(REMESH_DIR, "remeshes_v2", rel_path)
            tasks.append((input_path, output_path))
    else:
        # Normal mode with optional resume
        tasks = get_tasks(limit=args.max, resume=not args.no_resume, balance_by_size=args.balance_by_size)

    if not tasks:
        print("✅ No tasks to process. All tasks may have already completed. Exiting.")
        exit(0)

    print(f"🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")

    start_time = time.time()
    completed_count = 0
    failed_count = 0
    
    try:
        if args.dynamic:
            # Use dynamic work stealing approach
            completed, failed = process_tasks_dynamic(tasks, num_processes=MAX_PROCESSES)
        else:
            # Use standard multiprocessing pool with progress bar
            if TQDM_AVAILABLE:
                with Pool(processes=MAX_PROCESSES) as pool:
                    results = list(tqdm(pool.imap(run_blender_remesh, tasks), 
                                        total=len(tasks),
                                        desc="Processing", 
                                        unit="file",
                                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"))
            else:
                # Fallback to regular pool without progress bar
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
                
        # Generate detailed summary report
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All tasks processed in {elapsed_time:.2f} seconds.")
        print(f"📊 Summary: {completed_count} completed, {failed_count} failed")
        
        # Calculate processing rate
        if elapsed_time > 0:
            rate = (completed_count + failed_count) / elapsed_time
            print(f"⏱️ Processing rate: {rate:.2f} files/second")
        
        if failed_count > 0:
            print(f"⚠️ Some tasks failed. Run with --retry-failed to retry them.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Progress has been saved and can be resumed.")
        print("   Run the script again to resume from where you left off.")

if __name__ == "__main__":
    main()

