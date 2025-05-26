import os
import subprocess
import argparse
from multiprocessing import Pool

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 8
CACHE_FILE = "relative_file_list.txt"

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

    print(f"🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")

    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(run_blender_remesh, tasks)

    print("\n🎉 All tasks completed.")

