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
    tasks = [
        (
            os.path.join(BASE_DIR, rel_path),
            os.path.join(REMESH_DIR, "remeshes", rel_path)
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

