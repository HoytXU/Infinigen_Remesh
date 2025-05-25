import os
import glob
import subprocess
import argparse
from multiprocessing import Pool

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
TARGET_FACES = 50000
MAX_PROCESSES = 4 
CACHE_FILE = "relative_file_list.txt"

def get_tasks(limit=None):
    if os.path.exists(CACHE_FILE):
        print(f"📄 Loading relative file list from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r") as f:
            relative_paths = [line.strip() for line in f]
    else:
        print(f"📦 Scanning directories under {BASE_DIR} ...")
        relative_paths = []
        factory_dirs = sorted(d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)))
        total = len(factory_dirs)

        for idx, factory in enumerate(factory_dirs):
            print(f"[{idx:02}/{total}] {factory}")
            factory_path = os.path.join(BASE_DIR, factory)
            samples = sorted(os.listdir(factory_path))

            for sample in samples:
                rel_path = os.path.join(factory, sample)
                abs_path = os.path.join(BASE_DIR, rel_path)
                if os.path.isfile(abs_path):
                    print(f"    ├── {sample} → ✅ found")
                    relative_paths.append(rel_path)
                else:
                    print(f"    ├── {sample} → ❌ missing")

        print(f"💾 Caching {len(relative_paths)} paths to '{CACHE_FILE}'...")
        with open(CACHE_FILE, "w") as f:
            for path in relative_paths:
                f.write(path + "\n")

    # 生成任务列表（输入路径 + 输出路径）
    REMESH_DIR=BASE_DIR.rsplit("meshes", 1)[0]
    tasks = [
        (
            os.path.join(BASE_DIR, rel_path),
            os.path.join(REMESH_DIR, "remeshes", rel_path)
        )
        for rel_path in relative_paths
    ]

    if limit:
        tasks = tasks[:limit]
        print(f"🧪 Limiting to first {limit} tasks")

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

