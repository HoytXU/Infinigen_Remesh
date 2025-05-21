import os
import random
import subprocess
import re
import argparse
from collections import defaultdict
from multiprocessing import Pool
import statistics

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
CACHE_FILE = "relative_file_list.txt"
VOXEL_SIZE = 0.005
MAX_PROCESSES = 8

def load_all_paths():
    with open(CACHE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    category_paths = defaultdict(list)
    for line in lines:
        cat = line.split("/")[0]
        category_paths[cat].append(line)
    return category_paths

def run_and_extract_time(rel_path):
    cat = rel_path.split("/")[0]
    input_path = os.path.join(BASE_DIR, rel_path)
    output_path = input_path.replace("meshes", "remeshes")

    cmd = [
        "blender", "--background",
        "--python", "remesh_worker.py",
        "--", input_path, output_path, str(VOXEL_SIZE)
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = result.stdout.decode(errors="ignore")
        match = re.search(r"🧠 TOTAL TIME\s+:\s+([\d\.]+)", stdout)
        if match:
            return cat, float(match.group(1))
    except subprocess.CalledProcessError:
        print(f"❌ [FAIL] {input_path}")
    return cat, None

def benchmark(sample_per_class):
    cat_paths = load_all_paths()
    total_tasks = []

    for cat, paths in cat_paths.items():
        n = min(sample_per_class, len(paths))
        sampled = random.sample(paths, n)
        total_tasks.extend(sampled)

    print(f"🧪 Benchmarking {len(total_tasks)} tasks from {len(cat_paths)} categories...\n")

    with Pool(processes=MAX_PROCESSES) as pool:
        results = pool.map(run_and_extract_time, total_tasks)

    timing_by_cat = defaultdict(list)
    for cat, t in results:
        if t is not None:
            timing_by_cat[cat].append(t)

    print("📊 Category-wise average TOTAL TIME:")
    all_times = []
    for cat in sorted(cat_paths.keys()):
        times = timing_by_cat.get(cat)
        if times:
            avg = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            all_times.extend(times)
            print(f"{cat:25s}: {avg:.2f} ± {std:.2f} sec ({len(times)} samples)")
        else:
            print(f"{cat:25s}: ❌ All failed or not sampled")

    if all_times:
        print(f"\n✅ Overall average TOTAL TIME: {statistics.mean(all_times):.2f} sec")
    else:
        print("\n⚠️ No successful samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Samples per class")
    args = parser.parse_args()
    benchmark(sample_per_class=args.n)

