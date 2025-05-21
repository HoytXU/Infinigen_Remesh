import os
import random
import subprocess
import re
import argparse
from collections import defaultdict
from multiprocessing import Pool
import statistics
from tqdm import tqdm

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
    category_sizes = {k: len(v) for k, v in category_paths.items()}
    return category_paths, category_sizes

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
    cat_paths, cat_sizes = load_all_paths()

    print("📦 Total .obj file count per category:")
    for cat in sorted(cat_sizes.keys()):
        print(f"{cat:25s}: {cat_sizes[cat]} files")

    total_tasks = []
    for cat, paths in cat_paths.items():
        n = min(sample_per_class, len(paths))
        sampled = random.sample(paths, n)
        total_tasks.extend(sampled)

    print(f"\n🧪 Benchmarking {len(total_tasks)} tasks from {len(cat_paths)} categories...\n")

    with Pool(processes=MAX_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_and_extract_time, total_tasks),
            total=len(total_tasks),
            desc="⏱ Processing"
        ))

    timing_by_cat = defaultdict(list)
    for cat, t in results:
        if t is not None:
            timing_by_cat[cat].append(t)

    print("\n📊 Category-wise average TOTAL TIME:")
    weighted_sum = 0.0
    total_weight = 0

    for cat in sorted(cat_paths.keys()):
        total_count = cat_sizes[cat]
        times = timing_by_cat.get(cat)
        if times:
            avg = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0

            weighted_sum += avg * total_count
            total_weight += total_count

            print(f"{cat:25s}: {avg:.2f} ± {std:.2f} sec ({len(times)} samples, total {total_count})")
        else:
            print(f"{cat:25s}: ❌ All failed or not sampled (total {total_count})")

    if total_weight > 0:
        print(f"\n📦 Weighted overall TOTAL TIME (by class size): {weighted_sum / total_weight:.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Samples per class")
    args = parser.parse_args()
    benchmark(sample_per_class=args.n)

