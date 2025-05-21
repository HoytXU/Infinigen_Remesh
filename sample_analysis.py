# import os
# import random
# import subprocess
# import re
# from collections import Counter
# from multiprocessing import Pool, cpu_count
#
# BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
# VOXEL_SIZE = 0.005
# CACHE_FILE = "relative_file_list.txt"
# NUM_SAMPLES = 100
# MAX_PROCESSES = 4 
#
# def sample_paths():
#     with open(CACHE_FILE, "r") as f:
#         all_lines = [line.strip() for line in f if line.strip()]
#
#     sampled = random.sample(all_lines, NUM_SAMPLES)
#     categories = [line.split("/")[0] for line in sampled]
#     stats = Counter(categories)
#
#     print("📊 Category distribution (in 100 samples):")
#     for k, v in stats.items():
#         print(f"{k:25s}: {v}")
#
#     return sampled
#
# def run_and_extract_time(rel_path):
#     input_path = os.path.join(BASE_DIR, rel_path)
#     output_path = input_path.replace("meshes", "remeshes")
#
#     cmd = [
#         "blender", "--background",
#         "--python", "remesh_worker.py",
#         "--", input_path, output_path, str(VOXEL_SIZE)
#     ]
#     try:
#         result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout = result.stdout.decode(errors="ignore")
#         match = re.search(r"🧠 TOTAL TIME\s+:\s+([\d\.]+)", stdout)
#         if match:
#             return float(match.group(1))
#     except subprocess.CalledProcessError:
#         print(f"❌ [FAIL] {input_path}")
#     return None
#
# if __name__ == "__main__":
#     print("🎯 Sampling and analyzing 100 tasks...\n")
#     sampled_list = sample_paths()
#
#     print(f"\n🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")
#     with Pool(processes=MAX_PROCESSES) as pool:
#         durations = pool.map(run_and_extract_time, sampled_list)
#
#     valid_durations = [d for d in durations if d is not None]
#     if valid_durations:
#         avg_time = sum(valid_durations) / len(valid_durations)
#         print(f"\n✅ Average TOTAL TIME over {len(valid_durations)} successful runs: {avg_time:.2f} sec")
#     else:
#         print("\n⚠️ No successful runs found.")

import os
import random
import subprocess
import re
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count

BASE_DIR = "/cpfs05/shared/landmark_3dgen/lvzhaoyang_group/shape2code/datasets/part2code/meshes"
VOXEL_SIZE = 0.005
CACHE_FILE = "relative_file_list.txt"
NUM_SAMPLES = 100
MAX_PROCESSES = min(8, cpu_count())

def get_all_categories():
    with open(CACHE_FILE, "r") as f:
        all_lines = [line.strip() for line in f if line.strip()]
    all_categories = sorted(set(line.split("/")[0] for line in all_lines))
    return all_lines, all_categories

def sample_paths(all_lines):
    sampled = random.sample(all_lines, NUM_SAMPLES)
    categories = [line.split("/")[0] for line in sampled]
    stats = Counter(categories)

    print("📊 Category distribution (in 100 samples):")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")
    
    return sampled

def run_and_extract_time(rel_path):
    category = rel_path.split("/")[0]
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
            return category, float(match.group(1))
    except subprocess.CalledProcessError:
        print(f"❌ [FAIL] {input_path}")
    return category, None

if __name__ == "__main__":
    print("🎯 Sampling and analyzing 100 tasks...\n")
    all_lines, all_categories = get_all_categories()
    sampled_list = sample_paths(all_lines)

    print(f"\n🧵 Launching multiprocessing pool (workers = {MAX_PROCESSES})...\n")
    with Pool(processes=MAX_PROCESSES) as pool:
        results = pool.map(run_and_extract_time, sampled_list)

    # 分类统计
    category_times = defaultdict(list)
    for cat, duration in results:
        if duration is not None:
            category_times[cat].append(duration)

    print("\n⏱️ Average TOTAL TIME per category:")
    total_times = []
    for cat in sorted(all_categories):
        if cat in category_times:
            times = category_times[cat]
            avg = sum(times) / len(times)
            total_times.extend(times)
            print(f"{cat:25s}: {avg:.2f} sec ({len(times)} samples)")
        else:
            print(f"{cat:25s}: ❌ Not sampled")

    if total_times:
        print(f"\n✅ Overall average TOTAL TIME across {len(total_times)} successful samples: {sum(total_times)/len(total_times):.2f} sec")
    else:
        print("\n⚠️ No successful samples.")

