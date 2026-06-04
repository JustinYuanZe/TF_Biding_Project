"""
KAGGLE CELL SCRIPT 19 — G-CMAB Safe Core
Copy toàn bộ code dưới đây vào 1 cell Kaggle rồi chạy.

CÁCH DÙNG:
  - 1×T4: Chạy trực tiếp cell này (tự động dùng Accelerate single-GPU).
  - 2×T4: Chạy cell này — Accelerate tự phát hiện 2 GPU và phân phối.

Script sẽ tự động pull repo mới nhất và chạy Script 19.
"""
import os, subprocess

REPO_URL = "https://github.com/JustinYuanZe/SP1_TF_Biding_Project.git"
REPO_DIR = "/kaggle/working/SP1_TF_Biding_Project"

os.chdir("/kaggle/working")
print(os.getcwd())

if not os.path.isdir(REPO_DIR):
    print(f"--> Không tìm thấy thư mục. Tiến hành Clone Repo mới từ: {REPO_URL}")
    subprocess.run(["git", "clone", REPO_URL], check=True)
else:
    print("--> Thư mục đã tồn tại. Tiến hành Pull cập nhật code...")
    os.chdir(REPO_DIR)
    print(os.getcwd())
    subprocess.run(["git", "reset", "--hard"], check=True)
    subprocess.run(["git", "pull", "origin", "main"], check=True)

os.chdir("/kaggle/working")
print(os.getcwd())
os.chdir(REPO_DIR)
print(os.getcwd())

# ── Detect GPU count and launch accordingly ──
import torch
n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} GPU(s)")

if n_gpus >= 2:
    # Multi-GPU: use accelerate launch with 2 processes
    print(f"Launching with Accelerate multi-GPU ({n_gpus} processes)...")
    subprocess.run([
        "accelerate", "launch",
        "--multi_gpu",
        f"--num_processes={n_gpus}",
        "--mixed_precision=fp16",
        "notebooks/19_gcmab_safe_kaggle.py",
    ], check=True)
else:
    # Single GPU: accelerate still works, just 1 process
    print("Launching with Accelerate single-GPU...")
    subprocess.run([
        "accelerate", "launch",
        "--num_processes=1",
        "--mixed_precision=fp16",
        "notebooks/19_gcmab_safe_kaggle.py",
    ], check=True)
