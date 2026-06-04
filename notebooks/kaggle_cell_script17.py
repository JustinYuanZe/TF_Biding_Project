"""
KAGGLE CELL SCRIPT 17
Copy toàn bộ code dưới đây vào 1 cell Kaggle rồi chạy.
Script sẽ tự động pull repo mới nhất và chạy Script 17.
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

subprocess.run(["python3", "notebooks/17_unified_crossmodal_v2_kaggle.py"], check=True)
