# ═══════════════════════════════════════════════════════════════════════
# KAGGLE LAUNCHER FOR SCRIPT 20: Hierarchical KAN Architecture
# ═══════════════════════════════════════════════════════════════════════
import os
import subprocess
import torch

def run_script20_kaggle():
    print("🚀 Initializing Script 20 (Hierarchical KAN Architecture)...")
    
    # 1. Detect GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("❌ CRITICAL: No GPU detected. Please turn on GPU in Kaggle settings.")
        return
        
    print(f"✅ Detected {n_gpus} GPU(s).")
    
    # 2. Check if script exists
    script_path = "20_hierarchical_kan_kaggle.py"
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Cannot find '{script_path}'. Please upload it to your Kaggle environment.")
        return
        
    # 3. Configure Accelerate Launch command
    cmd = [
        "accelerate", "launch",
        "--mixed_precision=no"
    ]
    
    if n_gpus > 1:
        print(f"⚡ Multi-GPU Mode: Configuring Accelerate for {n_gpus} devices...")
        cmd.extend([
            "--multi_gpu",
            f"--num_processes={n_gpus}"
        ])
    else:
        print("⚡ Single-GPU Mode: Configuring Accelerate for 1 device...")
        cmd.extend([
            f"--num_processes=1"
        ])
        
    cmd.append(script_path)
    
    print("\n" + "="*60)
    print("▶️ LAUNCHING TRAINING PIPELINE")
    print("="*60)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Script 20 completed successfully!")
            print("To view diagnostics, run the 'plot_confusion_matrix.py' helper.")
        else:
            print(f"\n❌ Script failed with exit code {process.returncode}.")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == "__main__":
    run_script20_kaggle()
