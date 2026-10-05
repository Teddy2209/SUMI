#!/usr/bin/env python3
"""
Train Diffusion Policy (Absolute Cartesian SLAM) với LeRobot.
Kết hợp từ s3_train_diffusion.py và s3_train_diffusion_640x480.py

Tính năng:
  - Hỗ trợ truyền tham số cho backbone, batch_size.
  - Hỗ trợ Auto-Resume (Train tiếp từ checkpoint cũ).
  - Tham số không khai báo ở đây được truyền thẳng cho LeRobot (VD: --num_workers=8).
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# ======================================================================
# ARGPARSE & CONFIG
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy with LeRobot (Absolute)")
    
    # Bắt buộc
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Tên thư mục dataset nằm trong S2_datasets_lerobot (VD: lerobot_dataset_fpccam_slam_10fps)")
    
    # Tùy chọn cấu hình mô hình & train
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],
                        help="Vision backbone architecture (default: resnet18)")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size (default: 128 cho 320x240, 16 cho 640x480)")
    parser.add_argument("--steps", type=int, default=5000000, 
                        help="Số bước huấn luyện (default: 5M)")
    parser.add_argument("--gpu_id", type=str, default="0", 
                        help="CUDA_VISIBLE_DEVICES ID (default: 0)")
    
    # Auto-resume
    parser.add_argument("--resume_dir", type=str, default=None, 
                        help="Đường dẫn đến thư mục Output cũ nếu muốn train tiếp")
    
    return parser.parse_known_args()

# ======================================================================
# MAIN ROUTING
# ======================================================================
def main():
    args, extra_args = parse_args()
    
    # Thiết lập GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Các đường dẫn cơ bản
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    S2_DATASETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S2_Data_Processing_Standaization", "S2_datasets_lerobot"))
    DATASET_DIR = os.path.join(S2_DATASETS_DIR, args.dataset_name)
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Lỗi: Không tìm thấy dataset tại {DATASET_DIR}.")
        sys.exit(1)

    # ─── Xử lý đường dẫn Output & Resume ───
    if args.resume_dir:
        OUTPUT_DIR = os.path.abspath(args.resume_dir)
        resume_flag = "true"
        print(f"[MODE] Resume: Sẽ train tiếp từ {OUTPUT_DIR}")
    else:
        out_date_folder = datetime.now().strftime("Date_%d%m%Y")
        dataset_basename = os.path.basename(args.dataset_name)
        OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S3_output", out_date_folder, f"diffusion_checkpoints_{dataset_basename}"))
        resume_flag = "false"
        print(f"[MODE] Train Mới: Sẽ lưu vào {OUTPUT_DIR}")

    # ─── Xây dựng Command line cho LeRobot ───
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=diffusion",

        # Dataset
        "--dataset.repo_id=apicoo/robot_pick_place",
        f"--dataset.root={DATASET_DIR}",

        # Training loop
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--eval_freq=-1",
        "--save_freq=100000",
        "--save_checkpoint=true",
        "--log_freq=1000",
        "--seed=42",
        "--num_workers=16",

        # Optimizer
        "--optimizer.lr=1e-4",
        "--optimizer.weight_decay=1e-6",
        "--optimizer.grad_clip_norm=1.0",
        
        # Policy & Device
        "--policy.device=cuda",
        f"--policy.vision_backbone={args.backbone}",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.num_train_timesteps=100",

        # Output
        f"--output_dir={OUTPUT_DIR}",
        f"--resume={resume_flag}",

        # Tắt upload / telemetry
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    
    # Nạp config cũ nếu Resume
    if resume_flag == "true":
        cmd.append(f"--config_path={OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json")

    cmd.extend(extra_args)

    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN DIFFUSION POLICY (ABSOLUTE)")
    print("=" * 70)
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Command:\n  {' '.join(cmd)}")
    print("=" * 70 + "\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Đã huỷ quá trình huấn luyện bằng tay (Ctrl+C).")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Lỗi xảy ra trong quá trình huấn luyện: {e}")

if __name__ == "__main__":
    main()