#!/usr/bin/env python3
"""
Train ACT Policy (Absolute Cartesian SLAM) với LeRobot.

Tính năng:
  - Tự động detect & resume từ checkpoint cũ nếu `--output_dir` đã tồn tại.
  - Cấu trúc lại code rõ ràng, thêm argparse để dễ điều khiển.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ======================================================================
# ARGPARSE & CONFIG
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT Policy with LeRobot (Absolute)")
    
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Tên thư mục dataset nằm trong S2_datasets_lerobot (VD: Date_01102026/lerobot_dataset_...)")
    parser.add_argument("--output_name", type=str, default="act_slam",
                        help="Tên thư mục output trong S3_output (Mặc định: act_slam)")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--steps", type=int, default=150000, help="Số bước huấn luyện")
    parser.add_argument("--gpu_id", type=str, default="1", help="CUDA_VISIBLE_DEVICES ID")
    
    return parser.parse_args()

# ======================================================================
# MAIN ROUTING
# ======================================================================
def main():
    args = parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent
    dataset_dir = (BASE_DIR / ".." / ".." / "S2_Data_Processing_Standaization" / "S2_datasets_lerobot" / args.dataset_name).resolve()
    output_dir = (BASE_DIR / ".." / "S3_output" / args.output_name).resolve()
    repo_id = "apicoo/robot_pick_place"

    if not dataset_dir.exists():
        print(f"❌ Lỗi: Không tìm thấy dataset tại {dataset_dir}.")
        sys.exit(1)

    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()

    # ======================================================================
    # Environment Variables
    # ======================================================================
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["MASTER_PORT"] = "29505"
    env["HF_HUB_OFFLINE"] = "1"

    # ======================================================================
    # Base Command
    # ======================================================================
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        
        # Dataset Config
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_dir}",
        
        # Policy Config (Sử dụng ACT)
        "--policy.type=act",
        "--policy.device=cuda",
        f"--output_dir={output_dir}",
        
        # Hyperparameters
        f"--batch_size={args.batch_size}",
        "--num_workers=16",
        
        # Temporal Ensembling 
        "--policy.chunk_size=100",
        "--policy.n_action_steps=1",
        "--policy.temporal_ensemble_coeff=0.01",
        "--policy.vision_backbone=resnet34",
        "--policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1",
        "--policy.n_decoder_layers=7",
        
        # Multi-Position Generalization
        "--policy.use_vae=true",
        
        # Training Schedule
        f"--steps={args.steps}",
        "--save_freq=30000",
        "--log_freq=500",
        "--save_checkpoint=true",
        "--eval_freq=-1",
        
        # Disable external upload
        "--wandb.enable=false",
        "--policy.push_to_hub=false"
    ]
    
    # ======================================================================
    # Auto-Resume Logic
    # ======================================================================
    resume_mode = False
    last_checkpoint_cfg = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    
    if last_checkpoint_cfg.exists():
        print(f"\n[INFO] Found existing checkpoint config at {last_checkpoint_cfg}. Resuming training!")
        cmd.extend([f"--config_path={last_checkpoint_cfg}", "--resume=true"])
        resume_mode = True
    elif output_dir.exists():
        # Fallback to the latest step checkpoint if "last" doesn't exist
        step_checkpoints = list(output_dir.glob("checkpoints/*/pretrained_model/train_config.json"))
        if step_checkpoints:
            latest_cfg = sorted(step_checkpoints)[-1]
            print(f"\n[INFO] Found step checkpoint config at {latest_cfg}. Resuming training!")
            cmd.extend([f"--config_path={latest_cfg}", "--resume=true"])
            resume_mode = True

    print("\n" + "=" * 70)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN ACT POLICY ({'RESUME' if resume_mode else 'NEW'})")
    print("=" * 70)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Command:\n  {' '.join(cmd)}")
    print("=" * 70 + "\n")
    
    # Run the training process
    try:
        subprocess.run(cmd, env=env, check=True)
        print("\n✅ Huấn luyện thành công! Model đã được lưu tại:", output_dir)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with error code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Training aborted by user.")

if __name__ == "__main__":
    main()
