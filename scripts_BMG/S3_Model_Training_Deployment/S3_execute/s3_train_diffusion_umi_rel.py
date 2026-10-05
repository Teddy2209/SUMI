#!/usr/bin/env python3
"""
Train Diffusion Policy với tọa độ UMI Relative.

Cơ chế:
  - Khác với Absolute, mô hình này học cách predict toạ độ đích RELATIVE so với tọa độ tay hiện tại.
  - Dùng dataset TUYỆT ĐỐI (s2.4 không _rel); `s3_umi_relative_wrapper.py` đổi sang inv(T_hiện_tại) @ T_i
    ở runtime bằng cách patch `make_dataset` của LeRobot (obs, action, và normalization stats).
  - Inference tương ứng: s3.1_run_diffusion_320x240_EMA_rel.py.
  - Tham số không khai báo ở đây được truyền thẳng cho LeRobot (VD: --num_workers=8).
"""

import os
import sys
import argparse
from datetime import datetime

# ======================================================================
# ARGPARSE & CONFIG
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy (UMI Relative)")
    
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset folder name inside S2_datasets_lerobot")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],
                        help="Vision backbone architecture (default: resnet18)")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size (default: 128)")
    parser.add_argument("--steps", type=int, default=5000000, 
                        help="Số bước huấn luyện (default: 5M)")
    parser.add_argument("--gpu_id", type=str, default="1", 
                        help="CUDA_VISIBLE_DEVICES ID (default: 1)")
    
    # Auto-resume
    parser.add_argument("--resume_dir", type=str, default=None, 
                        help="Đường dẫn đến thư mục Output cũ nếu muốn train tiếp")
    
    return parser.parse_known_args()

# ======================================================================
# MAIN ROUTING
# ======================================================================
def main():
    args, remaining_args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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
    else:
        out_date_folder = datetime.now().strftime("Date_%d%m%Y")
        dataset_basename = os.path.basename(args.dataset_name)
        OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "S3_output", out_date_folder, f"diffusion_umi_rel_{dataset_basename}"))
        resume_flag = "false"

    # ─── Monkey-patch make_dataset (Đặc trưng của UMI Relative) ───
    sys.path.insert(0, BASE_DIR)
    from s3_umi_relative_wrapper import UMIRelativeDataset
    import lerobot.datasets.factory as factory

    _orig_make_dataset = factory.make_dataset

    def make_dataset_umi_relative(cfg):
        dataset = _orig_make_dataset(cfg)
        n_obs = getattr(cfg.policy, "n_obs_steps", 2)
        wrapped = UMIRelativeDataset(dataset, n_obs_steps=n_obs)
        wrapped.recompute_stats()
        return wrapped

    factory.make_dataset = make_dataset_umi_relative

    # ─── Xây dựng CLI Arguments cho LeRobot ───
    sys.argv = [
        "lerobot_train",
        "--policy.type=diffusion",
        
        # Dataset
        "--dataset.repo_id=apicoo/robot_pick_place",
        f"--dataset.root={DATASET_DIR}",
        
        # Train config
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
        
        # Policy
        "--policy.device=cuda",
        f"--policy.vision_backbone={args.backbone}",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.num_train_timesteps=100",
        
        # IO
        f"--output_dir={OUTPUT_DIR}",
        f"--resume={resume_flag}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    
    if resume_flag == "true":
        sys.argv.append(f"--config_path={OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json")
        
    sys.argv.extend(remaining_args)

    print("\n" + "=" * 70)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN DIFFUSION POLICY (UMI RELATIVE) - {'RESUME' if resume_flag == 'true' else 'NEW'}")
    print("=" * 70)
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    from lerobot.scripts.lerobot_train import main as lerobot_main
    lerobot_main()

if __name__ == "__main__":
    main()
