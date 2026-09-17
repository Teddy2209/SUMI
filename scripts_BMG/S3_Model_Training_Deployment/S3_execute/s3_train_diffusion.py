import os
import sys
import subprocess

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy with LeRobot")
    parser.add_argument("--dataset_name", type=str, required=True, help="Tên thư mục dataset nằm trong S2_datasets_lerobot (VD: lerobot_dataset_fpccam_slam_10fps)")
    args = parser.parse_args()

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    S2_DATASETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "S2_Data_Processing_Standaization", "S2_datasets_lerobot"))
    DATASET_DIR = os.path.join(S2_DATASETS_DIR, args.dataset_name)
    OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "output_trained", f"diffusion_checkpoints_{args.dataset_name}"))

    if not os.path.exists(DATASET_DIR):
        print(f"Lỗi: Không tìm thấy dataset tại {DATASET_DIR}.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=diffusion",

        # Dataset
        "--dataset.repo_id=apicoo/robot_pick_place",
        f"--dataset.root={DATASET_DIR}",

        # Training loop
        "--steps=100000",
        "--batch_size=32",
        "--eval_freq=-1",
        "--save_freq=20000",
        "--save_checkpoint=true",
        "--log_freq=10",
        "--seed=42",
        "--num_workers=16",

        # Optimizer (LeRobot 0.5+ chuẩn)
        "--optimizer.lr=1e-4",
        "--optimizer.weight_decay=1e-6",
        "--optimizer.grad_clip_norm=1.0",
        # Device (Vì đã dùng CUDA_VISIBLE_DEVICES ở ngoài, nên trong này mặc định là cuda:0)
        "--policy.device=cuda",

        # Diffusion Policy config
        "--policy.vision_backbone=resnet34",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.num_train_timesteps=100",

        # Output
        f"--output_dir={OUTPUT_DIR}",

        # Tắt tích hợp bên ngoài
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]

    print("--------------------------------------------------")
    print("Bắt đầu huấn luyện mô hình Diffusion Policy (LeRobot)")
    print("--------------------------------------------------")
    print("Thư mục lưu Checkpoint:", OUTPUT_DIR)
    print("Lệnh chạy:\n", " ".join(cmd))
    print("--------------------------------------------------\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nĐã huỷ quá trình huấn luyện bằng tay (Ctrl+C).")
    except subprocess.CalledProcessError as e:
        print(f"\nLỗi xảy ra trong quá trình huấn luyện: {e}")

if __name__ == "__main__":
    main()