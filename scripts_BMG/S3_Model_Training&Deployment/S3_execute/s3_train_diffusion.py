import os
import sys
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, "lerobot_dataset_diff_slam")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_trained", "diffusion_checkpoints_diff_slam_rn34")

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Lỗi: Không tìm thấy dataset tại {DATASET_DIR}. Vui lòng chạy s4 trước.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=diffusion",

        # Dataset
        "--dataset.repo_id=apicoo/robot_pick_place",
        f"--dataset.root={DATASET_DIR}",

        # Training loop
        "--steps=150000",
        "--batch_size=16",
        "--eval_freq=-1",
        "--save_freq=10000",
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