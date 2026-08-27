#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    # ======================================================================
    # Path configuration
    # ======================================================================
    repo_id = "apicoo/robot_pick_place"
    dataset_dir = "/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/ORB_SLAM3/lerobot_dataset_act_slam"
    output_dir = Path("/media/apicoo-ai/5511010c-3660-41c3-b501-36e739767b6a/ORB_SLAM3/output_trained/act_checkpoints_act_slam_v2")
    
    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()

    print("======================================================================")
    print("Launching ACT Training with GPU (Based on your Custom Script)")
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print("======================================================================")

    # ======================================================================
    # Environment Variables
    # ======================================================================
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["MASTER_PORT"] = "29505"
    env["HF_HUB_OFFLINE"] = "1"

    # ======================================================================
    # Base Command (Sử dụng argparse style `--key=value`)
    # ======================================================================
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        
        # Dataset Config
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_dir}",
        
        # Policy Config (Sử dụng ACT)
        "--policy.type=act",
        "--policy.device=cuda",
        
        # Output Config
        f"--output_dir={output_dir}",
        
        # Hyperparameters từ script cũ
        "--batch_size=32",
        "--num_workers=16",
        
        # Bật Temporal Ensembling (Nội suy quỹ đạo trung bình) giống hệt bài báo ACT gốc
        "--policy.chunk_size=100",
        "--policy.n_action_steps=1",
        "--policy.temporal_ensemble_coeff=0.01",
        "--policy.vision_backbone=resnet34",
        "--policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1",
        "--policy.n_decoder_layers=7",
        
        # Enable VAE for Multi-Position Distribution Generalization
        "--policy.use_vae=true",
        
        # Mitigate Overfitting Gradient Explosion
        "--steps=150000",
        "--save_freq=30000",
        "--log_freq=500",
        "--save_checkpoint=true",
        "--eval_freq=-1",
        
        # Disable wandb & push to hub
        "--wandb.enable=false",
        "--policy.push_to_hub=false"
    ]
    
    # ======================================================================
    # Auto-Resume Logic
    # ======================================================================
    last_checkpoint_cfg = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if last_checkpoint_cfg.exists():
        print(f"\n[INFO] Found existing checkpoint config at {last_checkpoint_cfg}. Resuming training!")
        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
            f"--config_path={last_checkpoint_cfg}",
            "--resume=true"
        ]
    elif output_dir.exists():
        # Fallback to the latest step checkpoint if "last" doesn't exist
        step_checkpoints = list(output_dir.glob("checkpoints/*/pretrained_model/train_config.json"))
        if step_checkpoints:
            latest_cfg = sorted(step_checkpoints)[-1]
            print(f"\n[INFO] Found step checkpoint config at {latest_cfg}. Resuming training!")
            cmd = [
                sys.executable, "-m", "lerobot.scripts.lerobot_train",
                f"--config_path={latest_cfg}",
                "--resume=true"
            ]

    # Display the command to run
    cmd_str = " \\\n  ".join(cmd)
    print("Command to execute:")
    print(f"QT_QPA_PLATFORM=offscreen PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '0')} \\\n{cmd_str}\n")
    
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
