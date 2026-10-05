"""
Train Diffusion Policy with UMI-style relative-to-current conversion.

Monkey-patches LeRobot's make_dataset to wrap with UMIRelativeDataset,
then runs the standard LeRobot training pipeline.

Usage:
    python s3_train_diffusion_umi_rel.py --dataset_name <name>
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Diffusion Policy (UMI Relative)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset folder name inside S2_datasets_lerobot")
    args, remaining_args = parser.parse_known_args()

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    S2_DATASETS_DIR = os.path.abspath(os.path.join(
        BASE_DIR, "..", "..", "S2_Data_Processing_Standaization", "S2_datasets_lerobot"
    ))
    DATASET_DIR = os.path.join(S2_DATASETS_DIR, args.dataset_name)

    from datetime import datetime
    out_date_folder = datetime.now().strftime("Date_%d%m%Y")
    dataset_basename = os.path.basename(args.dataset_name)
    OUTPUT_DIR = os.path.abspath(os.path.join(
        BASE_DIR, "..", "S3_output", out_date_folder,
        f"diffusion_umi_rel_{dataset_basename}"
    ))

    if not os.path.exists(DATASET_DIR):
        print(f"Lỗi: Không tìm thấy dataset tại {DATASET_DIR}.")
        sys.exit(1)

    # --- Monkey-patch make_dataset ---
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

    # --- Set CLI args for LeRobot's @parser.wrap() ---
    sys.argv = [
        "lerobot_train",
        "--policy.type=diffusion",
        "--dataset.repo_id=apicoo/robot_pick_place",
        f"--dataset.root={DATASET_DIR}",
        "--steps=5000000",
        "--batch_size=128",
        "--eval_freq=-1",
        "--save_freq=100000",
        "--save_checkpoint=true",
        "--log_freq=1000",
        "--seed=42",
        "--num_workers=16",
        "--optimizer.lr=1e-4",
        "--optimizer.weight_decay=1e-6",
        "--optimizer.grad_clip_norm=1.0",
        "--policy.device=cuda",
        "--policy.vision_backbone=resnet18",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.num_train_timesteps=100",
        f"--output_dir={OUTPUT_DIR}",
        "--resume=false",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ] + remaining_args

    print("=" * 60)
    print("DIFFUSION POLICY + UMI RELATIVE-TO-CURRENT")
    print("=" * 60)
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print("=" * 60)

    from lerobot.scripts.lerobot_train import main as lerobot_main
    lerobot_main()


if __name__ == "__main__":
    main()
