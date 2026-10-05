"""
UMI-style relative-to-current-timestep dataset wrapper for LeRobot.

Converts absolute Tool₀-frame poses to relative-to-current at __getitem__ time,
matching UMI's training approach: inv(T_t) @ T_i for all frames in the window.

Usage:
    dataset = LeRobotDataset(...)
    wrapped = UMIRelativeDataset(dataset, n_obs_steps=2)
    wrapped.recompute_stats()
"""

import torch
import numpy as np
from torch.utils.data import Dataset


def rot6d_to_matrix(rot6d):
    v1 = rot6d[:3].clone()
    v2 = rot6d[3:6].clone()
    v1 = v1 / (v1.norm() + 1e-8)
    v3 = torch.cross(v1, v2, dim=0)
    v3 = v3 / (v3.norm() + 1e-8)
    v2 = torch.cross(v3, v1, dim=0)
    return torch.stack([v1, v2, v3], dim=1)


def pose10d_to_matrix(state):
    T = torch.eye(4, dtype=state.dtype)
    T[:3, 3] = state[:3]
    T[:3, :3] = rot6d_to_matrix(state[3:9])
    return T


def matrix_to_pose9d(T):
    pose = torch.zeros(9, dtype=T.dtype)
    pose[:3] = T[:3, 3]
    pose[3:6] = T[:3, 0]
    pose[6:9] = T[:3, 1]
    return pose


def convert_poses_to_relative(poses, anchor_idx):
    """
    Convert (N, 10) absolute poses to relative w.r.t. anchor frame.
    Gripper (index 9) stays unchanged.
    """
    T_anchor = pose10d_to_matrix(poses[anchor_idx])
    T_anchor_inv = torch.linalg.inv(T_anchor)

    result = poses.clone()
    for i in range(poses.shape[0]):
        T_i = pose10d_to_matrix(poses[i])
        T_rel = T_anchor_inv @ T_i
        result[i, :9] = matrix_to_pose9d(T_rel)
    return result


class UMIRelativeDataset(Dataset):
    """
    Wraps LeRobotDataset to apply inv(T_current) @ T_i at __getitem__ time.

    After conversion:
    - observation.state[current] = [0,0,0, 1,0,0, 0,1,0, gripper] (identity)
    - All other frames = displacement relative to current pose
    - Gripper values preserved unchanged
    """

    def __init__(self, dataset, n_obs_steps=2):
        self.dataset = dataset
        self.n_obs_steps = n_obs_steps
        self.current_idx = n_obs_steps - 1

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        try:
            dataset = object.__getattribute__(self, 'dataset')
        except AttributeError:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        obs_state = item.get("observation.state")
        action = item.get("action")

        if obs_state is None:
            return item

        if obs_state.dim() == 2:
            current_pose = obs_state[self.current_idx]
        else:
            current_pose = obs_state

        T_current = pose10d_to_matrix(current_pose)
        T_current_inv = torch.linalg.inv(T_current)

        if obs_state.dim() == 2:
            new_obs = obs_state.clone()
            for i in range(obs_state.shape[0]):
                T_i = pose10d_to_matrix(obs_state[i])
                T_rel = T_current_inv @ T_i
                new_obs[i, :9] = matrix_to_pose9d(T_rel)
            item["observation.state"] = new_obs

        if action is not None and action.dim() == 2:
            new_action = action.clone()
            for i in range(action.shape[0]):
                T_i = pose10d_to_matrix(action[i])
                T_rel = T_current_inv @ T_i
                new_action[i, :9] = matrix_to_pose9d(T_rel)
            item["action"] = new_action

        return item

    def recompute_stats(self, num_samples=5000):
        """
        Recompute normalization stats for the relative representation.
        Accesses raw HF dataset to avoid loading images.
        """
        print(f"[UMI Wrapper] Recomputing normalization stats ({num_samples} samples)...")

        self.dataset._ensure_hf_dataset_loaded()
        hf = self.dataset.hf_dataset

        all_states = torch.tensor(
            np.array(hf["observation.state"]), dtype=torch.float32
        )
        all_eps = torch.tensor(np.array(hf["episode_index"]))

        N = len(all_states)
        sample_size = min(num_samples, N)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(N, sample_size, replace=False)

        if self.dataset.delta_indices and "action" in self.dataset.delta_indices:
            action_deltas = self.dataset.delta_indices["action"]
        else:
            action_deltas = list(range(-1, 15))

        obs_deltas = list(range(1 - self.n_obs_steps, 1))
        all_deltas = sorted(set(obs_deltas + action_deltas))

        relative_poses = []

        for t in sample_indices:
            ep = all_eps[t].item()
            T_current = pose10d_to_matrix(all_states[t])
            T_current_inv = torch.linalg.inv(T_current)

            for delta in all_deltas:
                i = int(t) + delta
                if i < 0 or i >= N:
                    continue
                if all_eps[i].item() != ep:
                    continue
                T_i = pose10d_to_matrix(all_states[i])
                T_rel = T_current_inv @ T_i
                rel_9d = matrix_to_pose9d(T_rel)
                gripper = all_states[i, 9]
                rel_10d = torch.cat([rel_9d, gripper.unsqueeze(0)])
                relative_poses.append(rel_10d)

        if not relative_poses:
            print("[UMI Wrapper] WARNING: No valid relative poses computed!")
            return

        rel_tensor = torch.stack(relative_poses)

        new_stats = {
            "min": rel_tensor.min(dim=0).values.numpy(),
            "max": rel_tensor.max(dim=0).values.numpy(),
            "mean": rel_tensor.mean(dim=0).numpy(),
            "std": rel_tensor.std(dim=0).numpy(),
        }

        eps = 1e-6
        range_val = new_stats["max"] - new_stats["min"]
        zero_range = range_val < eps
        new_stats["max"][zero_range] += eps
        new_stats["min"][zero_range] -= eps
        new_stats["std"][new_stats["std"] < eps] = eps

        self.dataset.meta.stats["observation.state"] = new_stats
        self.dataset.meta.stats["action"] = {k: v.copy() for k, v in new_stats.items()}

        print(f"[UMI Wrapper] Done. {len(relative_poses)} relative poses computed.")
        print(f"  Pos min: {np.round(new_stats['min'][:3], 5)}")
        print(f"  Pos max: {np.round(new_stats['max'][:3], 5)}")
        print(f"  Pos std: {np.round(new_stats['std'][:3], 5)}")
