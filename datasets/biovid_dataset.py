import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import KFold


class BiovidDataset(Dataset):
    def __init__(self, video_dirs, user_to_label):
        """
        Args:
            video_dirs (List[str]): list of paths to video folders (each has frames.npy and audio.wav)
            user_to_label (Dict[str, int]): maps user IDs like 'U001' to integer labels
        """
        self.video_dirs = video_dirs
        self.user_to_label = user_to_label

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vid_path = self.video_dirs[idx]
        vid_name = os.path.basename(vid_path)
        user_id = vid_name.split("_")[0]
        is_true = vid_name.split("_")[2]  # 'T' or 'F'

        frames_path = os.path.join(vid_path, "frames.npy")
        audio_path = os.path.join(vid_path, "audio.wav")

        frames = np.load(frames_path)  # shape: [3, 30, 96, 96]
        frames_tensor = torch.tensor(frames).float()

        label = 1 if is_true == "T" else 0
        user_label = self.user_to_label[user_id]

        return {
            'frames': frames_tensor,                  # for visual encoder
            'audio_path': audio_path,                 # for audio encoder
            'label': torch.tensor(label),             # 1 = genuine, 0 = impostor
            'user_id': user_id,                       # e.g., 'U001'
            'user_label': torch.tensor(user_label)    # int for triplet loss
        }


def create_3fold_user_split(root_dir):
    """
    Create user-level 3-fold split to avoid leakage.

    Args:
        root_dir (str): path to 'data/processed/train' folder

    Returns:
        folds: List of (train_paths, val_paths)
        user_to_label: Dict[str, int]
    """
    root = Path(root_dir)
    all_dirs = list(root.glob("*_*_*"))  # each video folder
    all_users = sorted(set([p.name.split("_")[0] for p in all_dirs]))

    user_to_label = {u: i for i, u in enumerate(all_users)}
    folds = []

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(all_users):
        train_users = set([all_users[i] for i in train_idx])
        val_users = set([all_users[i] for i in val_idx])

        train_videos = [str(p) for p in all_dirs if p.name.split("_")[0] in train_users]
        val_videos = [str(p) for p in all_dirs if p.name.split("_")[0] in val_users]

        folds.append((train_videos, val_videos))

    return folds, user_to_label
