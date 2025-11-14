# utils/dataset.py

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class BiovidDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.samples = []

        for user_folder in self.root.iterdir():
            if not user_folder.is_dir():
                continue
            user_id = user_folder.name.split("_")[0]
            label = 1 if "_T" in user_folder.name else 0  # True = genuine, False = impostor
            frames_path = user_folder / "frames.npy"
            audio_path = user_folder / "audio.wav"
            if frames_path.exists() and audio_path.exists():
                self.samples.append((str(frames_path), str(audio_path), label, user_id))

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(set(x[3] for x in self.samples)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, audio_path, label, user_id = self.samples[idx]
        frames = np.load(frames_path)  # [3, 30, 96, 96]
        frames = torch.tensor(frames, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        triplet_group = torch.tensor(self.user_id_to_idx[user_id], dtype=torch.long)
        return frames, audio_path, label, triplet_group
