# train_crossval.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.fusion_head import GMUFusion
from utils.dataset import BiovidDataset
from utils.metrics import calculate_metrics
from utils.losses import TripletLossWithMining

def train_one_fold(fold, train_loader, val_loader, device, results_dir):
    visual_encoder = VisualEncoder().to(device)
    audio_encoder = AudioEncoder(device=device)
    fusion_model = GMUFusion().to(device)

    criterion_cls = nn.BCELoss()
    criterion_triplet = TripletLossWithMining(margin=0.3)

    optimizer = optim.Adam(list(visual_encoder.parameters()) +
                           list(fusion_model.parameters()), lr=1e-4)

    best_eer = float('inf')
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f"fold{fold}_best_model.pt")

    for epoch in range(30):
        visual_encoder.train()
        fusion_model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}"):
            frames, audio_paths, labels, triplet_group = batch
            frames = frames.to(device)
            labels = labels.float().to(device)

            v_emb = visual_encoder(frames)
            a_emb = torch.stack([audio_encoder(p).squeeze(0) for p in audio_paths]).to(device)
            scores, emb = fusion_model(a_emb, v_emb)

            loss_cls = criterion_cls(scores.squeeze(), labels)
            loss_triplet = criterion_triplet(emb, triplet_group)
            loss = loss_cls + loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, eer, apcer, bpcer = evaluate_model(
            visual_encoder, audio_encoder, fusion_model, val_loader, device)

        print(f"Fold {fold} - Epoch {epoch+1}: Val Acc={acc:.4f}, EER={eer:.4f}")
        if eer < best_eer:
            best_eer = eer
            torch.save({
                'visual': visual_encoder.state_dict(),
                'fusion': fusion_model.state_dict()
            }, best_model_path)

    return acc, eer, apcer, bpcer

@torch.no_grad()
def evaluate_model(visual_encoder, audio_encoder, fusion_model, val_loader, device):
    visual_encoder.eval()
    fusion_model.eval()

    y_true, y_pred, embeddings = [], [], []

    for batch in val_loader:
        frames, audio_paths, labels, triplet_group = batch
        frames = frames.to(device)
        labels = labels.float().cpu().numpy()
        v_emb = visual_encoder(frames)
        a_emb = torch.stack([audio_encoder(p).squeeze(0) for p in audio_paths]).to(device)
        scores, emb = fusion_model(a_emb, v_emb)

        y_true.extend(labels)
        y_pred.extend(scores.squeeze().cpu().numpy())
        embeddings.append(emb.cpu())

    return calculate_metrics(np.array(y_true), np.array(y_pred))

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    DATA_ROOT = Path("/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/data/processed/train")
    dataset = BiovidDataset(root=DATA_ROOT)
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

        acc, eer, apcer, bpcer = train_one_fold(
            fold, train_loader, val_loader, device, results_dir="/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/results")

        results.append({
            "fold": fold,
            "accuracy": acc,
            "eer": eer,
            "apcer": apcer,
            "bpcer": bpcer
        })

    df = pd.DataFrame(results)
    avg = df[["accuracy", "eer", "apcer", "bpcer"]].mean().to_dict()
    df.loc[len(df.index)] = ["avg"] + list(avg.values())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/results", exist_ok=True)
    df.to_csv(f"/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/results/biovid_results_{timestamp}.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
