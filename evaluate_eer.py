import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
from collections import defaultdict

from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.gmu_fusion import GMUFusion
from models.output_head import OutputHead

import importlib.util
spec = importlib.util.spec_from_file_location("biovid_dataset", "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/datasets/biovid_dataset.py")
biovid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(biovid)

BiovidDataset = biovid.BiovidDataset
create_3fold_user_split = biovid.create_3fold_user_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_fused_embedding(model_v, model_a, fuser, sample):
    visual = model_v(sample['frames'].unsqueeze(0).to(device))
    audio = model_a(sample['audio_path'])[None, :].to(device)
    fused = fuser(visual, audio.squeeze(1))
    return fused.cpu().squeeze().numpy()

def main():
    data_dir = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/data/processed/train"
    model_path = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/results/gmu_fusion/fold0_best_model.pt"

    folds, user_to_label = create_3fold_user_split(data_dir)
    train_videos, val_videos = folds[0]

    v_model = VisualEncoder().to(device)
    a_model = AudioEncoder().to(device)
    fuser = GMUFusion().to(device)
    head = OutputHead().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    v_model.load_state_dict(checkpoint['visual'])
    a_model.load_state_dict(checkpoint['audio'])
    fuser.load_state_dict(checkpoint['fusion'])
    head.load_state_dict(checkpoint['head'])

    v_model.eval(); a_model.eval(); fuser.eval(); head.eval()

    train_set = BiovidDataset(train_videos, user_to_label)
    val_set = BiovidDataset(val_videos, user_to_label)

    print(f"🔎 Extracting enrollment embeddings for {len(train_set)} videos...")
    user_embeddings = defaultdict(list)
    for sample in tqdm(train_set):
        uid = sample['user_id']
        emb = extract_fused_embedding(v_model, a_model, fuser, sample)
        user_embeddings[uid].append(emb)

    for uid in user_embeddings:
        user_embeddings[uid] = np.mean(user_embeddings[uid], axis=0)

    print(f"🎯 Evaluating on {len(val_set)} validation videos...")
    y_true = []
    y_score = []
    same_sim = []
    diff_sim = []

    val_uids = set()
    for sample in tqdm(val_set):
        uid = sample['user_id']
        val_uids.add(uid)
        label = sample['label'].item()
        probe_emb = extract_fused_embedding(v_model, a_model, fuser, sample)

        if uid in user_embeddings:
            enrolled_emb = user_embeddings[uid]
            score = 1 - cosine(probe_emb, enrolled_emb)

            y_score.append(score)
            y_true.append(label)

            if label == 1:
                same_sim.append(score)
            else:
                diff_sim.append(score)

    # Show user overlap & match stats
    train_uids = set(user_embeddings.keys())
    overlap = val_uids & train_uids
    print(f"\n👥 Users in both train and validation: {len(overlap)} / {len(val_uids)}")
    print(f"🎯 Matched validation samples: {len(y_score)} / {len(val_set)}")

    # Cosine similarity stats
    print("\n📊 Cosine Similarity Stats")
    print(f"Avg same-user similarity:   {np.mean(same_sim) if same_sim else 'N/A'}")
    print(f"Avg different-user similarity: {np.mean(diff_sim) if diff_sim else 'N/A'}")

    # EER and AUC
    if len(set(y_true)) < 2:
        print("\n❌ Not enough genuine/impostor samples for EER calculation.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean((fpr[eer_idx], fnr[eer_idx]))
    print(f"\n✅ EER: {eer:.4f}")
    print(f"✅ AUC: {auc(fpr, tpr):.4f}")

if __name__ == "__main__":
    main()
