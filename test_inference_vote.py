# test_inference_vote.py

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.fusion_head import GMUFusion
from scipy.spatial.distance import cosine

# === Config ===
TEST_DIR = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/data/processed/test"
TRAIN_DIR = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/data/processed/train"
MODEL_DIR = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/results"
SUBMIT_PATH = "/content/drive/MyDrive/biovid_dual_auth/updated_pipeline/submission/submission.json"
FOLDS = [0, 1, 2]
THRESHOLD = 0.60 # set based on val EER

# === Load known user embeddings ===
def load_user_embeddings(visual_encoder, audio_encoder, fusion_model, device):
    visual_encoder.eval()
    fusion_model.eval()
    user_embs = {}

    for user_folder in Path(TRAIN_DIR).iterdir():
        audio_file = user_folder / "audio.wav"
        frame_file = user_folder / "frames.npy"
        if not audio_file.exists() or not frame_file.exists():
            print(f"⚠️ Skipping {user_folder.name} (missing audio or frames)")
            continue

        user_id = user_folder.name.split('_')[0]
        if user_id not in user_embs:
            user_embs[user_id] = []

        try:
            frames = np.load(frame_file)
            frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
            v_emb = visual_encoder(frames_tensor)
            a_emb = audio_encoder(str(audio_file)).to(device)
            _, joint_emb = fusion_model(a_emb, v_emb)
            user_embs[user_id].append(joint_emb.squeeze(0).detach().cpu().numpy())
        except Exception as e:
            print(f"❌ Error loading {user_folder.name}: {e}")
            continue

    for uid in user_embs:
        user_embs[uid] = np.mean(user_embs[uid], axis=0)
    return user_embs

# === Predict test set ===
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    predictions = {}
    test_videos = list(Path(TEST_DIR).glob("*/"))
    all_fold_preds = {vid.name: [] for vid in test_videos}

    for fold in FOLDS:
        v_model = VisualEncoder().to(device)
        a_model = AudioEncoder(device=device)
        f_model = GMUFusion().to(device)
        ckpt = torch.load(f"{MODEL_DIR}/fold{fold}_best_model.pt")
        v_model.load_state_dict(ckpt['visual'])
        f_model.load_state_dict(ckpt['fusion'])

        known_embs = load_user_embeddings(v_model, a_model, f_model, device)

        for vid_path in tqdm(test_videos, desc=f"FOLD {fold} Test"):
            audio_file = vid_path / "audio.wav"
            frame_file = vid_path / "frames.npy"

            if not audio_file.exists() or not frame_file.exists():
                print(f"⚠️ Skipping {vid_path.name} (missing audio or frames)")
                continue

            try:
                frames = np.load(frame_file)
                frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
                v_emb = v_model(frames_tensor)
                a_emb = a_model(str(audio_file)).to(device)
                _, joint_emb = f_model(a_emb, v_emb)
                test_emb = joint_emb.squeeze(0).detach().cpu().numpy()

                best_uid, best_score = "unknown", 0.0
                for uid, emb in known_embs.items():
                    sim = 1 - cosine(test_emb, emb)
                    if sim > best_score:
                        best_score = sim
                        best_uid = uid

                pred_uid = best_score >= THRESHOLD and best_uid or "unknown"
                all_fold_preds[vid_path.name].append((pred_uid, best_score))
            except Exception as e:
                print(f"❌ Error with {vid_path.name}: {e}")
                continue

    final_results = []
    for vid_name, preds in all_fold_preds.items():
        if len(preds) == 0:
            continue
        user_ids = [p[0] for p in preds]
        scores = [p[1] for p in preds]
        final_uid = max(set(user_ids), key=user_ids.count)
        avg_score = float(np.mean(scores))

        final_results.append({
            "video_file": f"{vid_name}.mp4",
            "score": round(avg_score, 4),
            "user_id": final_uid,
            "spoken_word": "unknown"
        })

    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    with open(SUBMIT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"✅ Submission saved to: {SUBMIT_PATH}")

if __name__ == "__main__":
    predict()
