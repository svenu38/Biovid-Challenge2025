# scripts/preprocess.py

import os
import cv2
import librosa
import numpy as np
import ffmpeg
from pathlib import Path
from tqdm import tqdm

def extract_audio(video_path, output_wav, sr=16000):
    y, _ = librosa.load(video_path, sr=sr, mono=True)
    librosa.output.write_wav(output_wav, y, sr)

def extract_frames(video_path, output_npy, num_frames=30, size=(96, 96)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if idx in frame_idxs:
            frame = cv2.resize(frame, size)
            frame = frame[..., ::-1]  # BGR to RGB
            frame = frame / 255.0  # normalize
            frames.append(frame.transpose(2, 0, 1))  # CxHxW

    cap.release()
    frames = np.stack(frames, axis=0)  # [30, 3, 96, 96]
    frames = frames.transpose(1, 0, 2, 3)  # [3, 30, 96, 96]
    np.save(output_npy, frames)

def preprocess_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    video_files = list(Path(input_folder).rglob("*.mp4"))

    for video_path in tqdm(video_files):
        uid = video_path.stem
        out_video_dir = Path(output_folder) / uid
        out_video_dir.mkdir(exist_ok=True, parents=True)

        frames_path = out_video_dir / "frames.npy"
        audio_path = out_video_dir / "audio.wav"

        try:
            extract_frames(str(video_path), frames_path)
            extract_audio(str(video_path), audio_path)
        except Exception as e:
            print(f"Error with {video_path}: {e}")

if __name__ == "__main__":
    preprocess_folder("dataset/train", "data/processed/train")
    preprocess_folder("dataset/test-set", "data/processed/test")
