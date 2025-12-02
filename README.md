# Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases  
### BIOVID Challenge 2025 – Dual-Factor Audio–Visual Authentication  

**Authors:**  
- **Venu Siddapura Govindaraju** – University of Naples Federico II, DIETI  
- **Stefano Marrone** – University of Naples Federico II, DIETI  
- **Carlo Sansone** – University of Naples Federico II, DIETI  

This repository contains the official implementation of our method submitted to the **BIOVID Challenge 2025**, titled:

> **“Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases.”**  
> Accepted in *ICIAP 2025 Workshop – BIOVID Challenge (LNCS Volume)*

Our system performs **open-set biometric authentication** using synchronized audio–visual MP4 videos. Each sample includes:

- **Lip-motion RGB frames**  
- **Spoken passphrase audio**

We use a **dual-stream architecture**:  
3D-ResNet18 + BiGRU for visual encoding, and ECAPA-TDNN for audio encoding.  
A **Gated Multimodal Unit (GMU)** fuses both streams into a 256-D embedding used for classification and identity verification.

---

## 🔥 Key Contributions

- **Dual-stream audio–visual architecture**  
- **GMU-based adaptive fusion**  
- **Hybrid Triplet + BCE loss**  
- **Open-set decision using cosine similarity + threshold**  
- **Full pipeline: preprocessing → training → inference → submission**

---

## 🧠 System Architecture

The architecture of the proposed multimodal authentication pipeline is shown below:

![Architecture](architecture.jpg)

---

## 📦 Dataset

The BIOVID dataset is **restricted** and not publicly available.

To access it:

1. Register for the **BIOVID Challenge 2025**
2. Submit your method description  
3. Receive download approval from the organizers  


---

## 🔧 Preprocessing

Each MP4 video sample is decomposed into synchronized visual and audio components through our preprocessing pipeline.

### 🔹 Visual Stream (Frame Extraction)

- Extracted using **FFmpeg**
- Uniformly sampled into **30 RGB frames**
- Center-cropped around the mouth region (based on BIOVID annotations)
- Resized to **96×96**
- Normalized to `[0, 1]`
- Stored as `(30 × 96 × 96 × 3)` tensors

### 🔹 Audio Stream (Waveform Extraction)

- Extracted from MP4 using FFmpeg  
- Resampled to **16 kHz**, mono  
- Pre-emphasis filtering applied  
- Normalized (zero-mean, unit variance)  
- Fed to ECAPA-TDNN for speaker embedding extraction  

### 🔹 Synchronization Guarantee

- Both modalities originate from the same MP4 file  
- No time warping or offsetting applied  
- Ensures audio–visual synchrony for fusion learning  

---

## 🔀 Multimodal Fusion (GMU)

After encoding:

- Visual embedding: **256-D**  
- Audio embedding: **192-D**

The **Gated Multimodal Unit (GMU)**:

- Learns per-dimension gates  
- Controls which modality contributes more information  
- Produces a fused **256-D multimodal embedding**

---

## 🧪 Classification & Metric Learning

Training uses two losses:

### 🔹 Binary Cross Entropy (BCE)
- Genuine vs impostor classification  
- Stabilizes decision boundaries  

### 🔹 Triplet Loss (Semi-Hard Mining)
- Encourages compact user clusters  
- Ensures separation between different identities  

Final loss:


---

## 🔐 Open-Set Verification (Cosine Similarity + Threshold)

During inference:

1. Compute cosine similarity between the test embedding and each enrolled template  
2. Select the highest similarity score  
3. Apply threshold τ = 0.60  

Decision rule:


---

## 📝 Output: BIOVID Submission File (`submission.json`)

The pipeline generates a JSON file for challenge submission.

Each entry contains:

- `video_filename`
- `predicted_user`
- `similarity_score`
- `spoken_word` (always `"unknown"` in our system)

### Example Format (`submission.json`)

```json
[
  {
    "video_filename": "6ad8f4bc2c7.mp4",
    "predicted_user": "user_012",
    "similarity_score": 0.82,
    "spoken_word": "unknown"
  },
  {
    "video_filename": "9bb0c3339.mp4",
    "predicted_user": "unknown",
    "similarity_score": 0.41,
    "spoken_word": "unknown"
  }
]




## Preprocessing

Each MP4 video sample is decomposed into synchronized visual and audio components through our preprocessing pipeline:

### 🔹 Visual Stream (Frame Extraction)
- Each input video is decoded using **FFmpeg**.  
- The video is uniformly sampled into **30 RGB frames**, regardless of its original duration.  
- This ensures fixed-length temporal representation across all samples.
- All frames are **center-cropped** around the mouth region (based on the BIOVID dataset annotations).
- Frames are resized to **96×96 pixels**.
- Pixel values are normalized to the range **[0, 1]** and stored as `(30, 96, 96, 3)` tensors.

This produces a **consistent visual sequence** capturing lip motion dynamics throughout the spoken passphrase.

### 🔹 Audio Stream (Waveform Extraction)
- The audio track is extracted from the MP4 file using **FFmpeg**.
- The waveform is resampled to **16 kHz**, mono channel, 16-bit PCM format.
- A **pre-emphasis filter** is applied to improve high-frequency information.
- The waveform is normalized to have zero mean and unit variance.
- The final audio tensor is fed into the ECAPA-TDNN pipeline for generating speaker embeddings.

This ensures a **clean, normalized speech signal** compatible with state-of-the-art speaker modeling.

### 🔹 Synchronization Guarantee
- Both audio and frames come from the same MP4 video.
- No temporal warping or skipping is applied.
- The system uses fixed-rate sampling (30 frames + full waveform) which preserves temporal synchrony between lip movements and speech.
### 🔹  Multimodal Fusion Using GMU (Gated Multimodal Unit)

The two embeddings (visual + audio) are fused with a **GMU**, which:

- Learns a per-dimension gating vector  
- Decides which modality is more informative for each sample  
- Produces a **256-D fused embedding**:

### 🔹 Classification & Metric Learning

Two training heads are applied on top of the fused embedding:

#### 🔹 Binary Classification Head**  
- Predicts whether an input is *genuine* or *impostor*  
- Trained with **Binary Cross-Entropy Loss (BCE)**  

#### 🔹 Triplet Loss Head (Semi-Hard Mining)**  
- Ensures embeddings of the same user are close  
- Ensures embeddings of different users are separated  

The final training objective combines both:

### 🔹 Open-Set Verification (Cosine Similarity + Threshold)

At inference time:

1. Compute cosine similarity between the fused embedding `f` and templates of enrolled users  
2. Choose the identity with the *maximum* similarity  
3. Apply a threshold (τ = 0.60):


## 📝 Output: BIOVID Submission File (`submission.json`)

The final step of the pipeline generates the official **BIOVID Challenge submission file**:

This JSON file contains one entry per test video, including:

- **video_filename** – the input MP4 file name  
- **predicted_user** – predicted user ID or `"unknown"`  
- **similarity_score** – cosine similarity between the test embedding and the closest enrolled template  
- **spoken_word** – `"unknown"` (for our method, which does not perform keyword recognition)  

### Example Format (`submission.json`)

```json
[
  {
    "video_filename": "6ad8f4bc2c7.mp4",
    "predicted_user": "user_012",
    "similarity_score": 0.82,
    "spoken_word": "unknown"
  },
  {
    "video_filename": "9bb0c3339.mp4",
    "predicted_user": "unknown",
    "similarity_score": 0.41,
    "spoken_word": "unknown"
  }
]




## 📂 Project Structure

```text
Biovid-Challenge2025/
├── data/                     # EXCLUDED – BIOVID dataset not included
│
├── datasets/
│   └── biovid_dataset.py     # Data loader + preprocessing
│
├── models/
│   ├── audio_encoder.py      # ECAPA-TDNN
│   ├── visual_encoder.py     # 3D-ResNet18 + BiGRU
│   ├── gmu_fusion.py         # Gated Multimodal Unit
│   ├── fusion_head.py
│   └── output_head.py
│
├── samplers/
│   └── triplet_sampler.py    # Semi-hard triplet mining
│
├── scripts/
│   ├── preprocessing/
│   │   └── preprocess.py     # Frame extraction & audio extraction
│   ├── inference/
│   └── utils/
│
├── notebooks/
│   ├── updated_pipeline.ipynb
│   └── 02_model_visual.ipynb
│
├── results/
│   ├── fold0_best_model.pt
│   ├── fold1_best_model.pt
│   ├── fold2_best_model.pt
│   └── gmu_fusion/
│       └── fold0_best_model.pt
│
├── submission/
│   └── submission.json
│
├── train_crossval.py
├── test_inference_vote.py
├── evaluate_eer.py
├── requirements.txt
└── README.md

