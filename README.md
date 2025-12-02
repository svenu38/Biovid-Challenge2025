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
---

## 📊 Results

The performance of our multimodal authentication system was evaluated on:

- **3-fold user-disjoint cross-validation** (validation set)  
- **BIOVID hidden test set** (official challenge evaluation)

Each evaluation reports the four official BIOVID metrics:

- **Accuracy** – overall identification correctness  
- **EER** – Equal Error Rate  
- **APCER** – Attack Presentation Classification Error Rate  
- **BPCER** – Bona Fide Presentation Classification Error Rate  

---

### ✓ 3-Fold Cross-Validation (Validation Set)

| **Fold** | **Accuracy (%)** | **EER (%)** | **APCER (%)** | **BPCER (%)** |
|---------:|------------------:|------------:|---------------:|---------------:|
| 0        | 72.48             | 27.53       | 27.47          | 27.58          |
| 1        | 68.46             | 31.57       | 31.39          | 31.75          |
| 2        | 73.15             | 26.74       | 27.17          | 26.31          |
| **Average** | **71.36**      | **28.61**   | **28.68**      | **28.55**      |

---

### ✓ BIOVID Hidden Test Set (Final Challenge Score)

| **Metric**              | **Value** |
|------------------------|----------:|
| Accuracy               | **71.00%** |
| Accepted Predictions   | 33         |
| Rejected as “unknown”  | 92         |

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


