# Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases  
### BIOVID Challenge 2025 – Dual-Factor Audio–Visual Authentication  

**Authors:**  
- **Venu Siddapura Govindaraju** – University of Naples Federico II, DIETI  
- **Stefano Marrone** – University of Naples Federico II, DIETI  
- **Carlo Sansone** – University of Naples Federico II, DIETI  

This repository contains the official implementation of our method submitted to the **BIOVID Challenge 2025**, titled:

> **“Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases.”**  
> (Accepted in ICIAP 2025 Workshop – BIOVID Challenge, LNCS Volume)

The system performs **open-set biometric authentication** using synchronized audio–visual MP4 videos. Each authentication sample includes:

- **Lip-motion RGB frames**  
- **Spoken passphrase audio**

We design a **dual-stream deep learning architecture** using a 3D-ResNet18 + BiGRU visual encoder and an ECAPA-TDNN audio encoder. These embeddings are fused using a **Gated Multimodal Unit (GMU)** to produce a 256-dimensional joint embedding for both **classification** and **identity verification**.

---

## 🔥 Key Contributions

- **Dual-stream architecture:**  
  - 3D-ResNet-18 + BiGRU (visual)  
  - ECAPA-TDNN (audio)  
- **Gated Multimodal Unit (GMU)** for adaptive, learned fusion  
- **Hybrid loss function** using Triplet Loss + Binary Cross Entropy  
- **Open-set verification** using cosine similarity + thresholding  
- Reproducible training, evaluation, and submission-generation pipeline  

---

## 🧠 System Architecture

Include your architecture figure here:

```markdown
![Architecture](architecture.jpg)


---

# 📂 Project Structure

```text
Biovid-Challenge2025/
├── data/                     # EXCLUDED – confidential BIOVID dataset
│
├── datasets/
│   └── biovid_dataset.py     # Video/audio reader + preprocessing
│
├── models/
│   ├── audio_encoder.py      # ECAPA-TDNN backbone
│   ├── visual_encoder.py     # 3D-ResNet18 + BiGRU
│   ├── gmu_fusion.py         # Gated Multimodal Unit
│   ├── fusion_head.py
│   └── output_head.py
│
├── samplers/
│   └── triplet_sampler.py
│
├── scripts/
│   ├── preprocessing/
│   │   └── preprocess.py     # Frame extraction, audio extraction
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


## 🔧 Preprocessing

Each MP4 sample is processed into:

- **30 RGB frames** (96×96)  
- **16 kHz mono audio waveform**  

Run preprocessing:

```bash
python scripts/preprocessing/preprocess.py \
    --input data/raw \
    --output data/processed


## Dataset (BIOVID Challenge 2025)

The BIOVID dataset is restricted and cannot be shared publicly.

To access it:

        Register for the BIOVID Challenge 2025

        Submit your method description

        Receive download approval from organizers

## Results
3-Fold Cross-Validation (Validation Set)
| Fold        | Accuracy   | EER        | APCER      | BPCER      |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| 0           | 72.48%     | 27.53%     | 27.47%     | 27.58%     |
| 1           | 68.46%     | 31.57%     | 31.39%     | 31.75%     |
| 2           | 73.15%     | 26.74%     | 27.17%     | 26.31%     |
| **Average** | **71.36%** | **28.61%** | **28.68%** | **28.55%** |


##BIOVID Hidden Test Set

        71.00% accuracy

        33 accepted predictions

        92 rejected as “unknown”