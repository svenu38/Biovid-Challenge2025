# Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases  
### BIOVID Challenge 2025 – Dual-Factor Lip-Based Authentication  
**Author:** Venu Siddapura Govindaraju  
**Affiliation:** University of Naples Federico II, DIETI

This repository contains the official implementation of our system submitted to the **BIOVID Challenge 2025**, titled:

**“Multimodal Biometric Authentication Using Lip Motion and Spoken Passphrases.”**

The method performs **open-set user authentication** using synchronized audio–visual MP4 videos. Each sample contains:

- **Lip-motion frames** (visual modality)  
- **Speech audio** (audio modality)  

The system extracts complementary biometric cues from both modalities using a *dual-stream encoder architecture* and fuses them using a **Gated Multimodal Unit (GMU)** to generate a discriminative 256-D joint embedding for both **classification** and **identity verification**.

---

# 🔥 Key Contributions

- Dual-stream biometric system combining **3D-ResNet-18 + BiGRU** (visual) and **ECAPA-TDNN** (audio).  
- **Gated Multimodal Unit (GMU)** for adaptive audio–visual fusion.  
- Composite training loss using **Triplet Loss (semi-hard mining)** + **Binary Cross Entropy**.  
- **Open-set decision mechanism** using cosine similarity + threshold rejection.  
- Fully reproducible pipeline with preprocessing, training, inference, and submission generation.

---

# 🧠 System Architecture

Below is the full architecture corresponding exactly to the design in the paper.

📌 Place your architecture image inside the repo as:  
`architecture.jpg`

Then reference it in README (as shown below):

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
