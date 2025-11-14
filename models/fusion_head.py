# models/fusion_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMUFusion(nn.Module):
    def __init__(self, audio_dim=192, visual_dim=256, fusion_dim=256):
        super(GMUFusion, self).__init__()

        # Project audio to visual dimension
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.visual_proj = nn.Identity()  # already 256

        # GMU gate
        self.z_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        
        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),  # Binary classification
            nn.Sigmoid()
        )

        # Triplet head (normalized embedding)
        self.triplet_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, audio_emb, visual_emb):
        a = self.audio_proj(audio_emb)  # [B, 256]
        v = self.visual_proj(visual_emb)  # [B, 256]
        
        z_input = torch.cat([a, v], dim=1)  # [B, 512]
        z = torch.sigmoid(self.z_gate(z_input))  # [B, 256]

        fused = z * a + (1 - z) * v  # [B, 256]

        score = self.out_head(fused)  # [B, 1]
        emb = F.normalize(self.triplet_head(fused), p=2, dim=1)  # [B, 256]

        return score, emb
