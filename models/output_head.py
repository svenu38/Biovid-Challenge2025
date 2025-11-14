import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputHead(nn.Module):
    def __init__(self, input_dim=256):
        super(OutputHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # binary class: genuine / impostor
        )
        self.embedding_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        logits = self.classifier(x)  # [B, 2]
        embedding = F.normalize(self.embedding_proj(x), p=2, dim=1)  # [B, 256]
        return logits, embedding
