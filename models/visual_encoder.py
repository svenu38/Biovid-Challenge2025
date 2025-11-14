# models/visual_encoder.py

import torch
import torch.nn as nn
import torchvision.models.video as video_models

class VisualEncoder(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=128):
        super(VisualEncoder, self).__init__()
        # Load pretrained ResNet3D-18
        resnet3d = video_models.r3d_18(pretrained=True)
        resnet3d.stem[0] = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.cnn = nn.Sequential(*list(resnet3d.children())[:-2])  # Remove avgpool and fc

        self.gru = nn.GRU(input_size=512, hidden_size=hidden_dim, num_layers=1,
                          batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2 * hidden_dim, embedding_dim)  # Bi-GRU → 2× hidden_dim

    def forward(self, x):  # x: [B, 3, 30, 96, 96]
        B = x.size(0)
        x = self.cnn(x)  # [B, 512, T/2, 3, 3]
        x = x.mean([-1, -2])  # [B, 512, T']
        x = x.permute(0, 2, 1)  # [B, T', 512]
        _, h = self.gru(x)     # h: [2, B, hidden_dim]
        h = torch.cat([h[0], h[1]], dim=1)  # [B, 2×hidden_dim]
        return self.fc(h)  # [B, embedding_dim]
