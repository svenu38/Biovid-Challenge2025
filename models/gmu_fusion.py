import torch
import torch.nn as nn

class GMUFusion(nn.Module):
    def __init__(self, input_v_dim=256, input_a_dim=192, hidden_dim=256):
        super(GMUFusion, self).__init__()
        self.v_proj = nn.Linear(input_v_dim, hidden_dim)
        self.a_proj = nn.Linear(input_a_dim, hidden_dim)
        self.z_gate = nn.Linear(input_v_dim + input_a_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, v, a):
        # v: [B, 256], a: [B, 192]
        a_proj = self.a_proj(a)        # [B, 256]
        z_input = torch.cat([v, a], dim=1)  # [B, 448]
        z = self.sigmoid(self.z_gate(z_input))  # [B, 256]

        h = z * self.tanh(self.v_proj(v)) + (1 - z) * self.tanh(a_proj)  # [B, 256]
        return h
