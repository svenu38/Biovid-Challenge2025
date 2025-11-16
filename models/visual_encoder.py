# models/visual_encoder.py

import torch
import torch.nn as nn
import torchvision.models.video as video_models

class VisualEncoder(nn.Module):
    
    """
    Visual encoder for lip-motion analysis.
    -------------------------------------------------------
    Input:
        x : Tensor of shape [B, 3, 30, 96, 96]
            Batch of RGB video clips (30 frames, 96×96 resolution).

    Output:
        embedding : Tensor of shape [B, 256]
            256-dimensional visual embedding representing
            spatiotemporal lip-motion features.

    Functionality:
        1. Extract spatiotemporal features using a pretrained 3D-ResNet-18.
        2. Perform spatial pooling to get per-frame feature vectors.
        3. Use a bidirectional GRU to capture temporal dynamics of lip motion.
        4. Compress the forward+backward GRU states into a compact 256-D embedding.
    """

    def __init__(self, embedding_dim=256, hidden_dim=128):
        super(VisualEncoder, self).__init__()

        # 1. Load pretrained 3D ResNet-18 for spatiotemporal feature extraction.
        
        resnet3d = video_models.r3d_18(pretrained=True)
        # Modify first convolution:
        #   - Preserve temporal resolution (stride=1 in time)
        #   - Downsample spatial dimensions (stride=2 in H/W)
        #   - Use temporal kernel size 3 to capture motion early
        resnet3d.stem[0] = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))

        # Remove the last average-pooling and classification layers.
        # The output becomes: [B, 512, T', 3, 3]
        self.cnn = nn.Sequential(*list(resnet3d.children())[:-2])
        
        # 2. Bidirectional GRU to model temporal evolution of lip movements.
        #    Input per timestep: 512-dimensional CNN feature vector
        #    hidden_dim=128 → BiGRU output size = 2 * 128 = 256

        self.gru = nn.GRU(input_size=512, 
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True, 
                          bidirectional=True)
        
        # 3. Final fully connected layer to map BiGRU output → 256-D embedding
        self.fc = nn.Linear(2 * hidden_dim, embedding_dim)  # Bi-GRU → 2× hidden_dim

    def forward(self, x): 
        """
        Forward pass through the visual encoder.

        Args:
            x : Tensor [B, 3, 30, 96, 96]
                Batch of preprocessed video frames.

        Returns:
            Tensor [B, 256] : visual embedding.
        """
         # x: [B, 3, 30, 96, 96]
        B = x.size(0)
        x = self.cnn(x)  # [B, 512, T/2, 3, 3]
        x = x.mean([-1, -2])  # [B, 512, T']
        x = x.permute(0, 2, 1)  # [B, T', 512]
        _, h = self.gru(x)     # h: [2, B, hidden_dim]
        h = torch.cat([h[0], h[1]], dim=1)  # [B, 2×hidden_dim]
        return self.fc(h)  # [B, embedding_dim]


# The VisualEncoder takes a batch of lip-motion video clips ([B, 3, 30, 96, 96]), 
# uses a pretrained 3D ResNet to extract spatiotemporal features, pools them over space
# feeds the resulting time sequence into a bidirectional GRU to model motion dynamics,
#  and finally compresses everything into a 256-dimensional visual embedding per video ([B, 256]), 
# which is then used for multimodal fusion and open-set verification.