# models/audio_encoder.py

import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

class AudioEncoder(torch.nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(AudioEncoder, self).__init__()
        self.device = device
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": self.device}
        )

    def forward(self, audio_path):
        signal, fs = torchaudio.load(audio_path)
        if fs != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = transform(signal)
        embedding = self.model.encode_batch(signal.to(self.device))
        return embedding.squeeze(0)  # shape: [192]
