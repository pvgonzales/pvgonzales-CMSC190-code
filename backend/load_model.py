import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Normal', 'Assault', 'Abuse', 'Robbery', 'Shooting']
CLASS_COLORS = {
    'Normal': '#2ecc71',
    'Assault': "#3c78e7",
    'Abuse': '#e67e22',
    'Robbery': '#9b59b6',
    'Shooting': '#c0392b',
}
CLIP_LENGTH = 16

class BYOLVideoClassifier(nn.Module):
    def __init__(self, num_classes=5, feature_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = r3d_18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class DINOv3Encoder(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.pooler_output


class DINOVideoClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes=5, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        frames_flat = video_frames.reshape(B * T, C, H, W)
        features_flat = self.encoder(frames_flat)  # [B*T, embed_dim]
        features = features_flat.reshape(B, T, -1).mean(dim=1)  # [B, embed_dim]
        return self.head(features)
