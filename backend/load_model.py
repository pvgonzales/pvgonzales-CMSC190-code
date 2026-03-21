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
        self.encoder.fc = nn.Identity()  # Remove original fc → 512-dim features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)
