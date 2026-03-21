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

#preprocessing pipeline
def get_byol_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dino_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


#manages the loading for BYOL and DINOv3
class ModelManager:
    def __init__(self, models_dir: str = "model"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        self.transforms = {}
        self.available_models = []

        logger.info(f"Inference device: {self.device}")
        self._discover_models()

    def _discover_models(self):
        os.makedirs(self.models_dir, exist_ok=True)
        self.available_models.append({
            "id": "byol",
            "name": "BYOL (R3D-18)",
            "description": "3D ResNet-18 with BYOL self-supervised pretraining.",
            "input_format": "clip",
            "crop_size": 112,
        })

        try:
            import transformers
            self.available_models.append({
                "id": "dino",
                "name": "DINOv3 (ViT-S/16)",
                "description": "Vision Transformer with DINO self-distillation.",
                "input_format": "frames",
                "crop_size": 224,
            })
        except ImportError:
            logger.warning("transformers not installed, DINOv3 model unavailable")

    def get_available_models(self):
        return self.available_models

    def load_model(self, model_id: str):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.transforms[model_id]

        if model_id == "byol":
            model, transform = self._load_byol()
        elif model_id == "dino":
            model, transform = self._load_dino()
        else:
            raise ValueError(f"Unknown model: {model_id}")

        self.loaded_models[model_id] = model
        self.transforms[model_id] = transform
        return model, transform

    def _load_byol(self):
        model = BYOLVideoClassifier(num_classes=5, dropout=0.5)

        byol_weights_path = os.path.join(self.models_dir, "byol_model.pth")
        if os.path.exists(byol_weights_path):
            logger.info(f"Loading BYOL model weights: {byol_weights_path}")
            ckpt = torch.load(byol_weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("BYOL model loaded.")
        else:
            logger.warning(f"No BYOL weights at {byol_weights_path}. ")
            from torchvision.models.video import R3D_18_Weights
            pretrained = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            pretrained.fc = nn.Identity()
            model.encoder.load_state_dict(pretrained.state_dict(), strict=False)

        model = model.to(self.device)
        model.eval()
        transform = get_byol_transform()
        return model, transform
