import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18
from huggingface_hub import hf_hub_download # IMPORT ADDED
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
HF_REPO_ID = os.environ.get("HF_REPO_ID")
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

# preprocessing pipeline
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


# manages the loading for BYOL and DINOv3
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

        if not os.path.exists(byol_weights_path):
            try:
                logger.info(f"Downloading BYOL weights from {HF_REPO_ID}...")
                byol_weights_path = hf_hub_download(
                    repo_id=HF_REPO_ID, 
                    filename="byol_model.pth", 
                    local_dir=self.models_dir
                )
            except Exception as e:
                logger.error(f"Failed to download BYOL model from Hugging Face: {e}")
                
        if os.path.exists(byol_weights_path):
            logger.info(f"Loading BYOL model weights: {byol_weights_path}")
            ckpt = torch.load(byol_weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("BYOL model loaded.")
        else:
            logger.warning(f"No BYOL weights at {byol_weights_path}. Falling back to pre-trained weights.")
            from torchvision.models.video import R3D_18_Weights
            pretrained = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            pretrained.fc = nn.Identity()
            model.encoder.load_state_dict(pretrained.state_dict(), strict=False)

        model = model.to(self.device)
        model.eval()
        transform = get_byol_transform()
        return model, transform

    def _load_dino(self):
        from transformers import AutoModel
        from huggingface_hub import login as hf_login

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_login(token=hf_token)
            logger.info("HuggingFace authenticated with HF_TOKEN.")
        else:
            logger.warning("HF_TOKEN environment variable not set.")

        DINOV3_HF_MODEL = os.environ.get("DINOV3_HF_MODEL")
        logger.info(f"loading DINOv3 from HuggingFace: {DINOV3_HF_MODEL}")

        hf_model = AutoModel.from_pretrained(DINOV3_HF_MODEL)
        vit_encoder = DINOv3Encoder(hf_model)
        embed_dim = hf_model.config.hidden_size

        model = DINOVideoClassifier(
            encoder=vit_encoder,
            embed_dim=embed_dim,
            num_classes=5,
            dropout=0.3,
        )

        dino_weights_path = os.path.join(self.models_dir, "dinov3_model.pth")

        if not os.path.exists(dino_weights_path):
            try:
                logger.info(f"Downloading DINOv3 weights from {HF_REPO_ID}...")
                dino_weights_path = hf_hub_download(
                    repo_id=HF_REPO_ID, 
                    filename="dinov3_model.pth", 
                    local_dir=self.models_dir
                )
            except Exception as e:
                logger.error(f"Failed to download DINOv3 model from Hugging Face: {e}")

        if os.path.exists(dino_weights_path):
            logger.info(f"Loading DINO model weights: {dino_weights_path}")
            ckpt = torch.load(dino_weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("DINO model loaded.")
        else:
            logger.warning(f"No DINO weights at {dino_weights_path}. Model heads will be uninitialized.")

        model = model.to(self.device)
        model.eval()
        transform = get_dino_transform()
        return model, transform

    @torch.no_grad()
    def predict_clip(self, model_id: str, frames: list[Image.Image]) -> dict:
        model, transform = self.load_model(model_id)
        model_info = next(m for m in self.available_models if m['id'] == model_id)

        tensors = [transform(frame) for frame in frames]

        if model_info['input_format'] == 'clip':
            clip = torch.stack(tensors, dim=1)
            clip = clip.unsqueeze(0).to(self.device)
        else:
            clip = torch.stack(tensors, dim=0)
            clip = clip.unsqueeze(0).to(self.device)

        logits = model(clip)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        class_scores = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            'prediction': pred_class,
            'confidence': confidence,
            'class_scores': class_scores,
            'color': CLASS_COLORS[pred_class],
            'is_crime': pred_class != 'Normal',
        }
        