import os
import shutil
import random
import glob
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

#gpu optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CLASS_NAMES = ['Normal', 'Assault', 'Abuse', 'Robbery', 'Shooting']

#prepare the dataset for DINOv3 evaluation
class DINOTrainDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, image_size=224):
        self.clip_length = clip_length
        self.image_size = image_size
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        valid_ext = ('.jpg', '.jpeg', '.png')

        for class_name in os.listdir(root_dir):
            if class_name not in self.classes:
                continue
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for video_folder in sorted(os.listdir(class_path)):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frames = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.lower().endswith(valid_ext)
                ])

                if len(frames) >= clip_length:
                    self.videos.append((frames, class_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames_list, label_str = self.videos[idx]

        #random temporal sampling
        max_start = len(frames_list) - self.clip_length
        start = random.randint(0, max_start)
        clip_paths = frames_list[start : start + self.clip_length]

        #augmentation params
        do_flip = random.random() < 0.5
        do_jitter = random.random() < 0.8
        if do_jitter:
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)

        #random resized crop
        scale = (0.7, 1.0)
        ratio = (3/4, 4/3)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        clip_tensors = []
        crop_params = None
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            tensor = transforms.ToTensor()(img)

            if crop_params is None:
                crop_params = transforms.RandomResizedCrop.get_params(
                    tensor, scale=scale, ratio=ratio
                )
            i, j, h, w = crop_params
            tensor = TF.resized_crop(tensor, i, j, h, w,[self.image_size, self.image_size])

            if do_flip:
                tensor = TF.hflip(tensor)
            if do_jitter:
                tensor = TF.adjust_brightness(tensor, brightness)
                tensor = TF.adjust_contrast(tensor, contrast)
                tensor = TF.adjust_saturation(tensor, saturation)
                tensor = TF.adjust_hue(tensor, hue)

            tensor = normalize(tensor)
            clip_tensors.append(tensor)

        video = torch.stack(clip_tensors)
        label = self.class_to_idx[label_str]
        return video, label


# test dataset for DINOv3 evaluation
class DINOEvalDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, image_size=224):
        self.clip_length = clip_length
        self.image_size = image_size
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        valid_ext = ('.jpg', '.jpeg', '.png')

        for class_name in os.listdir(root_dir):
            if class_name not in self.classes:
                continue
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for video_folder in sorted(os.listdir(class_path)):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frames = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.lower().endswith(valid_ext)
                ])

                if len(frames) >= clip_length:
                    self.videos.append((frames, class_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames_list, label_str = self.videos[idx]

        mid = len(frames_list) // 2
        half = self.clip_length // 2
        start = max(0, mid - half)
        start = min(start, len(frames_list) - self.clip_length)
        clip_paths = frames_list[start : start + self.clip_length]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        clip_tensors = []
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            tensor = transforms.ToTensor()(img)
            tensor = normalize(tensor)
            clip_tensors.append(tensor)

        video = torch.stack(clip_tensors)
        label = self.class_to_idx[label_str]
        return video, label


#sample n clips per video to assess DINOv3
class DINOMultiClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, image_size=224, num_clips=10):
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_clips = num_clips
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        valid_ext = ('.jpg', '.jpeg', '.png')

        for class_name in os.listdir(root_dir):
            if class_name not in self.classes:
                continue
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for video_folder in sorted(os.listdir(class_path)):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frames = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.lower().endswith(valid_ext)
                ])

                if len(frames) >= clip_length:
                    self.videos.append((frames, class_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.videos)

    def _get_clip_starts(self, total_frames):
        max_start = total_frames - self.clip_length
        if max_start <= 0:
            return [0] * self.num_clips
        if self.num_clips == 1:
            return [max_start // 2]
        return [int(round(i * max_start / (self.num_clips - 1)))
                for i in range(self.num_clips)]

    def __getitem__(self, idx):
        frames_list, label_str = self.videos[idx]
        starts = self._get_clip_starts(len(frames_list))

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        all_clips = []
        for start in starts:
            clip_paths = frames_list[start : start + self.clip_length]
            clip_tensors = []
            for path in clip_paths:
                img = Image.open(path).convert('RGB')
                img = img.resize((self.image_size, self.image_size))
                tensor = transforms.ToTensor()(img)
                tensor = normalize(tensor)
                clip_tensors.append(tensor)
            all_clips.append(torch.stack(clip_tensors))

        clips_tensor = torch.stack(all_clips)
        label = self.class_to_idx[label_str]
        return clips_tensor, label

#process each frame through the ViT backbone
class DINOVideoClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes, dropout=0.3):
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

@torch.no_grad()
def multiclip_predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for clips_batch, labels in dataloader:
        B, N, T, C, H, W = clips_batch.shape
        logits_per_clip = []

        for clip_idx in range(N):
            clip = clips_batch[:, clip_idx].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(clip)
            logits_per_clip.append(logits.float().cpu())

        avg_logits = torch.stack(logits_per_clip, dim=1).mean(dim=1)
        preds = avg_logits.argmax(dim=1)

        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


#main program
if __name__ == '__main__':
    DRIVE_DIR = '/content/drive/MyDrive/data-CMSC190/'
    LOCAL_DIR = '/content/dataset/'
    CHECKPOINT_DIR = '/content/drive/MyDrive/data-CMSC190/dinov3-checkpoints/'
    FINETUNE_DIR = '/content/drive/MyDrive/data-CMSC190/model/'
    EVAL_OUTPUT_DIR = '/content/drive/MyDrive/data-CMSC190/evaluation/'

    #hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 50
    NUM_WORKERS = 4
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.3
    PATIENCE = 10
    NUM_CLIPS = 10
    CLIP_LENGTH = 16
    IMAGE_SIZE = 224

    #DINOv3 HuggingFace model
    DINOV3_HF_MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(FINETUNE_DIR, exist_ok=True)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    ROOT_DIR = LOCAL_DIR

    #load DINOv3 ViT-S/16 via HuggingFace
    os.system('pip install -q --upgrade "transformers>=4.52" huggingface_hub')

    from huggingface_hub import login
    from transformers import AutoModel

    try:
        login(token="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")
    except Exception:
        print("HF login failed")

    class DINOv3Encoder(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model

        def forward(self, x):
            outputs = self.model(pixel_values=x)
            return outputs.pooler_output  # [B, 384]

    hf_model = AutoModel.from_pretrained(DINOV3_HF_MODEL)
    vit_encoder = DINOv3Encoder(hf_model)
    EMBED_DIM = hf_model.config.hidden_size
    print(f"DINOv3 loaded via Huggingface")
    
    #load pretrained DINOv3
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}*.pth")
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"\n  Loading DINO-pretrained encoder: {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, weights_only=False)
        encoder_state = {}
        for k, v in ckpt['student_state_dict'].items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v

        if encoder_state:
            vit_encoder.load_state_dict(encoder_state)
            print("DINO encoder loaded.")
        else:
            print("No encoder")
    else:
        print("No checkpoints")

    #build classifier
    num_classes = len(CLASS_NAMES)
    model = DINOVideoClassifier(
        encoder=vit_encoder,
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    #load dataset
    print("\nScanning dataset...")
    train_dataset = DINOTrainDataset(ROOT_DIR, CLIP_LENGTH, IMAGE_SIZE)
    eval_dataset = DINOEvalDataset(ROOT_DIR, CLIP_LENGTH, IMAGE_SIZE)

    all_indices = list(range(len(train_dataset)))
    all_labels = train_dataset.labels
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)

    #class-balanced sampling
    train_labels = [all_labels[i] for i in train_indices]
    class_counts = Counter(train_labels)
    class_weights_sample = {c: 1.0 / (num_classes * cnt)
                            for c, cnt in class_counts.items()}
    sample_weights = [class_weights_sample[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_subset), replacement=True
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    #class-weighted loss
    total_train = sum(class_counts.values())
    class_weight_tensor = torch.tensor(
        [total_train / (num_classes * class_counts.get(i, 1))
         for i in range(num_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    print(f"\nClass weights: {class_weight_tensor.cpu().tolist()}")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': BACKBONE_LR},
        {'params': model.head.parameters(), 'lr': HEAD_LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    #start training for DINOv3 evaluation
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_f1s = []
    val_accs = []

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(videos)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        all_preds, all_labels_val = [], []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(videos)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels_val.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels_val = torch.cat(all_labels_val).numpy()

        val_acc = accuracy_score(all_labels_val, all_preds)
        val_f1 = f1_score(all_labels_val, all_preds, average='weighted', zero_division=0)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head = optimizer.param_groups[1]['lr']
        #print results per epoch
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {lr_backbone:.1e}/{lr_head:.1e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'best_f1': best_f1,
                'epoch': epoch,
            }, os.path.join(FINETUNE_DIR, 'dinov3_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    #DINOv3 final evaluation
    best_ckpt = torch.load(
        os.path.join(FINETUNE_DIR, 'dinov3_model.pth'),
        weights_only=False,
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    mc_dataset = DINOMultiClipDataset(
        root_dir=ROOT_DIR, clip_length=CLIP_LENGTH,
        image_size=IMAGE_SIZE, num_clips=NUM_CLIPS
    )

    _, test_indices = train_test_split(
        list(range(len(mc_dataset))), test_size=0.2,
        stratify=mc_dataset.labels, random_state=42
    )
    test_mc = Subset(mc_dataset, test_indices)
    mc_loader = DataLoader(
        test_mc, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    preds_mc, true_mc = multiclip_predict(model, mc_loader, DEVICE)

    accuracy = accuracy_score(true_mc, preds_mc)
    precision = precision_score(true_mc, preds_mc, average='weighted', zero_division=0)
    recall_val = recall_score(true_mc, preds_mc, average='weighted', zero_division=0)
    f1_val = f1_score(true_mc, preds_mc, average='weighted', zero_division=0)

    print("\nDINOv3 Evaluation Results")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Clips per video: {NUM_CLIPS}")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall_val:.4f}")
    print(f"  F1 Score:        {f1_val:.4f}")

    print("\nPer-Class Results:")
    print(classification_report(true_mc, preds_mc,
                                target_names=CLASS_NAMES, zero_division=0))

    #generate confusion matrix
    cm_norm = confusion_matrix(true_mc, preds_mc, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Greens', values_format='.2f')
    ax.set_title(
        f'DINOv3 Confusion Matrix',fontsize=13, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(EVAL_OUTPUT_DIR, 'dinov3_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.show()
