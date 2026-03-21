import os
import shutil
import random
import copy
import glob
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torchvision.models.video import r3d_18
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

# prepare dataset for BYOL evaluation
class UFCCrimeTrainDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, image_size=112):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.image_size = image_size
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

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

        print(f"{len(self.videos)} videos initialized")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames_list, label_str = self.videos[idx]

        max_start = len(frames_list) - self.clip_length
        start = random.randint(0, max_start)
        clip_paths = frames_list[start : start + self.clip_length]
        
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            torch.zeros(3, 224, 224), scale=(0.5, 1.0), ratio=(3/4, 4/3)
        )
        do_flip = random.random() < 0.5
        do_jitter = random.random() < 0.8
        if do_jitter:
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        clip_tensors = []
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            tensor = transforms.ToTensor()(img)

            tensor = TF.resized_crop(tensor, i, j, h, w, [self.image_size, self.image_size])
            if do_flip:
                tensor = TF.hflip(tensor)
            if do_jitter:
                tensor = TF.adjust_brightness(tensor, brightness)
                tensor = TF.adjust_contrast(tensor, contrast)
                tensor = TF.adjust_saturation(tensor, saturation)
                tensor = TF.adjust_hue(tensor, hue)

            tensor = normalize(tensor)
            clip_tensors.append(tensor)
            
        video = torch.stack(clip_tensors).permute(1, 0, 2, 3)
        label = self.class_to_idx[label_str]
        return video, label


# test dataset for BYOL evaluation
class UFCCrimeEvalDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, image_size=112):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.image_size = image_size
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

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
        center_crop = transforms.CenterCrop(self.image_size)

        clip_tensors = []
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            tensor = transforms.ToTensor()(img)
            tensor = center_crop(tensor)
            tensor = normalize(tensor)
            clip_tensors.append(tensor)

        video = torch.stack(clip_tensors).permute(1, 0, 2, 3)
        label = self.class_to_idx[label_str]
        return video, label


#sample n clips per video to assess BYOL
class UFCCrimeMultiClipDataset(Dataset):

    def __init__(self, root_dir, clip_length=16, image_size=112, num_clips=10):
        self.clip_length = clip_length
        self.image_size = image_size
        self.num_clips = num_clips
        self.videos = []
        self.labels = []

        self.classes = CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

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
        center_crop = transforms.CenterCrop(self.image_size)

        all_clips = []
        for start in starts:
            clip_paths = frames_list[start : start + self.clip_length]
            clip_tensors = []
            for path in clip_paths:
                img = Image.open(path).convert('RGB')
                img = img.resize((224, 224))
                tensor = transforms.ToTensor()(img)
                tensor = center_crop(tensor)
                tensor = normalize(tensor)
                clip_tensors.append(tensor)

            clip = torch.stack(clip_tensors).permute(1, 0, 2, 3)
            all_clips.append(clip)

        clips_tensor = torch.stack(all_clips)
        label = self.class_to_idx[label_str]
        return clips_tensor, label


class VideoClassifier(nn.Module):

    def __init__(self, encoder, num_classes=5, feature_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


@torch.no_grad()
def multiclip_predict(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for clips_batch, labels in dataloader:
        B, N, C, T, H, W = clips_batch.shape
        clips_flat = clips_batch.view(B * N, C, T, H, W).to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_flat = model(clips_flat)

        logits = logits_flat.float().view(B, N, -1).mean(dim=1)
        preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


#main program
if __name__ == '__main__':
    ROOT_DIR = '/content/dataset/'
    CHECKPOINT_DIR = '/content/drive/MyDrive/data-CMSC190/byol-checkpoints/'
    FINETUNE_DIR = '/content/drive/MyDrive/data-CMSC190/model/'
    EVAL_OUTPUT_DIR = '/content/drive/MyDrive/data-CMSC190/evaluation/'

    #hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 50
    NUM_WORKERS = 4
    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.5
    NUM_CLIPS = 10
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(FINETUNE_DIR, exist_ok=True)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    #load the pretrained BYOL
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}*.pth")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {CHECKPOINT_DIR}")

    latest_ckpt = max(checkpoints, key=os.path.getctime)

    encoder = r3d_18(weights=None)
    encoder.fc = nn.Identity()

    ckpt = torch.load(latest_ckpt, weights_only=False)
    encoder_state = {
        k.replace('online_encoder.', ''): v
        for k, v in ckpt['model_state_dict'].items()
        if k.startswith('online_encoder.')
    }
    encoder.load_state_dict(encoder_state)
    print("pretrained BYOL loaded.")

    #build classifier model
    model = VideoClassifier(encoder, num_classes=5, dropout=DROPOUT).to(DEVICE)

    #prepare the data to be evaluated
    train_dataset_full = UFCCrimeTrainDataset(
        root_dir=ROOT_DIR, clip_length=16, image_size=112
    )

    all_indices = list(range(len(train_dataset_full)))
    all_labels = train_dataset_full.labels

    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_subset = Subset(train_dataset_full, train_indices)

    train_labels = [all_labels[i] for i in train_indices]
    class_counts = Counter(train_labels)
    num_classes = len(CLASS_NAMES)

    class_weights = {cls: 1.0 / (num_classes * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_subset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )

    eval_dataset = UFCCrimeEvalDataset(root_dir=ROOT_DIR, clip_length=16, image_size=112)
    test_subset_eval = Subset(eval_dataset, test_indices)
    val_loader = DataLoader(
        test_subset_eval, batch_size=32, shuffle=False,
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
    print(f"\nClass weights: {class_weight_tensor.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    #the backbone has lower LR to preserve BYOL pretrained features
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': BACKBONE_LR},
        {'params': model.head.parameters(), 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    #start finetuning; initialize variables/values
    best_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    train_losses = []
    val_f1s = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(videos)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_loss)

        #validate and evaluate the model
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

        val_preds = torch.cat(all_preds).numpy()
        val_labels = torch.cat(all_labels_val).numpy()
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_f1s.append(val_f1)

        #print results for each epoch
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Val F1: {val_f1:.3f} | "
              f"LR: {backbone_lr:.1e}/{head_lr:.1e}")
        
        #save the model everytime it gets good f1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch,
                'val_f1': best_f1,
                'val_acc': val_acc,
            }, os.path.join(FINETUNE_DIR, 'byol_model.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    #final evaluation using the last saved model
    model.load_state_dict(best_model_state)
    model.eval()

    mc_dataset = UFCCrimeMultiClipDataset(
        root_dir=ROOT_DIR, clip_length=16, image_size=112, num_clips=NUM_CLIPS
    )
    mc_test_subset = Subset(mc_dataset, test_indices)
    mc_loader = DataLoader(
        mc_test_subset, batch_size=8, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    preds_mc, true_mc = multiclip_predict(model, mc_loader, DEVICE)

    accuracy = accuracy_score(true_mc, preds_mc)
    precision = precision_score(true_mc, preds_mc, average='weighted', zero_division=0)
    recall_val = recall_score(true_mc, preds_mc, average='weighted', zero_division=0)
    f1_val = f1_score(true_mc, preds_mc, average='weighted', zero_division=0)

    print("\nBYOL Evaluation Results")
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
    disp.plot(ax=ax, cmap='Blues', values_format='.2f')
    ax.set_title(f'BYOL Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(EVAL_OUTPUT_DIR, 'byol_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.show()
