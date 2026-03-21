import os
import shutil
import math
import torch
import torch.nn as nn
import copy
from collections import Counter
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import re
import random
import matplotlib.pyplot as plt

#gpu optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#load and prepare the dataset
class UFCCrimeVideoDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.clip_length = clip_length
        self.videos = []
        self.labels = []

        self.classes = ['Normal', 'Assault', 'Abuse', 'Robbery', 'Shooting']
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
                    os.path.join(video_path, f) for f in os.listdir(video_path)
                    if f.lower().endswith(valid_ext)
                ])

                if len(frames) >= clip_length:
                    self.videos.append((frames, class_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames_list, label_str = self.videos[idx]

        max_start_idx = len(frames_list) - self.clip_length
        start_idx = random.randint(0, max_start_idx)
        clip_paths = frames_list[start_idx : start_idx + self.clip_length]

        clip_tensors = []
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            clip_tensors.append(transforms.ToTensor()(img))

        video_tensor = torch.stack(clip_tensors)

        if self.transform:
            v1, v2 = self.transform(video_tensor)
        else:
            v1, v2 = video_tensor, video_tensor

        v1 = v1.permute(1, 0, 2, 3)
        v2 = v2.permute(1, 0, 2, 3)

        label = self.class_to_idx[label_str]

        return (v1, v2), label


#applied variations of view for the video clip
#returns the augmented video
class BYOLVideoAugmentation:

    def __init__(self, image_size=112, blur_p=1.0, solarize_p=0.0):
        self.image_size = image_size
        self.blur_p = blur_p
        self.solarize_p = solarize_p

    def __call__(self, video):
        T, C, H, W = video.shape
        #RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            video[0], scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )

        #horizontal flip
        do_flip = random.random() < 0.5

        #colorJitter
        do_jitter = random.random() < 0.8
        if do_jitter:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            jitter_order = list(range(4))
            random.shuffle(jitter_order)

        #grayscale
        do_grayscale = random.random() < 0.2

        #gaussianBlur
        do_blur = random.random() < self.blur_p
        if do_blur:
            sigma = random.uniform(0.1, 2.0)

        #solarize
        do_solarize = random.random() < self.solarize_p

        #apply to each frame the same parameters for augmentation
        augmented_frames = []
        for t in range(T):
            frame = video[t]
            frame = TF.resized_crop(frame, i, j, h, w, [self.image_size, self.image_size])
            if do_flip:
                frame = TF.hflip(frame)
            if do_jitter:
                for op_idx in jitter_order:
                    if op_idx == 0:
                        frame = TF.adjust_brightness(frame, brightness)
                    elif op_idx == 1:
                        frame = TF.adjust_contrast(frame, contrast)
                    elif op_idx == 2:
                        frame = TF.adjust_saturation(frame, saturation)
                    elif op_idx == 3:
                        frame = TF.adjust_hue(frame, hue)
            if do_grayscale:
                frame = TF.rgb_to_grayscale(frame, num_output_channels=3)
            if do_blur:
                frame = TF.gaussian_blur(frame, kernel_size=11, sigma=sigma)
            if do_solarize:
                frame = TF.solarize(frame, threshold=0.5)
            frame = TF.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            augmented_frames.append(frame)

        return torch.stack(augmented_frames)

#generate two augmented views of the video
class BYOLTransform:
    def __init__(self, image_size=112):
        self.view1_aug = BYOLVideoAugmentation(
            image_size=image_size, blur_p=1.0, solarize_p=0.0
        )
        self.view2_aug = BYOLVideoAugmentation(
            image_size=image_size, blur_p=0.1, solarize_p=0.2
        )

    def __call__(self, video):
        return self.view1_aug(video), self.view2_aug(video)


#BYOL model
class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim=512, hidden_dim=4096, projection_dim=256):
        super().__init__()
        self.online_encoder = backbone
        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        z1 = self.online_projector(self.online_encoder(x1))
        p1 = self.online_predictor(z1)
        with torch.no_grad():
            z2 = self.target_projector(self.target_encoder(x2))
        return p1, z2

    @torch.no_grad()
    def update_target_network(self, m=0.99):
        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)


def regression_loss(p, z):
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


def cosine_ema_schedule(base_momentum, current_step, total_steps):
    return 1 - (1 - base_momentum) * (math.cos(math.pi * current_step / total_steps) + 1) / 2


#pretraining setup
if __name__ == '__main__':
    ROOT_DIR = '/content/dataset/'
    CHECKPOINT_DIR = '/content/drive/MyDrive/data-CMSC190/byol-checkpoints/'

    #hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    NUM_WORKERS = 4
    BASE_LR = 3e-4 * (BATCH_SIZE / 16)
    WARMUP_EPOCHS = 10
    BASE_EMA_MOMENTUM = 0.996
    MAX_GRAD_NORM = 1.0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    #initialize the data
    full_dataset = UFCCrimeVideoDataset(
        root_dir=ROOT_DIR, clip_length=16, transform=BYOLTransform()
    )

    all_indices = list(range(len(full_dataset)))
    all_labels = full_dataset.labels

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=42,
    )

    train_dataset = Subset(full_dataset, train_indices)

    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = Counter(train_labels)
    num_classes = len(full_dataset.classes)

    #weight = 1 / (num_classes * count_of_that_class)
    class_weights = {cls: 1.0 / (num_classes * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    #load the backbone ResNet3D-18 for BYOL
    r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    r3d.fc = nn.Identity()

    model = BYOL(backbone=r3d, feature_dim=512).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-6)

    #learning rate scheduler
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}*.pth")

    #training loop
    epoch_losses = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for idx, ((view1, view2), _) in enumerate(train_loader):
            view1 = view1.cuda(non_blocking=True)
            view2 = view2.cuda(non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p1, z2 = model(view1, view2)
                p2, z1 = model(view2, view1)
                loss = regression_loss(p1, z2) + regression_loss(p2, z1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            ema_m = cosine_ema_schedule(BASE_EMA_MOMENTUM, global_step, total_steps)
            model.update_target_network(m=ema_m)
            global_step += 1

            total_loss += loss.item()

            if idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"EMA: {ema_m:.5f} "
                    f"LR: {current_lr:.2e}"
                )

        avg_epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)

        #save checkpoint
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
            },
            f"{CHECKPOINT_DIR}byol_ckpt_e{epoch}_s{len(train_loader)}.pth",
        )

        print(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")