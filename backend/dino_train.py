import os
import shutil
import math
import random
import copy
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#gpu optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CLASS_NAMES = ['Normal', 'Assault', 'Abuse', 'Robbery', 'Shooting']

#prepare the dataset for augmentation
#returns 2 global augmented views and n local augmented views
class DINOVideoAugmentation:

    def __init__(self, global_size=224, local_size=96, n_local_crops=4):
        self.global_size = global_size
        self.local_size = local_size
        self.n_local_crops = n_local_crops

    def _augment_clip(self, video, crop_size, scale_range, is_global=True):
        T, C, H, W = video.shape

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            video[0], scale=scale_range, ratio=(3/4, 4/3)
        )
        do_flip = random.random() < 0.5
        do_jitter = random.random() < 0.8
        if do_jitter:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)

        do_grayscale = random.random() < 0.2
        do_blur = random.random() < (1.0 if is_global else 0.1)
        if do_blur:
            sigma = random.uniform(0.1, 2.0)
        do_solarize = random.random() < (0.0 if is_global else 0.2)

        frames = []
        for t in range(T):
            frame = video[t]
            frame = TF.resized_crop(frame, i, j, h, w, [crop_size, crop_size])
            if do_flip:
                frame = TF.hflip(frame)
            if do_jitter:
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
                frame = TF.adjust_hue(frame, hue)
            if do_grayscale:
                frame = TF.rgb_to_grayscale(frame, num_output_channels=3)
            if do_blur:
                frame = TF.gaussian_blur(frame, kernel_size=11, sigma=sigma)
            if do_solarize:
                frame = TF.solarize(frame, threshold=0.5)
            frame = TF.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            frames.append(frame)

        return torch.stack(frames)

    def __call__(self, video):
        views = []
        for _ in range(2):
            views.append(self._augment_clip(
                video, self.global_size, scale_range=(0.4, 1.0), is_global=True
            ))
        for _ in range(self.n_local_crops):
            views.append(self._augment_clip(
                video, self.local_size, scale_range=(0.05, 0.4), is_global=False
            ))
        return views

class UFCCrimeDINODataset(Dataset):
    def __init__(self, root_dir, clip_length=16, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
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

        #random temporal sampling
        max_start = len(frames_list) - self.clip_length
        start = random.randint(0, max_start)
        clip_paths = frames_list[start : start + self.clip_length]

        clip_tensors = []
        for path in clip_paths:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            clip_tensors.append(transforms.ToTensor()(img))

        video = torch.stack(clip_tensors)

        if self.transform:
            views = self.transform(video)
        else:
            views = [video, video]

        label = self.class_to_idx[label_str]
        return views, label


#DINOv3 Model preparation through projection head
class DINOHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, out_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOVideoModel(nn.Module):
    def __init__(self, vit_encoder, embed_dim, hidden_dim=2048,
                 bottleneck_dim=256, out_dim=4096):
        super().__init__()
        self.encoder = vit_encoder
        self.embed_dim = embed_dim
        self.head = DINOHead(embed_dim, hidden_dim, bottleneck_dim, out_dim)

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        frames_flat = video_frames.reshape(B * T, C, H, W)
        features_flat = self.encoder(frames_flat)
        features = features_flat.reshape(B, T, -1).mean(dim=1)
        logits = self.head(features)
        return logits

    def get_features(self, video_frames):
        B, T, C, H, W = video_frames.shape
        frames_flat = video_frames.reshape(B * T, C, H, W)
        features_flat = self.encoder(frames_flat)
        features = features_flat.reshape(B, T, -1).mean(dim=1)
        return features


#centering and sharpening to get the loss of the model
class DINOLoss(nn.Module):

    def __init__(self, out_dim, n_crops, warmup_teacher_temp=0.04,
                 teacher_temp=0.04, warmup_teacher_temp_epochs=0,
                 n_epochs=100, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = n_crops

        self.teacher_temp_schedule = self._build_temp_schedule(
            warmup_teacher_temp, teacher_temp,
            warmup_teacher_temp_epochs, n_epochs
        )

        self.register_buffer("center", torch.zeros(1, out_dim))

    def _build_temp_schedule(self, warmup_temp, final_temp, warmup_epochs, total_epochs):
        schedule = torch.linspace(warmup_temp, final_temp, warmup_epochs).tolist()
        schedule += [final_temp] * (total_epochs - warmup_epochs)
        return schedule

    def forward(self, student_output, teacher_output, epoch):
        student_temp = self.student_temp
        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        student_probs = [s / student_temp for s in student_output]
        teacher_probs = [(t - self.center) / teacher_temp for t in teacher_output]
        teacher_probs = [F.softmax(t, dim=-1).detach() for t in teacher_probs]

        total_loss = 0
        n_loss_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_logit in enumerate(student_probs):
                if s_idx == t_idx:
                    continue

                loss = torch.sum(
                    -t_prob * F.log_softmax(s_logit, dim=-1), dim=-1
                ).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        self._update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

def cosine_ema_schedule(base_momentum, current_step, total_steps):
    return 1 - (1 - base_momentum) * (math.cos(math.pi * current_step / total_steps) + 1) / 2


def dino_collate_fn(batch):
    views_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    n_views = len(views_list[0])
    collated_views = []
    for v in range(n_views):
        stacked = torch.stack([views_list[b][v] for b in range(len(batch))])
        collated_views.append(stacked)

    return collated_views, labels


#main program
if __name__ == '__main__':
    ROOT_DIR = '/content/dataset/'
    CHECKPOINT_DIR = '/content/drive/MyDrive/data-CMSC190/dinov3-checkpoints/'

    #hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 100
    NUM_WORKERS = 4
    BASE_LR = 5e-4
    WARMUP_EPOCHS = 10
    BASE_EMA_MOMENTUM = 0.996
    MAX_GRAD_NORM = 3.0 
    N_LOCAL_CROPS = 4
    LOCAL_CROP_SIZE = 96
    OUT_DIM = 4096

    #DINOv3 setup
    DINOV3_HF_MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    STUDENT_TEMP = 0.1
    TEACHER_TEMP = 0.04
    WARMUP_TEACHER_TEMP = 0.04
    WARMUP_TEACHER_TEMP_EPOCHS = 30

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # load DINOv3 via HuggingFace
    os.system('pip install -q --upgrade "transformers>=4.52" huggingface_hub')

    from huggingface_hub import login
    from transformers import AutoModel, AutoImageProcessor

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
            return outputs.pooler_output

    hf_model = AutoModel.from_pretrained(DINOV3_HF_MODEL)
    vit_encoder = DINOv3Encoder(hf_model)
    EMBED_DIM = hf_model.config.hidden_size
    print(f"DINOv3 loaded")

    #build DINO student and teacher
    student = DINOVideoModel(
        vit_encoder=vit_encoder,
        embed_dim=EMBED_DIM,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=OUT_DIM,
    ).cuda()

    teacher = DINOVideoModel(
        vit_encoder=copy.deepcopy(vit_encoder),
        embed_dim=EMBED_DIM,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=OUT_DIM,
    ).cuda()

    for p in teacher.parameters():
        p.requires_grad = False

    #initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())

    #load dataset through data loader
    transform = DINOVideoAugmentation(
        global_size=224, local_size=LOCAL_CROP_SIZE, n_local_crops=N_LOCAL_CROPS
    )
    full_dataset = UFCCrimeDINODataset(
        root_dir=ROOT_DIR, clip_length=16, transform=transform
    )

    all_indices = list(range(len(full_dataset)))
    all_labels = full_dataset.labels
    train_indices, _ = train_test_split(
        all_indices, test_size=0.2, stratify=all_labels, random_state=42
    )
    train_dataset = Subset(full_dataset, train_indices)

    train_labels = [all_labels[i] for i in train_indices]
    class_counts = Counter(train_labels)
    num_classes = len(CLASS_NAMES)

    class_weights = {c: 1.0 / (num_classes * cnt) for c, cnt in class_counts.items()}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
        collate_fn=dino_collate_fn,
    )

    #dino loss
    dino_loss = DINOLoss(
        out_dim=OUT_DIM,
        n_crops=2 + N_LOCAL_CROPS,
        warmup_teacher_temp=WARMUP_TEACHER_TEMP,
        teacher_temp=TEACHER_TEMP,
        warmup_teacher_temp_epochs=WARMUP_TEACHER_TEMP_EPOCHS,
        n_epochs=EPOCHS,
        student_temp=STUDENT_TEMP,
    ).cuda()

    param_groups = [
        {'params': student.encoder.parameters(), 'lr': BASE_LR * 0.1},
        {'params': student.head.parameters(), 'lr': BASE_LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.04)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    #start the training loop
    start_epoch = 0
    global_step = 0
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}*.pth")

    epoch_losses = []

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        teacher.eval()
        total_loss = 0

        for idx, (views, _) in enumerate(train_loader):
            views = [v.cuda(non_blocking=True) for v in views]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                student_outputs = [student(v) for v in views]

                with torch.no_grad():
                    teacher_outputs = [teacher(v) for v in views[:2]]

                loss = dino_loss(student_outputs, teacher_outputs, epoch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < 1:
                for p in student.head.last_layer.parameters():
                    p.grad = None
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            ema_m = cosine_ema_schedule(BASE_EMA_MOMENTUM, global_step, total_steps)
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(ema_m).add_(ps.data, alpha=1 - ema_m)

            global_step += 1
            total_loss += loss.item()

            #print results per epoch
            if idx % 5 == 0:
                lr_backbone = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"EMA: {ema_m:.5f} "
                    f"LR: {lr_backbone:.2e}/{lr_head:.2e}"
                )

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        #save checkpoint
        torch.save({
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'dino_loss_state_dict': dino_loss.state_dict(),
            'global_step': global_step,
        }, f"{CHECKPOINT_DIR}dino_ckpt_e{epoch}_s{len(train_loader)}.pth")

        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
