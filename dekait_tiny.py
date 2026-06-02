import os
import time
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from thop import profile, clever_format
from kan_convolutional.KANLinear import KANLinear


DATA_ROOT = r"..."
SAVE_DIR = r"./dekait_tiny_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["browsing", "chat", "file_transfer", "video", "voip"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

MAX_PER_CLASS = 1500
IMG_SIZE = 224
PATCH_SIZE = 16
IN_CHANS = 1
NUM_CLASSES = 5

EMBED_DIM = 128
DEPTH = 4
NUM_HEADS = 4
MLP_RATIO = 2.0

DROP_RATE = 0.1
ATTN_DROP = 0.1
DROP_PATH = 0.0

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "dekait_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "dekait_last.pth")
HISTORY_CSV_PATH = os.path.join(SAVE_DIR, "dekait_history.csv")
CM_PNG_PATH = os.path.join(SAVE_DIR, "dekait_confusion_matrix.png")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


class TrafficRPDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_and_subsample(data_root, class_names, max_per_class):
    per_class = collections.defaultdict(list)

    for cls in class_names:
        cls_dir = os.path.join(data_root, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, f)
                    per_class[cls].append((img_path, CLASS_TO_IDX[cls]))

    samples = []
    for cls in class_names:
        cls_samples = per_class[cls]
        if len(cls_samples) > max_per_class:
            cls_samples = random.sample(cls_samples, max_per_class)
        samples.extend(cls_samples)

    return samples


samples = collect_and_subsample(DATA_ROOT, CLASS_NAMES, MAX_PER_CLASS)

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.2,
    random_state=SEED,
    stratify=[y for _, y in samples]
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = TrafficRPDataset(train_samples, transform=train_transform)
val_dataset = TrafficRPDataset(val_samples, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KANFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.kan1 = KANLinear(
            in_features=dim,
            out_features=hidden_dim,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )
        self.kan2 = KANLinear(
            in_features=hidden_dim,
            out_features=dim,
            grid_size=5,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, n, c = x.shape
        x = x.reshape(b * n, c)
        x = self.kan1(x)
        x = self.drop(x)
        x = self.kan2(x)
        x = self.drop(x)
        x = x.reshape(b, n, c)
        return x


class DeKAITBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.kan_ffn = KANFeedForward(dim=dim, hidden_dim=hidden_dim, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.kan_ffn(self.norm2(x)))
        return x


class DeKAIT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=5,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DeKAITBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        dist_tokens = self.dist_token.expand(b, -1, -1)

        x = torch.cat((cls_tokens, x, dist_tokens), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]
        dist_feat = x[:, -1]
        return cls_feat, dist_feat

    def forward(self, x):
        cls_feat, dist_feat = self.forward_features(x)
        cls_logits = self.head(cls_feat)
        dist_logits = self.head_dist(dist_feat)
        return cls_logits, dist_logits


def count_kanlinear(m, x, y):
    inp = x[0]
    batch_elems = inp.shape[0]
    in_features = m.in_features
    out_features = m.out_features
    macs = batch_elems * in_features * out_features
    m.total_ops += torch.DoubleTensor([macs])


def profile_model(model, device, img_size, in_chans):
    model.eval()
    dummy = torch.randn(1, in_chans, img_size, img_size).to(device)
    macs, params = profile(
        model,
        inputs=(dummy,),
        custom_ops={KANLinear: count_kanlinear},
        verbose=False
    )
    flops = macs * 2
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops_str, _ = clever_format([flops, params], "%.3f")
    return macs, flops, params, macs_str, flops_str, params_str


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        cls_logits, dist_logits = model(images)
        cls_loss = criterion(cls_logits, labels)
        dist_loss = criterion(dist_logits, labels)
        loss = 0.5 * cls_loss + 0.5 * dist_loss

        loss.backward()
        optimizer.step()

        final_logits = (cls_logits + dist_logits) / 2.0
        preds = torch.argmax(final_logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        running_loss += loss.item()
        metrics = compute_metrics(all_labels, all_preds)

        pbar.set_postfix(
            loss=f"{running_loss / batch_idx:.4f}",
            acc=f"{metrics['acc']:.4f}",
            f1=f"{metrics['f1']:.4f}"
        )

    epoch_loss = running_loss / len(loader)
    epoch_metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, epoch_metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Val {epoch}/{epochs}", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        cls_logits, dist_logits = model(images)
        cls_loss = criterion(cls_logits, labels)
        dist_loss = criterion(dist_logits, labels)
        loss = 0.5 * cls_loss + 0.5 * dist_loss

        final_logits = (cls_logits + dist_logits) / 2.0
        preds = torch.argmax(final_logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        running_loss += loss.item()
        metrics = compute_metrics(all_labels, all_preds)

        pbar.set_postfix(
            loss=f"{running_loss / batch_idx:.4f}",
            acc=f"{metrics['acc']:.4f}",
            f1=f"{metrics['f1']:.4f}"
        )

    epoch_loss = running_loss / len(loader)
    epoch_metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, epoch_metrics, np.array(all_labels), np.array(all_preds)


model = DeKAIT(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=IN_CHANS,
    num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    drop_rate=DROP_RATE,
    attn_drop_rate=ATTN_DROP,
    drop_path_rate=DROP_PATH
).to(DEVICE)

total_params, trainable_params = count_parameters(model)
macs, flops, thop_params, macs_str, flops_str, params_str = profile_model(
    model, DEVICE, IMG_SIZE, IN_CHANS
)

print(f"Params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"MACs: {macs_str}")
print(f"FLOPs: {flops_str}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

history = []
best_val_f1 = -1.0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    train_start = time.time()
    train_loss, train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epoch=epoch,
        epochs=EPOCHS
    )
    train_time = time.time() - train_start

    val_start = time.time()
    val_loss, val_metrics, y_true, y_pred = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        epoch=epoch,
        epochs=EPOCHS
    )
    val_time = time.time() - val_start

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_metrics["acc"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_loss": val_loss,
        "val_acc": val_metrics["acc"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "train_time_sec": train_time,
        "val_time_sec": val_time,
        "params": total_params,
        "macs": macs,
        "flops": flops
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss {train_loss:.4f} | train_acc {train_metrics['acc']:.4f} | train_f1 {train_metrics['f1']:.4f} | "
        f"val_loss {val_loss:.4f} | val_acc {val_metrics['acc']:.4f} | val_f1 {val_metrics['f1']:.4f}"
    )

    torch.save(model.state_dict(), LAST_MODEL_PATH)

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False)

total_time = time.time() - start_time

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

final_val_loss, final_metrics, y_true, y_pred = validate_one_epoch(
    model=model,
    loader=val_loader,
    criterion=criterion,
    device=DEVICE,
    epoch="final",
    epochs="final"
)

print()
print(f"Params: {total_params:,}")
print(f"MACs: {macs_str}")
print(f"FLOPs: {flops_str}")
print(f"Training time: {total_time:.2f} sec")
print()
print("Final Evaluation")
print(f"Val Loss   : {final_val_loss:.4f}")
print(f"Val Acc    : {final_metrics['acc']:.4f}")
print(f"Val Prec   : {final_metrics['precision']:.4f}")
print(f"Val Recall : {final_metrics['recall']:.4f}")
print(f"Val F1     : {final_metrics['f1']:.4f}")
print()
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DeKAIT Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PNG_PATH, dpi=300)
plt.show()