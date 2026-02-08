"""
CRNN-Light Recognition Training Script
=========================================
Trains CRNN-Light (LightCNN + BiLSTM + CTC) on the DrishT recognition dataset.

Usage:
    python src/crnn/train.py [--epochs 80] [--batch-size 64] [--lr 1e-3]
    python src/crnn/train.py --resume models/recognition/best.pth

Requires: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.model import CRNN, CharCodec
from src.crnn.config import CRNNConfig as cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RecognitionDataset(Dataset):
    """Word crop + label dataset from CSV.

    Handles:
      - Aspect-ratio-preserving resize to fixed height (32px)
      - Right-padding to fixed width (128px)
      - Grayscale or RGB
      - Training augmentation (brightness jitter)
    """

    def __init__(self, csv_path, img_dir, codec, img_h=32, img_w=128,
                 num_channels=1, augment=False):
        self.img_dir = Path(img_dir)
        self.codec = codec
        self.img_h = img_h
        self.img_w = img_w
        self.num_channels = num_channels
        self.augment = augment

        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip()
                if not label or label == "UNK":
                    continue
                encoded = self.codec.encode(label)
                if any(idx == 0 for idx in encoded):
                    continue  # Skip if any char not in charset
                self.samples.append((row["image"], label))

        print(f"  Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self.img_dir / img_name

        try:
            mode = "L" if self.num_channels == 1 else "RGB"
            image = Image.open(img_path).convert(mode)
        except Exception:
            fill = 128 if self.num_channels == 1 else (128, 128, 128)
            mode = "L" if self.num_channels == 1 else "RGB"
            image = Image.new(mode, (self.img_w, self.img_h), fill)
            label = ""

        # Resize preserving aspect ratio
        w, h = image.size
        ratio = self.img_h / h
        new_w = min(int(w * ratio), self.img_w)
        image = image.resize((new_w, self.img_h), Image.BILINEAR)

        # Pad to fixed width
        fill = 0 if self.num_channels == 1 else (0, 0, 0)
        mode = "L" if self.num_channels == 1 else "RGB"
        padded = Image.new(mode, (self.img_w, self.img_h), fill)
        padded.paste(image, (0, 0))

        # Augmentations
        if self.augment:
            import random
            if random.random() < 0.3:
                padded = T.functional.adjust_brightness(padded, random.uniform(0.7, 1.3))
            if random.random() < 0.2:
                padded = T.functional.adjust_contrast(padded, random.uniform(0.8, 1.2))

        # To tensor + normalize
        tensor = T.ToTensor()(padded)
        if self.num_channels == 1:
            tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        else:
            tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)

        # Encode label
        encoded = self.codec.encode(label)
        target = torch.tensor(encoded, dtype=torch.long)
        target_length = torch.tensor(len(encoded), dtype=torch.long)

        return tensor, target, target_length


def collate_fn(batch):
    """Collate with padded targets for CTC loss."""
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)

    max_len = max(t.size(0) for t in targets) if targets[0].size(0) > 0 else 1
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, t in enumerate(targets):
        if t.size(0) > 0:
            padded_targets[i, :t.size(0)] = t

    target_lengths = torch.stack(target_lengths, 0)
    return images, padded_targets, target_lengths


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(model, loader, codec, device, max_batches=50):
    """Compute character accuracy and word accuracy on validation set."""
    model.eval()
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    with torch.no_grad():
        for i, (images, targets, target_lengths) in enumerate(loader):
            if i >= max_batches:
                break

            images = images.to(device)
            log_probs = model(images)
            decoded = model.decode_greedy(log_probs)

            for b in range(len(decoded)):
                pred_text = codec.decode(decoded[b])
                gt_indices = targets[b, :target_lengths[b]].tolist()
                gt_text = codec.decode(gt_indices)

                total_words += 1
                if pred_text == gt_text:
                    correct_words += 1

                # Character-level accuracy
                for p, g in zip(pred_text, gt_text):
                    total_chars += 1
                    if p == g:
                        correct_chars += 1
                total_chars += abs(len(pred_text) - len(gt_text))

    char_acc = correct_chars / max(total_chars, 1) * 100
    word_acc = correct_words / max(total_words, 1) * 100
    return char_acc, word_acc


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """Train for one epoch. Returns average CTC loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    ctc_loss = nn.CTCLoss(blank=cfg.CTC_BLANK, zero_infinity=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, targets, target_lengths in pbar:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                log_probs = model(images)           # (B, T, C)
                log_probs_t = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC
                input_lengths = torch.full(
                    (images.size(0),), log_probs_t.size(0),
                    dtype=torch.long, device=device)
                loss = ctc_loss(log_probs_t, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs = model(images)
            log_probs_t = log_probs.permute(1, 0, 2)
            input_lengths = torch.full(
                (images.size(0),), log_probs_t.size(0),
                dtype=torch.long, device=device)
            loss = ctc_loss(log_probs_t, targets, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Compute average validation CTC loss."""
    model.eval()
    ctc_loss = nn.CTCLoss(blank=cfg.CTC_BLANK, zero_infinity=True)
    running_loss = 0.0
    n_batches = 0

    for images, targets, target_lengths in tqdm(loader, desc="Val"):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        log_probs = model(images).permute(1, 0, 2)
        input_lengths = torch.full(
            (images.size(0),), log_probs.size(0),
            dtype=torch.long, device=device)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CRNN-Light Text Recognizer")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--charset", type=str, default=str(cfg.CHARSET_FILE))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CRNN-Light Recognition Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Character codec
    codec = CharCodec(args.charset)
    print(f"  Charset: {codec.num_classes} classes ({codec.num_classes - 1} chars + blank)")

    # Datasets
    train_dataset = RecognitionDataset(
        cfg.TRAIN_CSV, cfg.TRAIN_IMAGES, codec,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.NUM_CHANNELS, augment=True)
    val_dataset = RecognitionDataset(
        cfg.VAL_CSV, cfg.VAL_IMAGES, codec,
        cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.NUM_CHANNELS, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # Model
    model = CRNN(num_classes=codec.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"\n  Model: CRNN-Light (LightCNN + BiLSTM + CTC)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {size_mb:.1f} MB")

    # Optimizer
    if cfg.OPTIMIZER == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr,
            betas=cfg.BETAS, weight_decay=cfg.WEIGHT_DECAY)

    # Scheduler
    if cfg.LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=cfg.LR_MIN)
    elif cfg.LR_SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

    # AMP
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch - 1}")

    # Training loop
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs (AMP: {use_amp})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(model, val_loader, device)

        # Compute accuracy metrics
        char_acc, word_acc = 0.0, 0.0
        if epoch % cfg.EVAL_EVERY == 0:
            char_acc, word_acc = compute_metrics(model, val_loader, codec, device)

        if cfg.LR_SCHEDULER == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Char: {char_acc:.1f}% | Word: {word_acc:.1f}% | "
              f"LR: {lr:.6f} | {elapsed:.1f}s")

        # Save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(str(cfg.CHECKPOINT_DIR / "best.pth"))
            print(f"  -> New best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % cfg.SAVE_EVERY == 0:
            model.save(str(cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth"))

        if patience_counter >= cfg.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {cfg.CHECKPOINT_DIR / 'best.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
