"""
SSD-VGG16 Detection Training Script
======================================
Fine-tunes SSD-VGG16 on the DrishT detection dataset (COCO JSON format).

Usage:
    python src/ssd/train.py [--epochs N] [--batch-size N] [--lr LR] [--resume PATH]

Requires: pip install torch torchvision
"""

import os
import sys
import json
import time
import argparse
import random
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ssd.model import SSDVGG16, MultiBoxLoss, PriorBoxes
from src.ssd.config import SSDConfig as cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class COCODetectionDataset(Dataset):
    """COCO JSON format dataset for SSD training."""

    def __init__(self, json_path, img_dir, transform=None, input_size=300):
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.img_dir = Path(img_dir)
        self.input_size = input_size
        self.transform = transform

        self.images = {img["id"]: img for img in coco["images"]}
        self.img_ids = list(self.images.keys())

        # Group annotations by image_id
        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Filter to images with annotations
        self.img_ids = [iid for iid in self.img_ids if len(self.img_to_anns[iid]) > 0]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Get annotations
        anns = self.img_to_anns[img_id]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Normalize to [0, 1]
            boxes.append([
                x / orig_w,
                y / orig_h,
                (x + w) / orig_w,
                (y + h) / orig_h,
            ])
            labels.append(ann["category_id"])

        # Resize image
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
            image = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(image)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        return image, boxes, labels


def collate_fn(batch):
    """Custom collate: images stacked, boxes/labels as lists."""
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(boxes), list(labels)


# ---------------------------------------------------------------------------
# Box Encoding (match priors to ground truth)
# ---------------------------------------------------------------------------
def encode_targets(boxes_list, labels_list, priors, iou_threshold=0.5):
    """
    Match ground truth boxes to prior boxes and encode regression targets.

    Returns:
        cls_targets: (B, N) class labels for each prior
        reg_targets: (B, N, 4) encoded box offsets
    """
    batch_size = len(boxes_list)
    n_priors = priors.size(0)
    cls_targets = torch.zeros(batch_size, n_priors, dtype=torch.long)
    reg_targets = torch.zeros(batch_size, n_priors, 4)

    for b in range(batch_size):
        boxes = boxes_list[b]
        labels = labels_list[b]

        if len(boxes) == 0:
            continue

        # Compute IoU between priors (cx,cy,w,h) and GT (x1,y1,x2,y2)
        # Convert priors to x1,y1,x2,y2
        prior_x1y1x2y2 = torch.cat([
            priors[:, :2] - priors[:, 2:] / 2,
            priors[:, :2] + priors[:, 2:] / 2,
        ], dim=1)

        iou = box_iou(prior_x1y1x2y2, boxes)  # (N, M)

        # Best GT for each prior
        best_gt_iou, best_gt_idx = iou.max(dim=1)

        # Best prior for each GT (ensure each GT matches at least one prior)
        best_prior_iou, best_prior_idx = iou.max(dim=0)
        for gt_idx in range(len(boxes)):
            best_prior = best_prior_idx[gt_idx]
            best_gt_idx[best_prior] = gt_idx
            best_gt_iou[best_prior] = 2.0  # force match

        # Assign labels
        matched_labels = labels[best_gt_idx]
        matched_labels[best_gt_iou < iou_threshold] = 0  # background
        cls_targets[b] = matched_labels

        # Encode box offsets
        matched_boxes = boxes[best_gt_idx]  # (N, 4) in x1,y1,x2,y2
        # Convert to cx,cy,w,h
        matched_cx = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2
        matched_cy = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2
        matched_w = matched_boxes[:, 2] - matched_boxes[:, 0]
        matched_h = matched_boxes[:, 3] - matched_boxes[:, 1]

        variances = [0.1, 0.2]
        reg_targets[b, :, 0] = (matched_cx - priors[:, 0]) / (priors[:, 2] * variances[0])
        reg_targets[b, :, 1] = (matched_cy - priors[:, 1]) / (priors[:, 3] * variances[0])
        reg_targets[b, :, 2] = torch.log(matched_w / priors[:, 2] + 1e-6) / variances[1]
        reg_targets[b, :, 3] = torch.log(matched_h / priors[:, 3] + 1e-6) / variances[1]

    return cls_targets, reg_targets


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in x1,y1,x2,y2 format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, priors, epoch):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_loc = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, boxes_list, labels_list in pbar:
        images = images.to(device)

        # Encode targets
        cls_targets, reg_targets = encode_targets(boxes_list, labels_list, priors)
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                cls_preds, reg_preds = model(images)
                loss, cls_loss, loc_loss = criterion(cls_preds, reg_preds, cls_targets, reg_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_preds, reg_preds = model(images)
            loss, cls_loss, loc_loss = criterion(cls_preds, reg_preds, cls_targets, reg_targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_cls += cls_loss
        running_loc += loc_loss

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{cls_loss:.4f}",
            "loc": f"{loc_loss:.4f}",
        })

    n = len(loader)
    return running_loss / n, running_cls / n, running_loc / n


@torch.no_grad()
def validate(model, loader, criterion, device, priors):
    model.eval()
    running_loss = 0.0

    for images, boxes_list, labels_list in tqdm(loader, desc="Val"):
        images = images.to(device)
        cls_targets, reg_targets = encode_targets(boxes_list, labels_list, priors)
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)

        cls_preds, reg_preds = model(images)
        loss, _, _ = criterion(cls_preds, reg_preds, cls_targets, reg_targets)
        running_loss += loss.item()

    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SSD-VGG16 Text Detector")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data transforms
    train_transform = T.Compose([
        T.ColorJitter(brightness=cfg.COLOR_JITTER, contrast=cfg.COLOR_JITTER,
                       saturation=cfg.COLOR_JITTER, hue=cfg.COLOR_JITTER / 3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = COCODetectionDataset(
        cfg.TRAIN_JSON, cfg.TRAIN_IMAGES, train_transform, cfg.INPUT_SIZE)
    val_dataset = COCODetectionDataset(
        cfg.VAL_JSON, cfg.VAL_IMAGES, val_transform, cfg.INPUT_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # Model
    model = SSDVGG16(num_classes=cfg.NUM_CLASSES).to(device)
    priors = model.priors.cpu()  # Keep on CPU for encoding
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prior boxes: {priors.size(0)}")

    # Loss
    criterion = MultiBoxLoss(cfg.NUM_CLASSES, cfg.NEG_POS_RATIO, cfg.ALPHA)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=cfg.LR_MIN)

    # AMP
    scaler = GradScaler("cuda") if (cfg.USE_AMP and not args.no_amp and device.type == "cuda") else None

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")

    # Training loop
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  SSD-VGG16 Training — {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_cls, train_loc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, priors, epoch)

        val_loss = validate(model, val_loader, criterion, device, priors)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} (cls={train_cls:.4f}, loc={train_loc:.4f}) | "
              f"Val: {val_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, cfg.CHECKPOINT_DIR / "best.pth")
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth")

        # Early stopping
        if patience_counter >= cfg.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={cfg.PATIENCE})")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {cfg.CHECKPOINT_DIR / 'best.pth'}")


if __name__ == "__main__":
    main()
