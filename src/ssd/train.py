"""
SSDLite-MobileNetV3 Detection Training Script
================================================
Fine-tunes SSDLite320 with MobileNetV3-Large backbone on the
DrishT detection dataset (COCO JSON format).

Uses torchvision's built-in detection API which handles:
  - Anchor generation and matching
  - Multi-box loss computation
  - NMS during inference

Usage:
    python src/ssd/train.py [--epochs 80] [--batch-size 16] [--lr 0.01]
    python src/ssd/train.py --resume models/detection/best.pth

Requires: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ssd.model import create_model, save_model, model_info, freeze_backbone
from src.ssd.config import SSDConfig as cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class COCODetectionDataset(Dataset):
    """COCO JSON format dataset for torchvision detection models.

    Returns:
        image: Tensor [3, H, W] in [0, 1] range
        target: dict with 'boxes' (x1,y1,x2,y2 absolute) and 'labels'
    """

    def __init__(self, json_path, img_dir, augment=False, input_size=320):
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.img_dir = Path(img_dir)
        self.input_size = input_size
        self.augment = augment

        self.images = {img["id"]: img for img in coco["images"]}
        self.img_ids = list(self.images.keys())

        # Group annotations by image_id
        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Keep only images that have annotations
        self.img_ids = [iid for iid in self.img_ids if len(self.img_to_anns[iid]) > 0]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Collect annotations — boxes in absolute pixel coords
        boxes = []
        labels = []
        for ann in self.img_to_anns[img_id]:
            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])  # Convert to x1, y1, x2, y2
            labels.append(ann["category_id"])

        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Augmentations (applied before converting to tensor)
        if self.augment:
            import random
            # Random horizontal flip
            if random.random() < cfg.RANDOM_HORIZONTAL_FLIP:
                image = TF.hflip(image)
                if len(boxes) > 0:
                    boxes[:, [0, 2]] = orig_w - boxes[:, [2, 0]]

            # Color jitter
            if random.random() < 0.5:
                image = T.ColorJitter(
                    brightness=cfg.COLOR_JITTER,
                    contrast=cfg.COLOR_JITTER,
                    saturation=cfg.COLOR_JITTER,
                    hue=cfg.COLOR_JITTER / 3,
                )(image)

        # Resize to input_size (torchvision SSD also resizes internally,
        # but doing it here ensures consistent box coordinates)
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            # Clamp to image boundaries
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.input_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.input_size)

        # Convert image to tensor [0, 1] — torchvision models normalize internally
        image = TF.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


def collate_fn(batch):
    """Custom collate for variable-size targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_map(model, loader, device, iou_threshold=0.5, max_batches=None):
    """Compute mean Average Precision (mAP) at a given IoU threshold.

    Uses a simple per-class AP approximation (11-point interpolation).
    For production evaluation, use pycocotools.
    """
    model.eval()
    all_detections = defaultdict(list)  # class -> list of (score, is_tp)
    all_n_gt = defaultdict(int)         # class -> total ground truth count

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="mAP")):
        if max_batches and batch_idx >= max_batches:
            break

        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, gt in zip(predictions, targets):
            gt_boxes = gt["boxes"].to(device)
            gt_labels = gt["labels"].to(device)

            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            # Count ground truths per class
            for lbl in gt_labels.tolist():
                all_n_gt[lbl] += 1

            # Match predictions to ground truth
            matched_gt = set()
            for i in range(len(pred_boxes)):
                cls = pred_labels[i].item()
                score = pred_scores[i].item()

                # Find best matching GT box of same class
                best_iou = 0.0
                best_gt_idx = -1
                gt_mask = gt_labels == cls
                gt_cls_indices = gt_mask.nonzero(as_tuple=True)[0]

                for gt_idx in gt_cls_indices.tolist():
                    if gt_idx in matched_gt:
                        continue
                    iou = _box_iou_single(pred_boxes[i], gt_boxes[gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
                if is_tp:
                    matched_gt.add(best_gt_idx)

                all_detections[cls].append((score, is_tp))

    # Compute per-class AP
    aps = {}
    for cls in sorted(all_n_gt.keys()):
        dets = sorted(all_detections.get(cls, []), key=lambda x: -x[0])
        n_gt = all_n_gt[cls]
        if n_gt == 0:
            continue

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, is_tp in dets:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / n_gt)

        # 11-point interpolation
        ap = 0.0
        for r_threshold in [i / 10.0 for i in range(11)]:
            p_at_r = 0.0
            for p, r in zip(precisions, recalls):
                if r >= r_threshold:
                    p_at_r = max(p_at_r, p)
            ap += p_at_r / 11.0
        aps[cls] = ap

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    return mean_ap, aps


def _box_iou_single(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    """Train for one epoch. Returns average total loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        n_batches += 1

        cls_loss = loss_dict.get("classification", torch.tensor(0.0)).item()
        reg_loss = loss_dict.get("bbox_regression", torch.tensor(0.0)).item()
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "cls": f"{cls_loss:.4f}", "reg": f"{reg_loss:.4f}"})

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate_loss(model, loader, device):
    """Compute average validation loss."""
    model.train()  # Need train mode for loss computation
    running_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(loader, desc="Val-loss"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SSDLite-MobileNetV3 Text Detector")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--eval-map-every", type=int, default=5, help="Compute mAP every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SSDLite-MobileNetV3 Detection Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Datasets
    train_dataset = COCODetectionDataset(
        cfg.TRAIN_JSON, cfg.TRAIN_IMAGES, augment=True, input_size=cfg.INPUT_SIZE)
    val_dataset = COCODetectionDataset(
        cfg.VAL_JSON, cfg.VAL_IMAGES, augment=False, input_size=cfg.INPUT_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Model
    model = create_model(num_classes=cfg.NUM_CLASSES, pretrained_backbone=True)
    model.to(device)
    print(f"\n  Model: SSDLite320 + MobileNetV3-Large")
    model_info(model)

    # Freeze backbone for initial epochs
    if cfg.FREEZE_BACKBONE_EPOCHS > 0:
        freeze_backbone(model, freeze=True)
        print(f"\n  Backbone frozen for first {cfg.FREEZE_BACKBONE_EPOCHS} epochs")

    # Optimizer — separate param groups for backbone and head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone") and p.requires_grad]

    optimizer = optim.SGD([
        {"params": head_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * 0.1},  # Lower LR for backbone
    ], momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=cfg.LR_MIN)

    # AMP scaler
    use_amp = cfg.USE_AMP and not args.no_amp and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    best_map = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_metric", float("inf"))
        best_map = ckpt.get("extra", {}).get("best_map", 0.0)
        print(f"  Resumed from epoch {start_epoch - 1}")

    # Training loop
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs (AMP: {use_amp})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if epoch == cfg.FREEZE_BACKBONE_EPOCHS + 1:
            freeze_backbone(model, freeze=False)
            # Re-create optimizer with backbone params now requiring grad
            backbone_params = list(model.backbone.parameters())
            head_params = [p for n, p in model.named_parameters()
                           if not n.startswith("backbone") and p.requires_grad]
            optimizer = optim.SGD([
                {"params": head_params, "lr": scheduler.get_last_lr()[0]},
                {"params": backbone_params, "lr": scheduler.get_last_lr()[0] * 0.1},
            ], momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=cfg.LR_MIN)
            print(f"  >> Backbone unfrozen at epoch {epoch}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch)

        # Validation loss
        val_loss = validate_loss(model, val_loader, device)

        # mAP evaluation (periodically — it's expensive)
        mean_ap = 0.0
        if epoch % args.eval_map_every == 0 or epoch == args.epochs:
            mean_ap, per_class_ap = compute_map(
                model, val_loader, device, max_batches=50)
            ap_str = " | ".join(
                f"{cfg.CATEGORIES.get(c, c)}: {ap:.3f}"
                for c, ap in sorted(per_class_ap.items())
            )
            print(f"  mAP@0.5: {mean_ap:.4f}  [{ap_str}]")

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"mAP: {mean_ap:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save best (by val loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, cfg.CHECKPOINT_DIR / "best.pth",
                       epoch=epoch, optimizer=optimizer, scheduler=scheduler,
                       best_metric=best_val_loss,
                       extra={"best_map": max(best_map, mean_ap)})
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Also save best by mAP
        if mean_ap > best_map:
            best_map = mean_ap
            save_model(model, cfg.CHECKPOINT_DIR / "best_map.pth",
                       epoch=epoch, best_metric=best_map)
            print(f"  -> New best mAP model saved (mAP={mean_ap:.4f})")

        # Periodic checkpoint
        if epoch % cfg.SAVE_EVERY == 0:
            save_model(model, cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth",
                       epoch=epoch, optimizer=optimizer, scheduler=scheduler,
                       best_metric=best_val_loss)

        # Early stopping
        if patience_counter >= cfg.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={cfg.PATIENCE})")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best mAP@0.5: {best_map:.4f}")
    print(f"  Checkpoints:  {cfg.CHECKPOINT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
