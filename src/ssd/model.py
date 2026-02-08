"""
SSD-VGG16 Text Detector — Model Definition
=============================================
Single Shot MultiBox Detector with VGG16+BN backbone for
Indian scene text and object detection.

Architecture:
  VGG16+BN (layers up to conv5_3)  →  extra conv layers  →  multi-scale detection heads

Output: 8732 default boxes across 6 feature maps with class scores + box offsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from itertools import product
from math import sqrt
from pathlib import Path

from .config import SSDConfig as cfg


# ---------------------------------------------------------------------------
# VGG16 Backbone (modified for SSD)
# ---------------------------------------------------------------------------
class VGG16Base(nn.Module):
    """VGG16 with BatchNorm, truncated after conv5_3 and modified pool5/fc layers."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        features = list(vgg.features.children())

        # conv1_1 -> conv3_3 + pool3 (output: 38x38 for 300x300 input)
        self.stage1 = nn.Sequential(*features[:24])

        # conv4_1 -> conv4_3 (output: 38x38 -- SSD uses conv4_3 features)
        self.stage2 = nn.Sequential(*features[24:34])

        # conv4_3 -> conv5_3 + pool5
        self.stage3 = nn.Sequential(*features[34:])

        # Modified fc6 -> conv6 (1024 channels, 3x3, dilation=6)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        # fc7 -> conv7 (1024 channels, 1x1)
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # L2 normalization scale for conv4_3
        self.l2_norm_scale = nn.Parameter(torch.FloatTensor(1, 512, 1, 1).fill_(20))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        conv4_3_feat = x
        norm = conv4_3_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feat = conv4_3_feat / (norm + 1e-10) * self.l2_norm_scale

        x = self.stage3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = self.conv6(x)
        conv7_feat = self.conv7(x)

        return conv4_3_feat, conv7_feat


class SSDExtraLayers(nn.Module):
    """Additional convolution layers for multi-scale detection."""

    def __init__(self):
        super().__init__()
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv9_1 = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv10_1 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv11_1 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, conv7_feat):
        x = self.conv8_1(conv7_feat)
        conv8_2_feat = self.conv8_2(x)
        x = self.conv9_1(conv8_2_feat)
        conv9_2_feat = self.conv9_2(x)
        x = self.conv10_1(conv9_2_feat)
        conv10_2_feat = self.conv10_2(x)
        x = self.conv11_1(conv10_2_feat)
        conv11_2_feat = self.conv11_2(x)
        return conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat


class SSDDetectionHead(nn.Module):
    """Classification + regression heads for each feature map scale."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        n_anchors = []
        for ar in cfg.ASPECT_RATIOS:
            n_anchors.append(2 + 2 * len(ar))
        in_channels = [512, 1024, 512, 256, 256, 256]

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for ic, na in zip(in_channels, n_anchors):
            self.cls_heads.append(nn.Conv2d(ic, na * num_classes, 3, padding=1))
            self.reg_heads.append(nn.Conv2d(ic, na * 4, 3, padding=1))

    def forward(self, features):
        batch_size = features[0].size(0)
        cls_preds, reg_preds = [], []
        for feat, cls_head, reg_head in zip(features, self.cls_heads, self.reg_heads):
            cls = cls_head(feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            reg = reg_head(feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            cls_preds.append(cls)
            reg_preds.append(reg)
        return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)


class PriorBoxes:
    """Generate SSD default (prior) boxes."""

    def __init__(self):
        self.image_size = cfg.INPUT_SIZE
        self.feature_maps = cfg.FEATURE_MAPS
        self.steps = cfg.STEPS
        self.min_sizes = cfg.MIN_SIZES
        self.max_sizes = cfg.MAX_SIZES
        self.aspect_ratios = cfg.ASPECT_RATIOS

    def generate(self):
        priors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f
                s_k = self.min_sizes[k] / self.image_size
                priors.append([cx, cy, s_k, s_k])
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                priors.append([cx, cy, s_k_prime, s_k_prime])
                for ar in self.aspect_ratios[k]:
                    priors.append([cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)])
                    priors.append([cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)])
        priors = torch.tensor(priors, dtype=torch.float32)
        priors.clamp_(min=0, max=1)
        return priors


class MultiBoxLoss(nn.Module):
    """SSD MultiBox loss with hard negative mining."""

    def __init__(self, num_classes, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets):
        batch_size = cls_preds.size(0)
        n_priors = cls_preds.size(1)

        pos_mask = cls_targets > 0
        loc_loss = F.smooth_l1_loss(reg_preds[pos_mask], reg_targets[pos_mask], reduction="sum")

        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction="none"
        ).view(batch_size, n_priors)

        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0
        _, neg_idx = cls_loss_neg.sort(dim=1, descending=True)
        _, neg_rank = neg_idx.sort(dim=1)

        n_pos = pos_mask.long().sum(dim=1, keepdim=True)
        n_neg = torch.clamp(self.neg_pos_ratio * n_pos, max=n_priors - 1)
        neg_mask = neg_rank < n_neg

        cls_loss = (cls_loss_all * (pos_mask | neg_mask).float()).sum()
        n_positives = pos_mask.float().sum()

        if n_positives == 0:
            return torch.tensor(0.0, requires_grad=True, device=cls_preds.device), 0.0, 0.0

        loc_loss = self.alpha * loc_loss / n_positives
        cls_loss = cls_loss / n_positives
        return cls_loss + loc_loss, cls_loss.item(), loc_loss.item()


class SSDVGG16(nn.Module):
    """
    SSD300 with VGG16+BN backbone for multi-class text/object detection.

    Categories: background(0), text(1), license_plate(2), traffic_sign(3),
                autorickshaw(4), tempo(5), truck(6), bus(7)
    """

    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes or cfg.NUM_CLASSES
        self.backbone = VGG16Base()
        self.extras = SSDExtraLayers()
        self.heads = SSDDetectionHead(self.num_classes)
        self.prior_gen = PriorBoxes()
        self.register_buffer("priors", self.prior_gen.generate())

    def forward(self, x):
        conv4_3_feat, conv7_feat = self.backbone(x)
        conv8_2, conv9_2, conv10_2, conv11_2 = self.extras(conv7_feat)
        features = [conv4_3_feat, conv7_feat, conv8_2, conv9_2, conv10_2, conv11_2]
        return self.heads(features)

    def detect(self, x, conf_threshold=None, nms_threshold=None):
        conf_threshold = conf_threshold or cfg.CONFIDENCE_THRESHOLD
        nms_threshold = nms_threshold or cfg.NMS_THRESHOLD
        cls_preds, reg_preds = self.forward(x)
        cls_probs = F.softmax(cls_preds, dim=-1)

        batch_results = []
        for b in range(x.size(0)):
            boxes = self._decode_boxes(reg_preds[b], self.priors)
            probs = cls_probs[b]
            detections = []
            for c in range(1, self.num_classes):
                scores = probs[:, c]
                mask = scores > conf_threshold
                if mask.sum() == 0:
                    continue
                c_scores = scores[mask]
                c_boxes = boxes[mask]
                keep = torchvision.ops.nms(c_boxes, c_scores, nms_threshold)
                keep = keep[:cfg.MAX_DETECTIONS]
                for idx in keep:
                    detections.append({"box": c_boxes[idx].tolist(), "score": c_scores[idx].item(), "class": c})
            batch_results.append(detections)
        return batch_results

    @staticmethod
    def _decode_boxes(reg_preds, priors):
        variances = [0.1, 0.2]
        cx = priors[:, 0] + reg_preds[:, 0] * variances[0] * priors[:, 2]
        cy = priors[:, 1] + reg_preds[:, 1] * variances[0] * priors[:, 3]
        w = priors[:, 2] * torch.exp(reg_preds[:, 2] * variances[1])
        h = priors[:, 3] * torch.exp(reg_preds[:, 3] * variances[1])
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        return boxes.clamp(min=0, max=1)

    def load_pretrained_backbone(self, weights_path=None):
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, num_classes=None):
        model = cls(num_classes=num_classes)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
