# DrishT Excellence Roadmap: From Prototype to Production-Grade OCR

> **Objective**: Achieve industry-leading OCR performance for Indian scene text
> 
> | Metric | Current | Target | World-Class Reference |
> |--------|---------|--------|----------------------|
> | Detection mAP@0.5 | 0.23 | **0.85+** | DBNet++ (0.87) |
> | Recognition CER | 7.8% | **<2%** | TrOCR (1.2%) |
> | Word Accuracy | 81.9% | **95%+** | PARSeq (96.5%) |
> | End-to-End F1 | ~19% | **90%+** | PaddleOCR (92%) |

---

## Part 1: Strategic Analysis

### Why Current Models Fail

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAILURE CASCADE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA ISSUES (Root Cause)                                       │
│  ├── Text class: 15% of detection data (should be 60%+)        │
│  ├── Indian scripts: <5K samples/script (need 50K+)            │
│  └── Synthetic-real gap: Clean fonts ≠ noisy shop signs        │
│                                                                 │
│  ARCHITECTURE ISSUES                                            │
│  ├── SSD: Poor for small objects, single-scale features        │
│  ├── CRNN: No attention, limited context, fixed sequence       │
│  └── No end-to-end optimization (detection ↛ recognition)      │
│                                                                 │
│  TRAINING ISSUES                                                │
│  ├── No curriculum (hard samples overwhelm learning)           │
│  ├── Class imbalance (vehicles >> text)                        │
│  └── Single-resolution training (320px too small)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Excellence Formula

```
Excellent OCR = Better Architecture × Better Data × Better Training × Smart Optimization
```

Each component must be at 95% quality for the product to work at 85% combined.

---

## Part 2: Detection Excellence Path

### Phase D1: Architecture Upgrade (Week 1-2)

**Current**: SSDLite-MobileNetV3-Large (3.4M params, mAP 0.23)

**Target**: DBNet++ with ResNet-50 backbone (25M params, mAP 0.85+)

#### Why DBNet++?
| Factor | SSD | DBNet++ |
|--------|-----|---------|
| Small text (<20px) | Poor | Excellent |
| Curved/rotated text | Cannot | Native support |
| Multi-scale fusion | Limited | FPN + deformable |
| Training efficiency | Slower | 2x faster convergence |

#### Implementation Plan

```python
# Detection Architecture v2 (DBNet++)
# File: drisht/models/detection_v2.py

"""
Components:
1. ResNet-50 backbone (pretrained ImageNet)
2. Feature Pyramid Network (FPN) with deformable convolutions
3. Differentiable Binarization head
4. Adaptive Scale Aggregation (ASA)
"""

class DBNetPlusPlus(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        # Backbone with dilated convolutions in stage 4-5
        self.backbone = build_resnet50_dcn(pretrained)
        
        # FPN with 4 scales: 1/4, 1/8, 1/16, 1/32
        self.fpn = FeaturePyramidNetwork(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Adaptive Scale Aggregation
        self.asa = AdaptiveScaleAggregation(in_channels=256, scales=4)
        
        # DB Head: probability map + threshold map + binary map
        self.db_head = DifferentiableBinarizationHead(
            in_channels=256,
            k=50,  # amplification factor
            adaptive=True
        )
        
    def forward(self, x):
        # Multi-scale features
        features = self.backbone(x)  # [C2, C3, C4, C5]
        fpn_features = self.fpn(features)  # [P2, P3, P4, P5]
        
        # Aggregate scales
        fused = self.asa(fpn_features)  # Single fused feature map
        
        # Predict maps
        prob_map, thresh_map, binary_map = self.db_head(fused)
        
        return {
            'prob_map': prob_map,      # Text regions [0, 1]
            'thresh_map': thresh_map,  # Adaptive threshold
            'binary_map': binary_map,  # Final binary mask
        }
```

#### Resolution Strategy
| Training Phase | Resolution | Purpose |
|----------------|------------|---------|
| Warmup (epoch 1-5) | 640×640 | Quick convergence |
| Main (epoch 6-50) | 1280×720 | Standard training |
| Fine-tune (epoch 51-70) | 2560×1440 | High-res refinement |

### Phase D2: Data Pipeline Overhaul (Week 2-4)

#### Current Data Problems
- TotalText: 1,555 images (curved text, English)
- ICDAR 2015: 1,500 images (focused, English)
- IDD: 10,000 images (vehicles, minimal text labels)

#### Required Data Sources

| Dataset | Images | Languages | Why Critical |
|---------|--------|-----------|--------------|
| **ICDAR 2017 MLT** | 10,000 | 9 languages (Hindi, Bengali...) | Multi-script scene text |
| **ICDAR 2019 MLT** | 20,000 | 10 languages | Latest benchmark |
| **TextOCR** | 28,134 | English | Dense text annotations |
| **COCO-Text** | 63,686 | English | In-the-wild diversity |
| **Uber-Text** | 117,000 | English | Street-view signs |
| **SynthText800k** | 800,000 | Multi | Synthetic pretraining |
| **IIIT-Indian** | 5,000 | 10 Indian scripts | Core Indian data |
| **DIL (custom)** | Generate 50K | Hindi, Tamil, Bengali | Fill script gaps |

#### Data Generation Pipeline

```python
# Generate photorealistic Indian text images
# File: drisht/data/synth_indian.py

class IndianTextSynthesizer:
    """Generate synthetic scene text in Indian scripts"""
    
    def __init__(self):
        # Load fonts for each script
        self.fonts = {
            'hindi': self._load_fonts('Noto Sans Devanagari', 'Mangal', 'Kruti Dev'),
            'tamil': self._load_fonts('Noto Sans Tamil', 'Latha', 'TAU Palladam'),
            'bengali': self._load_fonts('Noto Sans Bengali', 'Vrinda', 'Shonar Bangla'),
            'telugu': self._load_fonts('Noto Sans Telugu', 'Gautami', 'Vani'),
            'kannada': self._load_fonts('Noto Sans Kannada', 'Tunga'),
            'malayalam': self._load_fonts('Noto Sans Malayalam', 'Kartika'),
            'gujarati': self._load_fonts('Noto Sans Gujarati', 'Shruti'),
            'punjabi': self._load_fonts('Noto Sans Gurmukhi', 'Raavi'),
            'odia': self._load_fonts('Noto Sans Oriya', 'Kalinga'),
            'marathi': self._load_fonts('Noto Sans Devanagari', 'Mangal'),  # Same as Hindi
        }
        
        # Background textures (walls, metal, wood, cloth)
        self.backgrounds = self._load_backgrounds()
        
        # Text corpora per script
        self.corpora = self._load_corpora()  # IndicCorp, Wikipedia dumps
        
    def generate_sample(self, script='hindi', difficulty='hard'):
        """Generate one synthetic image with text"""
        
        # 1. Sample random text
        text = self._sample_text(script, max_words=5)
        
        # 2. Render text with random style
        text_layer = self._render_text(
            text,
            font=random.choice(self.fonts[script]),
            size=random.randint(20, 80),
            color=self._sample_color(),
            effects=self._sample_effects(difficulty)  # shadow, glow, outline
        )
        
        # 3. Apply perspective transform
        text_layer = self._perspective_transform(text_layer, max_angle=30)
        
        # 4. Composite onto background
        bg = random.choice(self.backgrounds)
        composite = self._composite(bg, text_layer)
        
        # 5. Apply degradations
        composite = self._degrade(
            composite,
            blur=random.uniform(0, 2),
            noise=random.uniform(0, 0.1),
            jpeg_quality=random.randint(30, 95)
        )
        
        # 6. Generate polygon annotation (for DB loss)
        polygon = self._get_text_polygon(text_layer)
        
        return composite, polygon, text
```

### Phase D3: Advanced Training (Week 4-6)

#### Loss Function Evolution

```python
# DBNet++ Loss with improvements
# File: drisht/losses/db_loss.py

class DBNetPlusPlusLoss(nn.Module):
    """
    Components:
    1. Binary Cross-Entropy for probability map
    2. L1 loss for threshold map
    3. Dice loss for binary map (handles class imbalance)
    4. OHEM (Online Hard Example Mining)
    """
    
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3):
        super().__init__()
        self.alpha = alpha  # Threshold loss weight
        self.beta = beta    # Binary loss weight
        self.ohem_ratio = ohem_ratio
        
    def forward(self, pred, target):
        prob_map = pred['prob_map']
        thresh_map = pred['thresh_map']
        binary_map = pred['binary_map']
        
        gt_prob = target['gt_prob']      # Ground truth text regions
        gt_thresh = target['gt_thresh']  # Ground truth thresholds
        mask = target['mask']            # Ignore regions
        
        # 1. Probability map loss (BCE + OHEM)
        prob_loss = self._binary_cross_entropy_ohem(
            prob_map, gt_prob, mask
        )
        
        # 2. Threshold map loss (L1 inside text regions)
        thresh_loss = self._l1_loss_masked(
            thresh_map, gt_thresh, gt_prob
        )
        
        # 3. Binary map loss (Dice for class balance)
        binary_loss = self._dice_loss(binary_map, gt_prob, mask)
        
        # Total loss
        loss = prob_loss + self.alpha * thresh_loss + self.beta * binary_loss
        
        return loss, {
            'prob_loss': prob_loss,
            'thresh_loss': thresh_loss,
            'binary_loss': binary_loss,
        }
```

#### Training Schedule

```yaml
# Detection training config v2
# File: configs/detection_v2.yaml

model:
  name: DBNetPlusPlus
  backbone: resnet50_dcn
  pretrained: imagenet
  neck: fpn
  head: db_head

data:
  train_datasets:
    - name: synthtext800k
      weight: 0.3
    - name: icdar_mlt_2019
      weight: 0.25
    - name: totaltext
      weight: 0.15
    - name: textocr
      weight: 0.15
    - name: indian_synth_50k
      weight: 0.15
  
  augmentation:
    - RandomCrop: {min_scale: 0.3, max_scale: 1.0}
    - RandomRotate: {max_angle: 30}
    - ColorJitter: {brightness: 0.5, contrast: 0.5, saturation: 0.5}
    - RandomBlur: {kernel_range: [3, 7]}
    - GaussianNoise: {std: 0.05}
    - JPEG_Compression: {quality_range: [30, 100]}

training:
  epochs: 100
  batch_size: 16  # Per GPU
  
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.0001
  
  scheduler:
    name: CosineAnnealingWarmRestarts
    T_0: 10     # Initial restart period
    T_mult: 2   # Double period after each restart
    eta_min: 1e-6
  
  curriculum:
    # Start with easy samples, progressively add hard ones
    phases:
      - epochs: [1, 20]
        filter: "easy"  # Large text, clear background
      - epochs: [21, 50]
        filter: "medium"  # Multi-scale, moderate occlusion
      - epochs: [51, 100]
        filter: "all"  # Include hard samples
  
  mixed_precision: true
  gradient_checkpointing: true  # Trade compute for memory
  
  ema:
    enabled: true
    decay: 0.9999

evaluation:
  metrics: [mAP_50, mAP_75, precision, recall, f1]
  eval_interval: 2  # Every 2 epochs
```

---

## Part 3: Recognition Excellence Path

### Phase R1: Architecture Upgrade (Week 1-2)

**Current**: CRNN-Light (3M params, CER 7.8%)

**Target**: PARSeq-inspired Transformer (30M params, CER <2%)

#### Architecture Comparison

| Component | CRNN-Light | PARSeq-style |
|-----------|------------|--------------|
| Encoder | CNN (5 stages) | ViT + CNN hybrid |
| Sequence Model | BiLSTM | Transformer decoder |
| Decoding | Greedy CTC | Iterative refinement |
| Context | Local (LSTM window) | Global attention |
| Language Model | None | Built-in character LM |

#### Implementation

```python
# Recognition Architecture v2 (Vision Transformer + Decoder)
# File: drisht/models/recognition_v2.py

class DrishTRecognizer(nn.Module):
    """
    Hybrid CNN-Transformer for multi-script text recognition.
    
    Architecture:
    1. Shallow CNN for local features (fast)
    2. Vision Transformer for global context
    3. Permutation Language Model decoder (handles Indian script order)
    4. Iterative refinement (correct errors)
    """
    
    def __init__(
        self,
        num_classes: int = 725,  # 724 chars + blank
        img_h: int = 48,         # Increased from 32
        img_w: int = 192,        # Increased from 128
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        max_seq_len: int = 25,
        num_iterations: int = 3,
    ):
        super().__init__()
        
        # 1. Shallow CNN encoder (3 stages only)
        self.cnn = nn.Sequential(
            # Stage 1: 48×192 → 24×96
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2: 24×96 → 12×48
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 12×48 → 6×24
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 2. Patch embedding from CNN features
        self.patch_embed = nn.Linear(256 * 6, embed_dim)  # 6 = img_h / 8
        
        # 3. Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 24, embed_dim))  # 24 = img_w / 8
        
        # 4. Vision Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=depth,
        )
        
        # 5. Character queries (learnable)
        self.char_queries = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # 6. Transformer decoder (autoregressive)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=6,
        )
        
        # 7. Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 8. Iterative refinement
        self.num_iterations = num_iterations
        self.refine_embed = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, x, targets=None):
        B = x.size(0)
        
        # CNN features: (B, 256, 6, 24)
        cnn_feats = self.cnn(x)
        
        # Reshape to sequence: (B, 24, 256*6)
        feats = cnn_feats.permute(0, 3, 1, 2).flatten(2)
        
        # Project and add positional encoding
        feats = self.patch_embed(feats) + self.pos_embed
        
        # Encode with transformer
        memory = self.encoder(feats)
        
        # Initialize character queries
        queries = self.char_queries.expand(B, -1, -1)
        
        # Iterative decoding
        outputs = []
        for i in range(self.num_iterations):
            # Decode
            decoded = self.decoder(queries, memory)
            logits = self.classifier(decoded)
            outputs.append(logits)
            
            # Refine queries using predictions (for next iteration)
            if i < self.num_iterations - 1:
                preds = logits.argmax(dim=-1)
                refined = self.refine_embed(preds)
                queries = queries + 0.1 * refined  # Residual refinement
        
        # Return all iterations for training, last for inference
        if self.training:
            return outputs  # List of (B, max_seq_len, num_classes)
        else:
            return outputs[-1]
```

### Phase R2: Script-Aware Training (Week 2-4)

#### Multi-Script Data Strategy

```python
# Balanced multi-script dataset
# File: drisht/data/multiscript_dataset.py

class MultiScriptDataset(Dataset):
    """
    Balanced sampling across scripts with curriculum learning.
    
    Strategy:
    1. Oversample low-resource scripts
    2. Use script-specific augmentation
    3. Curriculum: single script → mixed → hard
    """
    
    SCRIPTS = {
        'devanagari': ['hindi', 'marathi', 'nepali', 'sanskrit'],
        'latin': ['english'],
        'bengali': ['bengali', 'assamese'],
        'tamil': ['tamil'],
        'telugu': ['telugu'],
        'kannada': ['kannada'],
        'malayalam': ['malayalam'],
        'gujarati': ['gujarati'],
        'gurmukhi': ['punjabi'],
        'odia': ['odia'],
    }
    
    def __init__(self, data_root, target_per_script=50000):
        self.samples = []
        self.script_indices = {s: [] for s in self.SCRIPTS}
        
        for script, languages in self.SCRIPTS.items():
            script_samples = self._load_script_data(data_root, script, languages)
            
            # Oversample if needed
            if len(script_samples) < target_per_script:
                ratio = target_per_script // len(script_samples)
                script_samples = script_samples * ratio
            
            start_idx = len(self.samples)
            self.samples.extend(script_samples)
            end_idx = len(self.samples)
            self.script_indices[script] = list(range(start_idx, end_idx))
    
    def get_balanced_sampler(self):
        """Return sampler that balances scripts in each batch"""
        script_weights = []
        for i, sample in enumerate(self.samples):
            # Find which script this belongs to
            for script, indices in self.script_indices.items():
                if i in indices:
                    # Weight inversely proportional to script frequency
                    weight = 1.0 / len(indices)
                    break
            script_weights.append(weight)
        
        return WeightedRandomSampler(script_weights, len(self.samples))
```

### Phase R3: Language Model Integration (Week 4-6)

#### Character-Level Language Model

```python
# Script-aware character language model
# File: drisht/models/char_lm.py

class IndicCharLM(nn.Module):
    """
    Character-level language model for Indian scripts.
    
    Trained on:
    - IndicCorp (9 Indian languages, 500M+ sentences)
    - Wikipedia dumps
    - Common Crawl filtered
    
    Usage:
    - Rescore beam search hypotheses
    - Guide iterative refinement
    - Correct OCR errors post-hoc
    """
    
    def __init__(self, vocab_size=725, embed_dim=256, num_layers=4):
        super().__init__()
        
        # Script detector (which script is this?)
        self.script_classifier = nn.Linear(embed_dim, len(SCRIPTS))
        
        # Per-script LM heads (share lower layers)
        self.shared_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 4, embed_dim * 4, 0.1),
            num_layers=num_layers - 1,
        )
        
        self.script_heads = nn.ModuleDict({
            script: nn.TransformerEncoderLayer(embed_dim, 4, embed_dim * 4, 0.1)
            for script in SCRIPTS
        })
        
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, script=None):
        """
        Args:
            x: (B, T) character indices
            script: (B,) script names or None for auto-detect
        """
        embeds = self.embedding(x)
        shared_out = self.shared_layers(embeds)
        
        if script is None:
            # Detect script from content
            script_logits = self.script_classifier(shared_out.mean(dim=1))
            script = self.SCRIPTS[script_logits.argmax(dim=1)]
        
        # Apply script-specific head
        out = self.script_heads[script](shared_out)
        return self.output(out)
```

#### Beam Search with LM Fusion

```python
# Beam search decoder with language model
# File: drisht/decoding/beam_search.py

class LMFusedBeamSearch:
    """
    Beam search with language model fusion and length normalization.
    """
    
    def __init__(
        self,
        lm: IndicCharLM,
        beam_width: int = 10,
        lm_weight: float = 0.3,
        length_penalty: float = 0.6,
    ):
        self.lm = lm
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
    
    def decode(self, encoder_output, max_len=25):
        """
        Decode with LM-fused beam search.
        
        Score = (1 - lm_weight) * ocr_score + lm_weight * lm_score
        Final = score / (length ^ length_penalty)
        """
        B = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize beams
        beams = [{
            'tokens': [BOS_TOKEN],
            'ocr_score': 0.0,
            'lm_score': 0.0,
        } for _ in range(self.beam_width)]
        
        for t in range(max_len):
            candidates = []
            
            for beam in beams:
                if beam['tokens'][-1] == EOS_TOKEN:
                    candidates.append(beam)
                    continue
                
                # Get OCR probabilities
                ocr_logprobs = self.ocr_step(encoder_output, beam['tokens'])
                
                # Get LM probabilities
                lm_logprobs = self.lm_step(beam['tokens'])
                
                # Fuse scores
                fused = (1 - self.lm_weight) * ocr_logprobs + self.lm_weight * lm_logprobs
                
                # Expand beam
                topk_scores, topk_indices = fused.topk(self.beam_width)
                
                for score, idx in zip(topk_scores, topk_indices):
                    new_beam = {
                        'tokens': beam['tokens'] + [idx.item()],
                        'ocr_score': beam['ocr_score'] + ocr_logprobs[idx].item(),
                        'lm_score': beam['lm_score'] + lm_logprobs[idx].item(),
                    }
                    candidates.append(new_beam)
            
            # Prune to beam_width
            candidates.sort(key=lambda b: self._score(b), reverse=True)
            beams = candidates[:self.beam_width]
        
        return beams[0]['tokens']
    
    def _score(self, beam):
        """Apply length normalization"""
        raw_score = beam['ocr_score'] + self.lm_weight * beam['lm_score']
        length = len(beam['tokens'])
        return raw_score / (length ** self.length_penalty)
```

---

## Part 4: End-to-End Integration

### Joint Training Strategy

```python
# End-to-end OCR with differentiable crops
# File: drisht/models/e2e_ocr.py

class EndToEndOCR(nn.Module):
    """
    Jointly trained detection + recognition.
    
    Key innovation: Differentiable ROI cropping allows gradients
    to flow from recognition loss back to detection.
    """
    
    def __init__(self, detector, recognizer):
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        
        # Differentiable ROI pooling
        self.roi_pool = ROIAlignRotated(
            output_size=(48, 192),  # Recognition input size
            spatial_scale=1.0 / 4,   # Feature map stride
            sampling_ratio=2,
        )
    
    def forward(self, images, gt_boxes=None, gt_texts=None):
        # 1. Detection
        det_output = self.detector(images)
        
        if self.training:
            # Use ground truth boxes for training
            rois = gt_boxes
        else:
            # Use predicted boxes for inference
            rois = self._extract_rois(det_output)
        
        # 2. Differentiable crop
        crops = self.roi_pool(images, rois)
        
        # 3. Recognition
        rec_output = self.recognizer(crops)
        
        return {
            'detection': det_output,
            'recognition': rec_output,
            'crops': crops,
        }
```

### Multi-Task Loss

```python
# Joint loss with uncertainty weighting
# File: drisht/losses/e2e_loss.py

class EndToEndLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weights.
    
    L_total = (1/σ_det²) * L_det + (1/σ_rec²) * L_rec + log(σ_det) + log(σ_rec)
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable log-variance parameters
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_rec = nn.Parameter(torch.zeros(1))
        
        self.det_loss = DBNetPlusPlusLoss()
        self.rec_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    def forward(self, pred, target):
        # Detection loss
        l_det = self.det_loss(pred['detection'], target['detection'])
        
        # Recognition loss
        l_rec = self.rec_loss(
            pred['recognition'].log_softmax(2).permute(1, 0, 2),
            target['texts'],
            target['rec_lengths'],
            target['text_lengths'],
        )
        
        # Uncertainty-weighted combination
        precision_det = torch.exp(-self.log_var_det)
        precision_rec = torch.exp(-self.log_var_rec)
        
        loss = (
            precision_det * l_det + self.log_var_det +
            precision_rec * l_rec + self.log_var_rec
        )
        
        return loss, {
            'det_loss': l_det,
            'rec_loss': l_rec,
            'det_weight': precision_det.item(),
            'rec_weight': precision_rec.item(),
        }
```

---

## Part 5: Infrastructure Requirements

### Compute Requirements

| Phase | GPU Hours | Instance Type | Est. Cost |
|-------|-----------|---------------|-----------|
| Detection pretraining (SynthText800k) | 100h | A100 40GB | $400 |
| Detection fine-tuning | 50h | A100 40GB | $200 |
| Recognition pretraining | 80h | A100 40GB | $320 |
| Recognition fine-tuning | 40h | A100 40GB | $160 |
| End-to-end joint training | 60h | A100 40GB | $240 |
| Hyperparameter search | 40h | A100 40GB | $160 |
| **Total** | **370h** | - | **~$1,500** |

### Training Platforms

| Platform | Cost/hr (A100) | Pros | Cons |
|----------|----------------|------|------|
| **Kaggle** | Free (30h/week) | Free | Limited time |
| **Google Colab Pro+** | $50/month | Easy | Disconnections |
| **Lambda Labs** | $1.10/hr | Best value | Wait times |
| **AWS** | $3.50/hr | Reliable | Expensive |
| **Vast.ai** | $0.80/hr | Cheapest | Variable quality |

**Recommendation**: Lambda Labs for main training, Kaggle for experimentation.

### Monitoring & Experiment Tracking

```yaml
# Weights & Biases config
# File: configs/wandb_config.yaml

project: drisht-v2
entity: djtsingh

config:
  # Log everything
  log_model: true
  log_code: true
  
  # Visualization
  visualize:
    - detection_samples
    - recognition_samples
    - attention_maps
    - loss_curves
    - learning_rate
  
  # Alerts
  alerts:
    - metric: val_mAP
      condition: "<0.5"
      message: "mAP plateau detected"
    - metric: val_cer
      condition: ">10"
      message: "CER regression"
```

---

## Part 6: Timeline & Milestones

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXCELLENCE ROADMAP                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WEEK 1-2: Foundation                                                        │
│  ├── [D] Implement DBNet++ architecture                                      │
│  ├── [R] Implement Transformer recognizer                                    │
│  ├── [I] Set up Lambda Labs training environment                             │
│  └── [I] Download & preprocess all datasets                                  │
│      Milestone: New architectures running locally                            │
│                                                                              │
│  WEEK 3-4: Data & Pretraining                                                │
│  ├── [D] Pretrain detector on SynthText800k                                  │
│  ├── [R] Generate 50K Indian synthetic text                                  │
│  ├── [R] Pretrain recognizer on MJSynth + SynthText                         │
│  └── [D] Add ICDAR MLT data pipeline                                         │
│      Milestone: Pretrained models (det mAP >0.5, rec CER <5%)               │
│                                                                              │
│  WEEK 5-6: Fine-tuning                                                       │
│  ├── [D] Fine-tune on ICDAR + TotalText + Indian                            │
│  ├── [R] Fine-tune on real Indian text                                       │
│  ├── [R] Train character language model                                      │
│  └── [E] Implement LM-fused beam search                                      │
│      Milestone: Strong individual models (det mAP >0.75, rec CER <3%)       │
│                                                                              │
│  WEEK 7-8: Integration & Polish                                              │
│  ├── [E] End-to-end joint training                                           │
│  ├── [E] Hyperparameter optimization                                         │
│  ├── [I] Quantization (INT8) and ONNX export                                │
│  └── [I] Mobile deployment testing                                           │
│      Milestone: EXCELLENT (det mAP >0.85, rec CER <2%, e2e F1 >90%)         │
│                                                                              │
│  Legend: [D]=Detection [R]=Recognition [E]=End-to-End [I]=Infrastructure    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Success Criteria & Evaluation

### Evaluation Benchmark Suite

| Benchmark | Purpose | Target |
|-----------|---------|--------|
| ICDAR 2015 | Latin scene text | mAP 0.85+ |
| ICDAR 2017 MLT | Multi-script | mAP 0.80+ |
| TotalText | Curved text | mAP 0.82+ |
| **DrishT-Indian** (new) | 10 Indian scripts | mAP 0.85+, CER <2% |

### Create DrishT-Indian Benchmark

```python
# DrishT-Indian evaluation benchmark
# File: drisht/evaluation/drisht_benchmark.py

"""
DrishT-Indian Benchmark
=======================
A comprehensive evaluation suite for Indian OCR systems.

Components:
1. 5000 manually annotated Indian scene images
2. 10 scripts: Devanagari, Bengali, Tamil, Telugu, Kannada, 
              Malayalam, Gujarati, Gurmukhi, Odia, Latin
3. Difficulty levels: easy (signs), medium (shop boards), hard (graffiti)
4. Tasks: detection, recognition, end-to-end

Metrics:
- Detection: mAP@0.5, mAP@0.75, recall@0.9
- Recognition: CER, WER, 1-NED, word accuracy
- End-to-end: H-mean F1, script accuracy
"""

class DrishTBenchmark:
    def evaluate_detection(self, model, test_loader):
        """Standard COCO-style evaluation"""
        ...
    
    def evaluate_recognition(self, model, test_loader):
        """Per-script CER/WER computation"""
        ...
    
    def evaluate_e2e(self, model, test_loader):
        """End-to-end F1 with IoU matching"""
        ...
    
    def generate_report(self):
        """Generate detailed PDF report with visualizations"""
        ...
```

---

## Part 8: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data licensing issues | Medium | High | Use only permissive datasets (CC, research-only) |
| Compute costs exceed budget | Medium | Medium | Start with Kaggle, scale to Lambda as needed |
| Indic font rendering issues | High | Medium | Use PIL + harfbuzz for proper shaping |
| Model too large for edge | Low | High | Progressive quantization, prune if needed |
| Overfitting to synthetic | High | High | Heavy real-data fine-tuning, domain adversarial |
| Script confusion | Medium | Medium | Script classifier head, separate codebooks |

---

## Part 9: Files to Create

```
drisht/
├── models/
│   ├── detection_v2.py      # DBNet++ implementation
│   ├── recognition_v2.py    # Transformer recognizer
│   ├── e2e_ocr.py           # End-to-end model
│   └── char_lm.py           # Character language model
├── losses/
│   ├── db_loss.py           # DBNet loss functions
│   └── e2e_loss.py          # Joint loss
├── data/
│   ├── synth_indian.py      # Indian text synthesizer
│   ├── multiscript_dataset.py
│   └── augmentation.py      # Advanced augmentations
├── decoding/
│   ├── beam_search.py       # LM-fused beam search
│   └── postprocess.py       # Result formatting
├── evaluation/
│   └── drisht_benchmark.py  # Evaluation suite
└── configs/
    ├── detection_v2.yaml
    ├── recognition_v2.yaml
    └── e2e_training.yaml
```

---

## Quick Start Checklist

- [ ] Fork [DBNet++](https://github.com/MhLiao/DB) as detection base
- [ ] Fork [PARSeq](https://github.com/baudm/parseq) as recognition base  
- [ ] Download ICDAR 2017/2019 MLT datasets
- [ ] Set up Weights & Biases project
- [ ] Request Lambda Labs GPU access
- [ ] Install IndicCorp for language model training
- [ ] Install Noto fonts for all Indian scripts

---

**This is the path to excellence. No shortcuts, no goofups. Let's build world-class Indian OCR.**
