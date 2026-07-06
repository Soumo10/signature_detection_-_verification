# ResNet-50 Siamese Network — Signature Forgery Detection

## Goal

Build a Siamese Network with **twin ResNet-50 feature extractors** (shared weights) that learns to verify whether two signature images belong to the same person. The model maps signatures into a compact embedding space using **both Contrastive Loss and Triplet Loss**, then uses Euclidean distance to classify genuine vs forged pairs.

## Dataset

The CEDAR signature dataset at `d:\signature_detection_-_verification\signatures`:

| Folder | Files | Naming Convention | Count |
|--------|-------|-------------------|-------|
| `full_org/` | Genuine signatures | `original_{writerID}_{sampleID}.png` | 1,320 (55 writers × 24) |
| `full_forg/` | Skilled forgeries | `forgeries_{writerID}_{sampleID}.png` | 1,320 (55 writers × 24) |

- **Train/Test split by writer** (not by image) — writers 1–44 train, writers 45–55 test
- This ensures the model generalises to **unseen writers**

---

## Proposed Changes

### Data Pipeline

#### [NEW] Dataset — Dual-Mode Loader

Two dataset classes to support both loss functions:

**`PairDataset`** (for Contrastive Loss):
- Returns `(img_A, img_B, label)` where label=1 (genuine pair) or label=0 (forged pair)
- Balanced 50/50 genuine vs forged pairs per writer
- Genuine pairs: two different originals from the same writer
- Forged pairs: one original + one forgery from the same writer

**`TripletDataset`** (for Triplet Loss):
- Returns `(anchor, positive, negative)` triplets
- Anchor: an original signature
- Positive: another original from the **same** writer
- Negative: a forgery of the **same** writer (hard negative) OR an original from a **different** writer (easy negative)
- Mix of hard/easy negatives for curriculum learning

#### [NEW] Augmentation

Signature-specific transforms to improve generalisation:
- `RandomAffine` (±10° rotation, ±10% translate, 90–110% scale)
- `RandomPerspective` (distortion_scale=0.15)
- `ColorJitter` (brightness/contrast ±0.3)
- `RandomErasing` (simulates ink blots / occlusion)

---

### Model Architecture

#### [NEW] `ResNet50Backbone`

```
Input (224×224×3)
       │
   ResNet-50 (pretrained on ImageNet)
   (remove final FC layer → num_classes=0)
       │
   2048-d feature vector
       │
   Projection Head:
   ├── Linear(2048 → 512) + BatchNorm + ReLU + Dropout(0.3)
   └── Linear(512 → 256)
       │
   256-d L2-normalised embedding
```

- Uses `torchvision.models.resnet50(pretrained=True)` or `timm`
- **Freeze** `conv1`, `bn1`, `layer1`, `layer2` (first 60% of the network)
- **Fine-tune** `layer3`, `layer4`, and the projection head
- L2-normalisation on the final embedding for stable distance computation

#### [NEW] `SiameseResNet50`

```python
class SiameseResNet50(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone  # shared weights

    def forward(self, x1, x2):
        e1 = self.backbone(x1)
        e2 = self.backbone(x2)
        return e1, e2
```

---

### Loss Functions

#### [NEW] Contrastive Loss

$$\mathcal{L} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

- `y=1` (genuine): minimise distance
- `y=0` (forged): push distance beyond margin `m=1.0`

#### [NEW] Triplet Loss

$$\mathcal{L} = \max\big(d(a, p) - d(a, n) + \alpha,\; 0\big)$$

- `d(a,p)`: anchor–positive distance (should be small)
- `d(a,n)`: anchor–negative distance (should be large)
- `α = 0.5`: margin

#### Training Strategy

> [!IMPORTANT]
> The notebook will train **two separate models** — one with Contrastive Loss and one with Triplet Loss — and compare their performance side-by-side.

---

### Training Loop

#### [NEW] Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image size | 224 × 224 |
| Embedding dim | 256 |
| Batch size | 32 |
| Epochs | 30 |
| Optimiser | AdamW (lr=3e-4, weight_decay=1e-4) |
| LR Schedule | CosineAnnealingLR |
| Mixed precision | FP16 via `GradScaler` + `autocast` |
| Gradient clipping | max_norm = 1.0 |

#### [NEW] Checkpointing
- Save best model weights based on validation loss
- Save full checkpoint (model + optimiser + threshold + metrics)

---

### Evaluation

#### [NEW] Metrics Computed
- **ROC curve** & **AUC** score
- **Equal Error Rate (EER)** — the operating point where FAR = FRR
- **Confusion matrix** at the EER-optimal threshold
- **Classification report** (precision, recall, F1)
- **Distance distribution histogram** (genuine vs forged)

#### [NEW] Side-by-Side Comparison
- Table comparing Contrastive vs Triplet loss:
  - AUC, EER, accuracy, training time
- Overlaid ROC curves for both models

---

### Visualisation & Inference

#### [NEW] t-SNE Embedding Plot
- Extract 256-d embeddings for test writers
- Project to 2D via t-SNE
- Colour-coded by writer, marker-coded by type (genuine vs forged)

#### [NEW] `verify_signatures()` Utility
- Accepts two image paths → returns `(is_genuine, distance, confidence)`
- Visual side-by-side comparison with verdict overlay

#### [NEW] Model Export
- Full `.pth` checkpoint with threshold and metrics
- TorchScript trace for deployment

---

## Output File

#### [NEW] [Siamese_ResNet50_Verification.ipynb](file:///d:/signature_detection_-_verification/Siamese_ResNet50_Verification.ipynb)

A single Colab-ready notebook containing the complete implementation.

---

## Verification Plan

### Automated
- Shape verification: confirm ResNet-50 backbone outputs 2048-d, projection head outputs 256-d
- Pair/triplet dataset: verify balanced label distribution
- Forward pass sanity check with random input

### Model Evaluation
- Train both models and evaluate on held-out test writers (45–55)
- Plot ROC curves, compute EER, print confusion matrices
- Compare Contrastive vs Triplet loss performance

### Manual
- Run `verify_signatures()` on known genuine and forged pairs to confirm sensible verdicts

---

## Open Questions

- Do you want **both** Contrastive and Triplet loss trained and compared, or should I pick just one?
- Should the notebook include **hard negative mining** (online selection of the hardest negatives during training) for the triplet variant?
- Any preference on the **embedding dimension** (128 vs 256 vs 512)?
