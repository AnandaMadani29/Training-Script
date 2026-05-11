"""
model.py
────────────────────────────────────────────────────────────
Model yang tersedia:
  - efficientnet_b0 → EfficientNet-B0  (~4.3M params) ✅ sudah diuji
  - resnet50        → ResNet-50        (~23M params)
  - densenet121     → DenseNet-121     (~7M params)

Cara pakai:
  from src.model import get_model
  model = get_model("densenet121")
────────────────────────────────────────────────────────────
"""

import torch.nn as nn
import timm

# Registry semua model yang didukung
MODEL_REGISTRY = {
    "efficientnet_b0": {"timm_name": "efficientnet_b0", "params": "~4.3M"},
    "resnet50":        {"timm_name": "resnet50",         "params": "~23M"},
    "densenet121":     {"timm_name": "densenet121",      "params": "~7M"},
}


def get_model(model_name: str, freeze_backbone: bool = True):
    """
    Buat model dengan classifier head untuk binary classification.
    
    Args:
        model_name    : nama model dari MODEL_REGISTRY
        freeze_backbone: True = hanya latih head (Phase 1)
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' tidak dikenal. Pilihan: {available}")

    timm_name = MODEL_REGISTRY[model_name]["timm_name"]
    model     = timm.create_model(timm_name, pretrained=True)

    # Freeze backbone untuk Phase 1
    for param in model.parameters():
        param.requires_grad = not freeze_backbone

    # Ambil jumlah fitur output backbone
    # timm menyediakan model.classifier atau model.fc tergantung arsitektur
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") \
                      else model.classifier[-1].in_features
        model.classifier = _build_head(in_features)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = _build_head(in_features)
    else:
        raise AttributeError(f"Tidak bisa menemukan classifier head untuk model {model_name}")

    # Classifier head selalu trainable
    head = model.classifier if hasattr(model, "classifier") else model.fc
    for param in head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    info      = MODEL_REGISTRY[model_name]
    print(f"[MODEL] {model_name} ({info['params']}) | "
          f"Trainable: {trainable:,} / {total:,} | Frozen: {freeze_backbone}")

    return model


def unfreeze_backbone(model):
    """Buka semua layer untuk fine-tuning (Phase 2)."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Backbone unfrozen — semua {trainable:,} params trainable")


def _build_head(in_features: int) -> nn.Sequential:
    """Classifier head dengan regularisasi kuat untuk dataset kecil."""
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )