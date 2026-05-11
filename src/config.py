"""
config.py
────────────────────────────────────────────────────────────
Untuk ganti model, ubah MODEL_NAME di bawah:

    MODEL_NAME = "efficientnet_b0"  # EfficientNet-B0  (~4.3M params)
    MODEL_NAME = "resnet50"         # ResNet-50        (~23M params)
    MODEL_NAME = "densenet121"      # DenseNet-121     (~7M params)
────────────────────────────────────────────────────────────
"""

import torch
import random
import numpy as np

# ════════════════════════════════════════════════════════════
#  ▶ GANTI MODEL DI SINI
MODEL_NAME = "efficientnet_b0"   # pilihan: efficientnet_b0 | resnet50 | densenet121
# ════════════════════════════════════════════════════════════

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS     = 60
LR         = 1e-4

DATA_DIR        = "Dataset"
MODEL_SAVE_PATH = f"outputs/models/best_{MODEL_NAME}.pth"
RESULTS_PATH    = f"outputs/results/{MODEL_NAME}_results.json"

print(f"[CONFIG] Model  : {MODEL_NAME}")
print(f"[CONFIG] Device : {DEVICE}")
print(f"[CONFIG] Batch  : {BATCH_SIZE} | LR: {LR} | Epochs: {EPOCHS}")
print(f"[CONFIG] Save   : {MODEL_SAVE_PATH}")