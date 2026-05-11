"""
export_onnx.py
────────────────────────────────────────────────────────────
Export model .pth ke format ONNX untuk dipakai di webapp.
Jalankan SETELAH training & testing selesai:
    python export_onnx.py

ONNX_SAVE_PATH otomatis mengikuti MODEL_NAME di config.py
Contoh: outputs/models/best_efficientnet_b0.onnx
────────────────────────────────────────────────────────────
"""

import os
import torch
from src.model import get_model
from src.config import DEVICE, MODEL_NAME, MODEL_SAVE_PATH

ONNX_SAVE_PATH = MODEL_SAVE_PATH.replace(".pth", ".onnx")

def main():
    os.makedirs("outputs/models", exist_ok=True)

    print(f"[EXPORT] Model  : {MODEL_NAME}")
    print(f"[EXPORT] Source : {MODEL_SAVE_PATH}")
    print(f"[EXPORT] Target : {ONNX_SAVE_PATH}")

    model = get_model(MODEL_NAME, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        export_params=True,
        opset_version=11,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logit": {0: "batch_size"}
        }
    )

    size_mb = os.path.getsize(ONNX_SAVE_PATH) / (1024 * 1024)
    print(f"\n✅ Export berhasil → {ONNX_SAVE_PATH} ({size_mb:.1f} MB)")
    print("   Gunakan file .onnx ini di webapp FastAPI.")


if __name__ == "__main__":
    main()