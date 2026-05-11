"""
main_train_kfold.py — K-Fold Training dengan penyimpanan hasil otomatis
────────────────────────────────────────────────────────────────────────
Cara pakai:
  1. Buka config.py, ganti MODEL_NAME
  2. Jalankan: python main_train_kfold.py
  3. Hasil tersimpan otomatis ke outputs/results/<model>_results.json
  4. Setelah semua model selesai, jalankan: python compare_models.py
"""

import os, shutil, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from src.config import (set_seed, SEED, DEVICE, BATCH_SIZE, EPOCHS,
                         LR, DATA_DIR, MODEL_SAVE_PATH, RESULTS_PATH, MODEL_NAME)
from src.dataset import DeepfakeDataset
from src.model import get_model
from src.train import train_model


def main():
    set_seed(SEED)
    os.makedirs("outputs/models",   exist_ok=True)
    os.makedirs("outputs/results",  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL: {MODEL_NAME.upper()}")
    print(f"{'='*60}")

    # ── Dataset pool ──────────────────────────────────────────────────────────
    ds_train = DeepfakeDataset(f"{DATA_DIR}/Train",      train=True)
    ds_val   = DeepfakeDataset(f"{DATA_DIR}/Validation", train=False)
    combined   = ConcatDataset([ds_train, ds_val])
    all_labels = np.array(ds_train.labels + ds_val.labels)

    print(f"\n[K-FOLD] Pool: {len(all_labels)} gambar | "
          f"Real: {(all_labels==0).sum()} | Fake: {(all_labels==1).sum()}")

    # ── K-Fold ────────────────────────────────────────────────────────────────
    K   = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):

        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{K}  |  Train: {len(train_idx)}  |  Val: {len(val_idx)}")
        print(f"{'='*60}")

        train_loader = DataLoader(Subset(combined, train_idx),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(Subset(combined, val_idx),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        set_seed(SEED + fold)
        model = get_model(MODEL_NAME, freeze_backbone=True).to(DEVICE)

        save_path = f"outputs/models/{MODEL_NAME}_fold{fold+1}.pth"
        best_loss, best_acc = train_model(
            model, train_loader, val_loader, DEVICE, EPOCHS, LR, save_path
        )

        fold_results.append({"fold": fold+1, "val_loss": best_loss, "val_acc": best_acc})
        print(f"\n  [FOLD {fold+1}] Val Loss: {best_loss:.4f} | Val Acc: {best_acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    accs   = [r["val_acc"]  for r in fold_results]
    losses = [r["val_loss"] for r in fold_results]

    summary = {
        "model":         MODEL_NAME,
        "folds":         fold_results,
        "mean_val_acc":  round(float(np.mean(accs)),   4),
        "std_val_acc":   round(float(np.std(accs)),    4),
        "mean_val_loss": round(float(np.mean(losses)), 4),
        "std_val_loss":  round(float(np.std(losses)),  4),
        "best_fold":     fold_results[int(np.argmin(losses))]["fold"],
    }

    print(f"\n{'='*60}")
    print(f"  HASIL K-FOLD — {MODEL_NAME.upper()}")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']} | Val Acc: {r['val_acc']:.4f} | Val Loss: {r['val_loss']:.4f}")
    print(f"\n  Mean Val Acc  : {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
    print(f"  Mean Val Loss : {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
    print(f"  Best Fold     : Fold {summary['best_fold']}")
    print(f"{'='*60}")

    # Simpan hasil ke JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Hasil disimpan ke: {RESULTS_PATH}")

    # Salin model terbaik
    best_fold = summary["best_fold"]
    shutil.copy(f"outputs/models/{MODEL_NAME}_fold{best_fold}.pth", MODEL_SAVE_PATH)
    print(f"✅ Model terbaik (Fold {best_fold}) → {MODEL_SAVE_PATH}")
    print(f"\n   Langkah selanjutnya:")
    print(f"   1. Jalankan: python main_test.py  (evaluasi test set)")
    print(f"   2. Ganti MODEL_NAME di config.py, ulangi training")
    print(f"   3. Setelah semua model selesai: python compare_models.py")


if __name__ == "__main__":
    main()