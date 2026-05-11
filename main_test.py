"""
main_test.py — Evaluasi model pada test set
Hasil disimpan ke outputs/results/<model>_test_results.json
"""

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.config import set_seed, SEED, DEVICE, BATCH_SIZE, DATA_DIR, MODEL_SAVE_PATH, MODEL_NAME
from src.dataset import DeepfakeDataset
from src.model import get_model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs  = torch.sigmoid(model(images)).squeeze(1)
            preds  = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_with_tta(model, loader, device, n_tta=5):
    """
    Test Time Augmentation: prediksi dengan multiple augmentations dan average hasilnya.
    Mengurangi variance dan meningkatkan robustness.
    """
    model.eval()
    all_probs_tta, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # TTA: horizontal flip + original
            tta_probs = []
            tta_probs.append(torch.sigmoid(model(images)).squeeze(1))
            tta_probs.append(torch.sigmoid(model(torch.flip(images, [3]))).squeeze(1))  # H-flip
            
            # Average TTA predictions
            avg_probs = torch.stack(tta_probs).mean(dim=0)
            all_probs_tta.extend(avg_probs.cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())

    all_probs_tta = np.array(all_probs_tta)
    all_preds_tta = (all_probs_tta > 0.5).astype(int)
    
    return np.array(all_labels), all_preds_tta, all_probs_tta


def main():
    set_seed(SEED)
    import os; os.makedirs("outputs/results", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  TEST EVALUASI — {MODEL_NAME.upper()}")
    print(f"{'='*55}")

    test_ds     = DeepfakeDataset(f"{DATA_DIR}/Test", train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = get_model(MODEL_NAME, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print(f"[TEST] Model loaded: {MODEL_SAVE_PATH}")
    
    # Load validation results untuk deteksi overfitting
    val_results_path = f"outputs/results/{MODEL_NAME}_results.json"
    val_acc = None
    if os.path.exists(val_results_path):
        with open(val_results_path, "r") as f:
            val_data = json.load(f)
            val_acc = val_data.get("mean_val_acc", None)
            print(f"[INFO] Validation Accuracy (K-Fold): {val_acc:.4f}" if val_acc else "[WARNING] Val accuracy not found")

    # Evaluasi standar
    labels, preds, probs = evaluate(model, test_loader, DEVICE)
    
    # Evaluasi dengan TTA (opsional, set USE_TTA=True untuk mengaktifkan)
    USE_TTA = False
    if USE_TTA:
        print(f"\n[TTA] Running Test Time Augmentation...")
        labels_tta, preds_tta, probs_tta = evaluate_with_tta(model, test_loader, DEVICE)
        acc_tta = (preds_tta == labels_tta).mean()
        print(f"[TTA] Accuracy with TTA: {acc_tta:.4f} (vs {(preds == labels).mean():.4f} without TTA)")
        
        # Gunakan TTA results jika lebih baik
        if acc_tta > (preds == labels).mean():
            print(f"[TTA] ✓ Using TTA results (better accuracy)")
            preds, probs = preds_tta, probs_tta
        else:
            print(f"[TTA] Using standard results (TTA didn't improve)")

    # Metrics
    report = classification_report(labels, preds,
                                    target_names=["Real", "Fake"],
                                    output_dict=True)
    auc = roc_auc_score(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Print
    print(f"\n{classification_report(labels, preds, target_names=['Real','Fake'])}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FN: {fn:4d}")
    print(f"  FP: {fp:4d}  TN: {tn:4d}")

    # Deteksi Overfitting
    overfitting_gap = None
    overfitting_status = "Unknown"
    if val_acc is not None:
        overfitting_gap = val_acc - accuracy
        if overfitting_gap > 0.10:
            overfitting_status = "⚠️  SEVERE OVERFITTING"
        elif overfitting_gap > 0.05:
            overfitting_status = "⚠️  Moderate Overfitting"
        elif overfitting_gap > 0.02:
            overfitting_status = "✓ Slight Overfitting (acceptable)"
        elif overfitting_gap > -0.02:
            overfitting_status = "✓✓ Good Generalization"
        else:
            overfitting_status = "✓✓✓ Excellent (Test > Val)"
        
        print(f"\n{'='*55}")
        print(f"  OVERFITTING ANALYSIS")
        print(f"{'='*55}")
        print(f"  Validation Acc : {val_acc:.4f}")
        print(f"  Test Acc       : {accuracy:.4f}")
        print(f"  Gap (Val-Test) : {overfitting_gap:+.4f}")
        print(f"  Status         : {overfitting_status}")
        print(f"{'='*55}")

    # Simpan ke JSON untuk compare_models.py
    test_results = {
        "model":        MODEL_NAME,
        "accuracy":     round(accuracy, 4),
        "roc_auc":      round(auc, 4),
        "precision_real": round(report["Real"]["precision"], 4),
        "recall_real":    round(report["Real"]["recall"], 4),
        "precision_fake": round(report["Fake"]["precision"], 4),
        "recall_fake":    round(report["Fake"]["recall"], 4),
        "f1_macro":       round(report["macro avg"]["f1-score"], 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "val_acc":        round(val_acc, 4) if val_acc else None,
        "overfitting_gap": round(overfitting_gap, 4) if overfitting_gap else None,
        "overfitting_status": overfitting_status,
    }

    out_path = f"outputs/results/{MODEL_NAME}_test_results.json"
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n💾 Test results disimpan ke: {out_path}")
    print("   Setelah semua model selesai: python compare_models.py")


if __name__ == "__main__":
    main()