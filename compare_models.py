"""
compare_models.py — Tabel perbandingan semua model
────────────────────────────────────────────────────────────
Jalankan SETELAH semua model selesai training & testing:
    python compare_models.py

Membaca dari: outputs/results/<model>_test_results.json
Output      : tabel di terminal + outputs/results/comparison.json
────────────────────────────────────────────────────────────
"""

import json
import os
import glob

MODELS    = ["efficientnet_b0", "resnet50", "densenet121"]
MODEL_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50":        "ResNet-50",
    "densenet121":     "DenseNet-121",
}

def load_results():
    results = {}
    for model in MODELS:
        path = f"outputs/results/{model}_test_results.json"
        if os.path.exists(path):
            with open(path) as f:
                results[model] = json.load(f)
        else:
            print(f"[WARNING] Belum ada hasil untuk {model} — skip")
    return results

def load_kfold_results():
    kfold = {}
    for model in MODELS:
        path = f"outputs/results/{model}_results.json"
        if os.path.exists(path):
            with open(path) as f:
                kfold[model] = json.load(f)
    return kfold


def main():
    results = load_results()
    kfold   = load_kfold_results()

    if not results:
        print("❌ Belum ada hasil. Latih minimal satu model dulu.")
        return

    print(f"\n{'='*75}")
    print("  PERBANDINGAN MODEL — DEEPFAKE DETECTION")
    print(f"{'='*75}")

    # Header
    print(f"\n{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'F1-Macro':>10} "
          f"{'Recall Fake':>13} {'Val Acc (KFold)':>16}")
    print("-" * 85)

    comparison = []

    for model in MODELS:
        if model not in results:
            print(f"  {MODEL_LABELS[model]:<20} {'(belum selesai)':>10}")
            continue

        r = results[model]
        kf_str = ""
        if model in kfold:
            k = kfold[model]
            kf_str = f"{k['mean_val_acc']:.4f} ± {k['std_val_acc']:.4f}"

        print(f"  {MODEL_LABELS[model]:<20} "
              f"{r['accuracy']:>10.4f} "
              f"{r['roc_auc']:>10.4f} "
              f"{r['f1_macro']:>10.4f} "
              f"{r['recall_fake']:>13.4f} "
              f"{kf_str:>16}")

        comparison.append({
            "model":           MODEL_LABELS[model],
            "accuracy":        r["accuracy"],
            "roc_auc":         r["roc_auc"],
            "f1_macro":        r["f1_macro"],
            "recall_fake":     r["recall_fake"],
            "precision_fake":  r["precision_fake"],
            "recall_real":     r["recall_real"],
            "precision_real":  r["precision_real"],
            "kfold_mean_acc":  kfold[model]["mean_val_acc"] if model in kfold else None,
            "kfold_std_acc":   kfold[model]["std_val_acc"]  if model in kfold else None,
        })

    print(f"{'='*75}")

    # Tentukan pemenang jika semua model sudah selesai
    if len(comparison) == len(MODELS):
        best_acc = max(comparison, key=lambda x: x["accuracy"])
        best_auc = max(comparison, key=lambda x: x["roc_auc"])
        best_stable = min(
            [c for c in comparison if c["kfold_std_acc"] is not None],
            key=lambda x: x["kfold_std_acc"],
            default=None
        )

        print(f"\n  🏆 Akurasi Tertinggi  : {best_acc['model']} ({best_acc['accuracy']:.4f})")
        print(f"  🏆 ROC-AUC Tertinggi  : {best_auc['model']} ({best_auc['roc_auc']:.4f})")
        if best_stable:
            print(f"  🏆 Paling Stabil (std): {best_stable['model']} (±{best_stable['kfold_std_acc']:.4f})")
        print(f"\n  ➡ Rekomendasi untuk Webapp: {best_acc['model']}")

    print(f"{'='*75}\n")

    # Simpan comparison
    out_path = "outputs/results/comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"💾 Tabel perbandingan disimpan ke: {out_path}")


if __name__ == "__main__":
    main()