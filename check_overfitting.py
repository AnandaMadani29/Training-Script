"""
check_overfitting.py — Analisis mendalam overfitting untuk semua model
────────────────────────────────────────────────────────────────────────
Membandingkan validation accuracy vs test accuracy untuk deteksi overfitting.
Jalankan setelah training dan testing selesai.
"""

import json
import os
from pathlib import Path

def analyze_overfitting():
    results_dir = Path("outputs/results")
    
    print(f"\n{'='*70}")
    print(f"  OVERFITTING ANALYSIS — ALL MODELS")
    print(f"{'='*70}\n")
    
    models = ["efficientnet_b0", "resnet50", "densenet121"]
    analysis_data = []
    
    for model_name in models:
        val_file = results_dir / f"{model_name}_results.json"
        test_file = results_dir / f"{model_name}_test_results.json"
        
        if not val_file.exists() or not test_file.exists():
            print(f"⚠️  {model_name:20s} — Missing files (skipped)")
            continue
        
        with open(val_file) as f:
            val_data = json.load(f)
        with open(test_file) as f:
            test_data = json.load(f)
        
        val_acc = val_data.get("mean_val_acc", 0)
        test_acc = test_data.get("accuracy", 0)
        gap = val_acc - test_acc
        
        # Status overfitting
        if gap > 0.10:
            status = "🔴 SEVERE OVERFITTING"
            recommendation = "Tingkatkan regularisasi (dropout, weight decay), kurangi model complexity"
        elif gap > 0.05:
            status = "🟡 Moderate Overfitting"
            recommendation = "Tambah data augmentation, pertimbangkan early stopping lebih agresif"
        elif gap > 0.02:
            status = "🟢 Slight Overfitting"
            recommendation = "Acceptable untuk dataset kecil, monitor saat production"
        elif gap > -0.02:
            status = "✅ Good Generalization"
            recommendation = "Model generalize dengan baik!"
        else:
            status = "⭐ Excellent (Test > Val)"
            recommendation = "Model sangat baik! Test set mungkin lebih mudah atau lucky split"
        
        analysis_data.append({
            "model": model_name,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": gap,
            "status": status,
            "recommendation": recommendation,
            "test_auc": test_data.get("roc_auc", 0),
            "val_std": val_data.get("std_val_acc", 0)
        })
        
        print(f"{'─'*70}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'─'*70}")
        print(f"  Validation Acc (K-Fold) : {val_acc:.4f} ± {val_data.get('std_val_acc', 0):.4f}")
        print(f"  Test Acc                : {test_acc:.4f}")
        print(f"  ROC-AUC (Test)          : {test_data.get('roc_auc', 0):.4f}")
        print(f"  Gap (Val - Test)        : {gap:+.4f}")
        print(f"  Status                  : {status}")
        print(f"  Recommendation          : {recommendation}")
        print()
    
    # Summary
    if analysis_data:
        print(f"{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        
        # Best model by test accuracy
        best_test = max(analysis_data, key=lambda x: x["test_acc"])
        print(f"  🏆 Best Test Accuracy   : {best_test['model']:20s} ({best_test['test_acc']:.4f})")
        
        # Best generalization (smallest gap)
        best_gen = min(analysis_data, key=lambda x: abs(x["gap"]))
        print(f"  🎯 Best Generalization  : {best_gen['model']:20s} (gap: {best_gen['gap']:+.4f})")
        
        # Best AUC
        best_auc = max(analysis_data, key=lambda x: x["test_auc"])
        print(f"  📊 Best ROC-AUC         : {best_auc['model']:20s} ({best_auc['test_auc']:.4f})")
        
        print(f"{'='*70}\n")
        
        # Recommendations
        severe_overfit = [x for x in analysis_data if x["gap"] > 0.10]
        if severe_overfit:
            print(f"⚠️  WARNING: {len(severe_overfit)} model(s) with severe overfitting!")
            print(f"   Consider:")
            print(f"   1. Increase dropout rate (current: 0.4-0.6)")
            print(f"   2. Increase weight decay (current: 5e-4)")
            print(f"   3. More aggressive data augmentation")
            print(f"   4. Reduce model complexity or use smaller architecture")
            print(f"   5. Collect more training data if possible\n")
        else:
            print(f"✅ All models show acceptable generalization!\n")
    else:
        print("⚠️  No model results found. Run training and testing first.\n")


if __name__ == "__main__":
    analyze_overfitting()
