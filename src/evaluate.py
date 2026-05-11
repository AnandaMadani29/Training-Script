import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def evaluate_model(model, loader, device):
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

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    print("\n" + "="*45)
    print("          PERFORMANCE REPORT")
    print("="*45)
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC Score : {auc:.4f}")

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TP (Fake → Fake) : {tp:4d}   FN (Fake → Real) : {fn:4d}")
    print(f"  TN (Real → Real) : {tn:4d}   FP (Real → Fake) : {fp:4d}")
    print("="*45)

    return all_labels, all_preds, all_probs