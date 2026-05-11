import torch
import torch.nn as nn
from tqdm import tqdm
from src.model import unfreeze_backbone


def train_model(model, train_loader, val_loader, device, epochs, lr, save_path):

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    PHASE1_EPOCHS = 20
    PHASE2_LR     = 5e-6

    def make_optimizer(model, learning_rate):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=5e-4
        )

    optimizer = make_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-7
    )

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience_counter = 0
    early_stop_patience = 15
    phase = 1

    for epoch in range(epochs):

        if epoch == PHASE1_EPOCHS and phase == 1:
            print(f"\n{'='*50}")
            print("🔓 PHASE 2: Unfreeze backbone — fine-tuning seluruh jaringan")
            print(f"{'='*50}")
            unfreeze_backbone(model)
            optimizer = make_optimizer(model, PHASE2_LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-8
            )
            patience_counter = 0
            phase = 2

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"  Ep{epoch+1:02d} P{phase} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds       = torch.sigmoid(outputs) > 0.5
            correct    += (preds == labels.bool()).sum().item()
            total      += labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = correct / total

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  Ep{epoch+1:02d} P{phase} [Val]  ", leave=False):
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs  = model(images)
                loss     = criterion(outputs, labels)
                val_loss += loss.item()

                preds    = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels.bool()).sum().item()
                total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total
        scheduler.step(val_loss)

        print(f"  Ep{epoch+1:02d} P{phase} | "
              f"Train {train_loss:.4f}/{train_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ Saved (val_loss={best_val_loss:.4f} | val_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if phase == 2 and patience_counter >= early_stop_patience:
            print("  ⚠ Early stopping triggered!")
            break

    return best_val_loss, best_val_acc