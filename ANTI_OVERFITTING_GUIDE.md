# 🛡️ Anti-Overfitting Guide untuk Dataset Kecil (1200 gambar)

## ✅ Perubahan yang Sudah Diterapkan

### 1. **Hyperparameter Optimization** (`src/config.py`)
```python
BATCH_SIZE = 16        # ↓ dari 32 (lebih stabil untuk dataset kecil)
EPOCHS     = 60        # ↑ dari 40 (lebih banyak iterasi)
LR         = 1e-4      # ↓ dari 5e-4 (learning rate lebih konservatif)
```

### 2. **Aggressive Data Augmentation** (`src/dataset.py`)
- ✅ RandomResizedCrop (scale 0.8-1.0)
- ✅ ShiftScaleRotate (rotate ±25°, shift 10%, scale 15%)
- ✅ Color augmentation: CLAHE, HSV, Brightness/Contrast
- ✅ Noise & Blur: Gaussian, Motion Blur
- ✅ CoarseDropout (1-8 holes, 8-32px)
- ✅ Image Compression (50-95% quality)

### 3. **Model Regularization** (`src/model.py`)
```python
# Classifier head lebih dalam dengan dropout bertingkat:
512 → Dropout(0.6) → 256 → Dropout(0.5) → 128 → Dropout(0.4) → 1
```

### 4. **Training Strategy** (`src/train.py`)
- ✅ **Phase 1**: 20 epochs (↑ dari 10) — latih classifier head
- ✅ **Phase 2**: Fine-tune dengan LR 5e-6 (↓ dari 1e-5)
- ✅ **Weight Decay**: 5e-4 (↑ dari 1e-4)
- ✅ **Scheduler Patience**: 5 epochs (↑ dari 3)
- ✅ **Early Stopping**: 15 epochs (↑ dari 10)

### 5. **Overfitting Detection** (`main_test.py`)
Script testing sekarang otomatis mendeteksi overfitting dengan membandingkan:
- Validation Accuracy (K-Fold mean)
- Test Accuracy

**Status Overfitting:**
- Gap > 10%: 🔴 **SEVERE OVERFITTING**
- Gap 5-10%: 🟡 **Moderate Overfitting**
- Gap 2-5%: 🟢 **Slight Overfitting** (acceptable)
- Gap ±2%: ✅ **Good Generalization**
- Test > Val: ⭐ **Excellent**

---

## 📊 Hasil dengan Konfigurasi Lama vs Baru

### **SEBELUM (Konfigurasi Lama)**
```
Model: ResNet50
Validation Acc: 82.98% ± 2.21%
Test Acc: 91.11%
Gap: -8.13% (Test > Val — Excellent!)
ROC-AUC: 0.9736
```
✅ **Tidak ada overfitting!** Test accuracy bahkan lebih tinggi dari validation.

### **TARGET dengan Konfigurasi Baru**
Dengan perubahan yang sudah diterapkan, target:
- **Validation Acc: >90%**
- **Test Acc: >90%**
- **Gap: <5%** (Good generalization)

---

## 🚀 Cara Menggunakan

### 1. **Training dengan Konfigurasi Baru**
```bash
cd /Users/adindamadani/Downloads/Training\ Script
python3 main_train_kfold.py
```

### 2. **Testing & Deteksi Overfitting**
```bash
python3 main_test.py
```
Output akan menampilkan:
```
=======================================================
  OVERFITTING ANALYSIS
=======================================================
  Validation Acc : 0.9012
  Test Acc       : 0.8945
  Gap (Val-Test) : +0.0067
  Status         : ✓✓ Good Generalization
=======================================================
```

### 3. **Analisis Semua Model**
```bash
python3 check_overfitting.py
```
Membandingkan overfitting untuk semua model (EfficientNet, ResNet50, DenseNet121).

### 4. **Test Time Augmentation (TTA)** - Opsional
Untuk hasil lebih robust, aktifkan TTA di `main_test.py`:
```python
USE_TTA = True  # Line 91
```
TTA akan:
- Prediksi dengan original + horizontal flip
- Average hasilnya untuk mengurangi variance
- Biasanya meningkatkan accuracy 0.5-2%

---

## 🔧 Troubleshooting Overfitting

### Jika Masih Terjadi Overfitting (Gap > 5%)

#### **Opsi 1: Tingkatkan Regularisasi**
Edit `src/model.py`:
```python
# Tingkatkan dropout
nn.Dropout(0.7),  # dari 0.6
nn.Dropout(0.6),  # dari 0.5
nn.Dropout(0.5),  # dari 0.4
```

Edit `src/train.py`:
```python
weight_decay=1e-3  # dari 5e-4
```

#### **Opsi 2: Ganti ke Model Lebih Kecil**
Edit `src/config.py`:
```python
MODEL_NAME = "efficientnet_b0"  # ~4.3M params (lebih cocok untuk 1200 gambar)
# Daripada resnet50 (~23M params)
```

#### **Opsi 3: Tambah Augmentasi**
Edit `src/dataset.py`, tambahkan:
```python
A.GridDistortion(p=0.3),
A.ElasticTransform(p=0.3),
A.RandomGamma(p=0.3),
```

#### **Opsi 4: Label Smoothing**
Edit `src/train.py`:
```python
# Ganti BCEWithLogitsLoss dengan label smoothing
# Labels: 0 → 0.05, 1 → 0.95
criterion = nn.BCEWithLogitsLoss()
# Lalu di training loop, smooth labels:
labels = labels * 0.9 + 0.05  # sebelum criterion(outputs, labels)
```

---

## 📈 Monitoring Selama Training

### **Tanda-tanda Overfitting:**
1. ✅ **Train accuracy >> Val accuracy** (gap >10%)
2. ✅ **Val loss mulai naik** setelah beberapa epoch
3. ✅ **Train loss terus turun** tapi val loss stagnan

### **Tanda-tanda Good Training:**
1. ✅ Train & Val accuracy naik bersamaan
2. ✅ Gap Train-Val < 5%
3. ✅ Val loss turun konsisten

### **Contoh Output Training yang Baik:**
```
Ep20 P1 | Train 0.1234/0.9456 | Val 0.1456/0.9123 | LR 1.0e-04
  ✔ Saved (val_loss=0.1456 | val_acc=0.9123)
```
Gap: 94.56% - 91.23% = 3.33% ✅ Good!

---

## 🎯 Best Practices untuk Dataset Kecil

1. **Gunakan model pre-trained** ✅ (sudah diterapkan)
2. **Freeze backbone dulu** ✅ (Phase 1: 20 epochs)
3. **Augmentasi agresif** ✅ (sudah diterapkan)
4. **Regularisasi kuat** ✅ (dropout 0.4-0.6, weight decay 5e-4)
5. **K-Fold Cross Validation** ✅ (5-fold)
6. **Early stopping** ✅ (patience 15)
7. **Learning rate kecil** ✅ (1e-4 → 5e-6)
8. **Monitor val_loss, bukan val_acc** untuk early stopping

---

## 📝 Checklist Sebelum Production

- [ ] Validation accuracy >90%
- [ ] Test accuracy >90%
- [ ] Overfitting gap <5%
- [ ] ROC-AUC >0.95
- [ ] Precision & Recall balanced (>85% untuk kedua kelas)
- [ ] Test dengan TTA untuk robustness
- [ ] Confusion matrix: FP dan FN <10%

---

## 🔍 File yang Sudah Dimodifikasi

1. ✅ `src/config.py` — Hyperparameters
2. ✅ `src/dataset.py` — Augmentasi
3. ✅ `src/model.py` — Architecture & regularisasi
4. ✅ `src/train.py` — Training strategy
5. ✅ `main_test.py` — Overfitting detection + TTA
6. ✅ `check_overfitting.py` — Analisis mendalam (NEW)

---

## 💡 Tips Tambahan

### Jika Accuracy Masih <90%:
1. Cek distribusi dataset (Real vs Fake harus balanced)
2. Cek kualitas gambar (corrupt images?)
3. Coba ensemble 3 model (EfficientNet + ResNet + DenseNet)
4. Pertimbangkan collect data tambahan

### Jika Test Acc > Val Acc (seperti hasil Anda):
✅ **Ini BAGUS!** Artinya:
- Model generalize dengan baik
- Test set mungkin sedikit lebih mudah
- Tidak ada overfitting sama sekali

**Rekomendasi:** Lanjutkan dengan konfigurasi ini!

---

**Last Updated:** May 11, 2026
**Author:** Cascade AI Assistant
