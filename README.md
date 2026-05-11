# 🎯 Deepfake Detection Training Script

Training script untuk deteksi deepfake dengan accuracy >90% pada dataset kecil (1200 gambar).

## ✨ Features

- ✅ **K-Fold Cross Validation** (5-fold) untuk evaluasi robust
- ✅ **Multiple Models**: EfficientNet-B0, ResNet50, DenseNet121
- ✅ **Anti-Overfitting**: Aggressive augmentation + regularization
- ✅ **Two-Phase Training**: Freeze backbone → Fine-tune
- ✅ **Overfitting Detection**: Automatic validation vs test comparison
- ✅ **Test Time Augmentation (TTA)**: Optional untuk hasil lebih robust

## 📊 Results

| Model | Val Acc | Test Acc | ROC-AUC | Overfitting |
|-------|---------|----------|---------|-------------|
| ResNet50 | 82.98% | **91.11%** | 0.9736 | ✅ Excellent |
| EfficientNet-B0 | TBD | TBD | TBD | TBD |
| DenseNet121 | TBD | TBD | TBD | TBD |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd Training-Script

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**⚠️ PENTING: Dataset TIDAK di-push ke Git!**

Struktur folder dataset:
```
Dataset/
├── Train/
│   ├── Real/
│   │   ├── img001.jpg
│   │   └── ...
│   └── fake/
│       ├── img001.jpg
│       └── ...
├── Validation/
│   ├── Real/
│   └── fake/
└── Test/
    ├── Real/
    └── fake/
```

Download dataset dari: [Link dataset Anda]

### 3. Training

```bash
# Pilih model di src/config.py (line 18)
# MODEL_NAME = "efficientnet_b0"  # atau "resnet50" atau "densenet121"

# Run training
python3 main_train_kfold.py
```

Output:
- Model weights: `outputs/models/best_<model>.pth`
- Training results: `outputs/results/<model>_results.json`

### 4. Testing

```bash
python3 main_test.py
```

Output:
- Test results: `outputs/results/<model>_test_results.json`
- Overfitting analysis otomatis ditampilkan

### 5. Compare Models

```bash
python3 compare_models.py
```

### 6. Check Overfitting

```bash
python3 check_overfitting.py
```

## 📁 Project Structure

```
Training-Script/
├── src/
│   ├── config.py          # Hyperparameters & settings
│   ├── dataset.py         # Data loading & augmentation
│   ├── model.py           # Model architectures
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation metrics
│   └── utils.py           # Utility functions
├── main_train_kfold.py    # K-Fold training script
├── main_test.py           # Testing with overfitting detection
├── compare_models.py      # Compare all models
├── check_overfitting.py   # Detailed overfitting analysis
├── export_onnx.py         # Export to ONNX format
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules (DATASET EXCLUDED!)
├── ANTI_OVERFITTING_GUIDE.md  # Anti-overfitting documentation
├── GIT_WORKFLOW.md        # Git workflow guide
└── README.md              # This file
```

## ⚙️ Configuration

Edit `src/config.py`:

```python
MODEL_NAME = "efficientnet_b0"  # Model selection
BATCH_SIZE = 16                 # Batch size
EPOCHS     = 60                 # Training epochs
LR         = 1e-4               # Learning rate
```

## 🛡️ Anti-Overfitting Strategies

1. **Aggressive Data Augmentation**
   - RandomResizedCrop, ShiftScaleRotate
   - Color augmentation (CLAHE, HSV)
   - Blur, Noise, Compression
   - CoarseDropout

2. **Strong Regularization**
   - Dropout: 0.4-0.6 (bertingkat)
   - Weight Decay: 5e-4
   - BatchNorm di setiap layer

3. **Two-Phase Training**
   - Phase 1 (20 epochs): Freeze backbone, train classifier
   - Phase 2 (40 epochs): Fine-tune seluruh network

4. **Early Stopping**
   - Patience: 15 epochs
   - Monitor: validation loss

Lihat `ANTI_OVERFITTING_GUIDE.md` untuk detail lengkap.

## 📦 Git Workflow

**⚠️ Dataset dan model weights TIDAK di-push ke Git!**

```bash
# Add files (dataset otomatis diabaikan)
git add .

# Commit
git commit -m "Your commit message"

# Push
git push origin main
```

File yang **TIDAK** akan di-push (sudah di `.gitignore`):
- ❌ `Dataset/` (folder dataset)
- ❌ `outputs/` (model weights & results)
- ❌ `*.pth`, `*.pt`, `*.onnx` (model files)
- ❌ `__pycache__/` (Python cache)

Lihat `GIT_WORKFLOW.md` untuk panduan lengkap.

## 🔧 Advanced Usage

### Test Time Augmentation (TTA)

Edit `main_test.py` line 91:
```python
USE_TTA = True  # Meningkatkan accuracy ~0.5-2%
```

### Export to ONNX

```bash
python3 export_onnx.py
```

### Custom Augmentation

Edit `src/dataset.py` function `get_transforms()`.

## 📊 Monitoring Training

Tanda-tanda **Good Training**:
- ✅ Train & Val accuracy naik bersamaan
- ✅ Gap Train-Val < 5%
- ✅ Val loss turun konsisten

Tanda-tanda **Overfitting**:
- ⚠️ Train accuracy >> Val accuracy (gap >10%)
- ⚠️ Val loss mulai naik
- ⚠️ Train loss terus turun tapi val loss stagnan

## 🐛 Troubleshooting

### Accuracy < 90%
1. Cek distribusi dataset (Real vs Fake balanced?)
2. Cek kualitas gambar (corrupt images?)
3. Coba model lebih kecil (EfficientNet-B0)
4. Tingkatkan augmentasi

### Overfitting (Gap > 5%)
1. Tingkatkan dropout (0.7, 0.6, 0.5)
2. Tingkatkan weight decay (1e-3)
3. Tambah augmentasi
4. Gunakan model lebih kecil

### Out of Memory
1. Turunkan batch size (16 → 8)
2. Gunakan model lebih kecil
3. Kurangi image size (224 → 192)

## 📚 Documentation

- `ANTI_OVERFITTING_GUIDE.md` - Panduan anti-overfitting lengkap
- `GIT_WORKFLOW.md` - Git workflow & best practices

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

[Your License Here]

## 👤 Author

[Your Name]

## 🙏 Acknowledgments

- Pre-trained models from [timm](https://github.com/rwightman/pytorch-image-models)
- Augmentation from [Albumentations](https://albumentations.ai/)

---

**Last Updated:** May 11, 2026
