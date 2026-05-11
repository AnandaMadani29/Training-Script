# 📦 Git Workflow - Training Script

## ✅ File `.gitignore` Sudah Dibuat

File `.gitignore` sudah dikonfigurasi untuk **TIDAK PUSH**:
- ✅ Dataset (folder `Dataset/`, `data/`, dll)
- ✅ Model weights (`.pth`, `.pt`, `.onnx`)
- ✅ Outputs (folder `outputs/`)
- ✅ Python cache (`__pycache__/`)
- ✅ Virtual environment (`venv/`, `env/`)
- ✅ OS files (`.DS_Store`)

---

## 🚀 Cara Push ke Git (Tanpa Dataset)

### 1. **Cek Status**
```bash
cd /Users/adindamadani/Downloads/Training\ Script
git status
```

### 2. **Add Files (Dataset Otomatis Diabaikan)**
```bash
# Add semua file kecuali yang ada di .gitignore
git add .

# Atau add file spesifik:
git add src/
git add main_train_kfold.py
git add main_test.py
git add requirements.txt
git add .gitignore
git add ANTI_OVERFITTING_GUIDE.md
git add check_overfitting.py
```

### 3. **Commit**
```bash
git commit -m "Optimasi training script untuk accuracy >90% dan anti-overfitting"
```

### 4. **Push ke Remote**
```bash
git push origin main
```

---

## 🔍 Verifikasi Dataset TIDAK Ter-push

### Sebelum Push:
```bash
# Cek file yang akan di-commit
git status

# Pastikan Dataset/ TIDAK muncul di list
# Jika muncul, berarti .gitignore belum bekerja
```

### Jika Dataset Sudah Ter-commit Sebelumnya:
```bash
# Hapus dari git tracking (file lokal tetap ada)
git rm -r --cached Dataset/
git rm -r --cached outputs/
git rm --cached *.pth

# Commit perubahan
git commit -m "Remove dataset dan model weights dari git tracking"

# Push
git push origin main
```

---

## 📋 File yang AKAN Di-push

✅ **Source Code:**
- `src/config.py`
- `src/dataset.py`
- `src/model.py`
- `src/train.py`
- `src/evaluate.py`
- `src/utils.py`

✅ **Main Scripts:**
- `main_train_kfold.py`
- `main_test.py`
- `compare_models.py`
- `export_onnx.py`
- `check_overfitting.py`

✅ **Documentation:**
- `README.md` (jika ada)
- `requirements.txt`
- `ANTI_OVERFITTING_GUIDE.md`
- `GIT_WORKFLOW.md`
- `.gitignore`

---

## 🚫 File yang TIDAK AKAN Di-push

❌ **Dataset:**
- `Dataset/Train/`
- `Dataset/Validation/`
- `Dataset/Test/`

❌ **Model Weights:**
- `outputs/models/*.pth`
- `*.pt`, `*.onnx`

❌ **Results:**
- `outputs/results/*.json`

❌ **Cache & Temp:**
- `__pycache__/`
- `.DS_Store`
- `*.pyc`

---

## 💡 Best Practices

### 1. **Jangan Commit File Besar**
- Dataset biasanya >100MB → Gunakan Git LFS atau simpan terpisah
- Model weights → Simpan di cloud (Google Drive, S3, dll)

### 2. **Commit Message yang Baik**
```bash
# ✅ Good
git commit -m "Add overfitting detection in test script"
git commit -m "Increase data augmentation for small dataset"
git commit -m "Fix: Update dropout rate to 0.6 for better regularization"

# ❌ Bad
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

### 3. **Branch Strategy**
```bash
# Buat branch untuk eksperimen
git checkout -b experiment/higher-dropout
# ... lakukan perubahan ...
git add .
git commit -m "Experiment: Increase dropout to 0.7"
git push origin experiment/higher-dropout

# Jika berhasil, merge ke main
git checkout main
git merge experiment/higher-dropout
git push origin main
```

---

## 🔧 Troubleshooting

### Problem: Dataset Ter-push (Repo Jadi Besar)

**Solusi 1: Remove dari Git History**
```bash
# Install git-filter-repo (jika belum)
brew install git-filter-repo  # macOS

# Remove Dataset dari history
git filter-repo --path Dataset --invert-paths

# Force push (HATI-HATI!)
git push origin main --force
```

**Solusi 2: Gunakan Git LFS (untuk file >100MB)**
```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track file besar
git lfs track "*.pth"
git lfs track "Dataset/**"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Problem: .gitignore Tidak Bekerja

**Penyebab:** File sudah ter-commit sebelum .gitignore dibuat

**Solusi:**
```bash
# Clear git cache
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules"
```

---

## 📊 Cek Ukuran Repository

```bash
# Cek ukuran repo
du -sh .git

# Cek file terbesar di repo
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort --numeric-sort --key=2 | \
  tail -10
```

---

## 🎯 Checklist Sebelum Push

- [ ] `git status` - Cek file yang akan di-commit
- [ ] Dataset/ **TIDAK** muncul di list
- [ ] outputs/ **TIDAK** muncul di list
- [ ] *.pth **TIDAK** muncul di list
- [ ] .gitignore sudah di-commit
- [ ] Commit message jelas dan deskriptif
- [ ] Test script masih jalan setelah perubahan

---

## 📝 Quick Commands

```bash
# Status
git status

# Add & Commit
git add .
git commit -m "Your message here"

# Push
git push origin main

# Pull (update dari remote)
git pull origin main

# Cek remote
git remote -v

# Cek branch
git branch -a
```

---

**Last Updated:** May 11, 2026  
**Author:** Cascade AI Assistant
