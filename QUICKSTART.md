# 🚀 Quick Start Guide

## 📝 Installation and Setup Steps

### 1️⃣ Install Libraries

```bash
pip install -r requirements.txt
```

Or manual installation:

```bash
pip install numpy opencv-python pillow scipy matplotlib requests
```

### 2️⃣ Immediate Usage

#### Quick Check (Noise Detection Only)

```bash
python ai_noise_detector.py image.jpg
```

#### Full Analysis

```bash
python ai_forensic_detector.py image.jpg
```

#### Professional Check (Professional + Visuals)

```bash
python ai_forensic_pro.py image.jpg
```

---

## 🌐 Loading from Internet

All tools support URL links:

```bash
python ai_forensic_pro.py https://example.com/image.png
```

---

## 📊 Understanding Results

### Example Results:

```
🤖 AI-Generated → Std < 3
⚠️  Likely AI → Std between 3-6
✅ Real Image → Std > 6
```

---

## 📚 Complete Documentation

| File                      | Description         |
| ------------------------- | ------------------- |
| `README.md`               | Comprehensive docs  |
| `USAGE_GUIDE.md`          | Detailed usage      |
| `CHANGELOG.md`            | Change history      |
| `IMPROVEMENTS_SUMMARY.md` | Improvements detail |

---

## 🆘 Troubleshooting Common Issues

### Error: "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

### Error: "Failed to load image"

- Check file path
- Ensure image format (.jpg, .png)

### Slow Processing

- Use `ai_noise_detector.py` instead of Pro version
- Use smaller images

---

**For more information, read USAGE_GUIDE.md** 📖

---

**Last Updated:** December 2025
