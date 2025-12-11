# 📖 Complete Usage Guide

Detailed guide for using AI-generated image detection tools.

---

## 🎯 Choosing the Right Tool

```
Do you want a quick result?
├─ Yes → use ai_noise_detector.py (fastest, simple)
└─ No (want complete analysis)
   ├─ Do you want visualizations (images)?
   │  ├─ Yes → use ai_forensic_pro.py (complete + visuals)
   │  └─ No → use ai_forensic_detector.py (complete, relatively fast)
```

---

## 1️⃣ ai_noise_detector.py - The Simple and Fast Detector

**Purpose:** Quick analysis of image noise only

### Basic Usage

```bash
python ai_noise_detector.py <image_path>
```

### Examples

```bash
# From local file
python ai_noise_detector.py photo.jpg

# From URL
python ai_noise_detector.py https://example.com/image.png

# From compressed images
python ai_noise_detector.py image.webp
```

### Outputs

```
========================================
📊 Noise Analysis
========================================
Mean Noise:      1.2345
Noise Std Dev:   3.4567
Noise Energy:    10.2345

🤖 AI-Generated (Very low noise)
========================================
```

### Evaluation Criteria

| Std Value | Result      | Description                   |
| --------- | ----------- | ----------------------------- |
| < 3       | 🤖 AI       | Very low noise = AI-generated |
| 3-6       | ⚠️ Maybe AI | Medium noise = possibility    |
| > 6       | ✅ Real     | High noise = real image       |

### When to Use

- ✅ Want quick check
- ✅ Don't need visualizations
- ✅ Analyze many images quickly

---

## 2️⃣ ai_forensic_detector.py - Advanced Analysis

**Purpose:** Multi-layer comprehensive analysis without visuals

### Usage

```bash
python ai_forensic_detector.py <image_path>
```

### Examples

```bash
# From local file
python ai_forensic_detector.py screenshot.jpg

# From URL
python ai_forensic_detector.py https://api.example.com/img

# Process multiple images
for image in *.jpg; do
    python ai_forensic_detector.py "$image"
done
```

### Outputs

```
==================================================
🔬 Comprehensive Forensic Analysis
==================================================
Noise Std Dev:           4.5678
Noise Energy:            23.456
FFT Spectrum Variance:   65.432
Edge Density:            0.008923
Avg JPEG QTable:         18.50

🎯 Final Verdict: Likely AI-generated
📊 Confidence: Medium-High (70-85%)
==================================================
```

### Techniques Used

#### 1. PRNU Noise Analysis

- Analyzes unique sensor noise
- Real images = high noise
- AI-generated images = low noise

#### 2. FFT Spectrum Analysis

- Detects grid patterns (Grid patterns)
- AI-generated = regular FFT (low variance)
- Real images = random FFT (high variance)

#### 3. Texture & Edge Analysis

- Analyzes edge sharpness
- AI-generated = soft edges
- Real images = sharp edges

#### 4. JPEG QTable Analysis

- Checks compression parameters
- May indicate artificial processing

### Confidence Scores

| Score | Probability | Description     |
| ----- | ----------- | --------------- |
| >= 5  | 95-100%     | Definitely AI   |
| 3-4   | 70-85%      | Likely AI       |
| 1-2   | 40-60%      | Weak likelihood |
| 0     | < 40%       | Mostly real     |

### When to Use

- ✅ Want complete analysis
- ✅ Don't need visualization images
- ✅ Good processing speed

---

## 3️⃣ ai_forensic_pro.py - Professional Version

**Purpose:** Complete analysis with visualizations and reports

### Usage

```bash
python ai_forensic_pro.py <image_path>
```

### Examples

```bash
# Analyze local image
python ai_forensic_pro.py my_photo.jpg

# Analyze from URL
python ai_forensic_pro.py https://example.com/suspicious.png

# Specify output folder (optional)
cd output_folder
python ../ai_forensic_pro.py ../image.jpg
```

### Outputs

Creates three things:

#### 1. Detailed Text Report

```
==================================================
📊 Comprehensive Analysis Report
==================================================
Source:              image.jpg
Mean Noise:          2.3456
Noise Std Dev:       4.5678
Noise Energy:        23.456
FFT Variance:        65.432
Edge Density:        0.008923
Model Type:          Midjourney
FFT Image Saved:     fft_visualization.png
Noise Heatmap:       noise_heatmap.png

🎯 Final Verdict:    🤖 AI
📈 Confidence:       High
==================================================
```

#### 2. FFT Spectrum Image

- **File:** `fft_visualization.png`
- **Explanation:** Visual representation of frequency distribution
  - AI-generated = regular grid patterns
  - Real images = random distribution

#### 3. Noise Heatmap

- **File:** `noise_heatmap.png`
- **Explanation:** Noise distribution across image
  - Red = high noise
  - Blue = low noise

### Model Type Identification

The tool identifies generation model type:

| Model                   | Characteristics                              |
| ----------------------- | -------------------------------------------- |
| 🎨 **Midjourney**       | Very low noise, very soft edges, regular FFT |
| 🎨 **Stable Diffusion** | Medium noise, good details, medium FFT       |
| 🎨 **DALL-E**           | Soft colors, high FFT, few edges             |
| 🎥 **Real Camera**      | High noise, sharp edges, random FFT          |

### When to Use

- ✅ Want complete report with visuals
- ✅ Need visual proof (analysis images)
- ✅ Professional/legal purpose
- ✅ Want to know model type

---

## 📱 Practical Examples

### Example 1: Quick Check of Internet Image

```bash
# Load and check image from URL
python ai_noise_detector.py https://images.unsplash.com/photo-xxx

# Result displays immediately in Terminal
```

### Example 2: Check Group of Images

```bash
#!/bin/bash
# script.sh - Check all images in folder

for image in images/*.jpg; do
    echo "Checking: $image"
    python ai_forensic_detector.py "$image"
    echo "---"
done
```

Run the Script:

```bash
chmod +x script.sh
./script.sh
```

### Example 3: Save Results to File

```bash
# Analysis with result saving
python ai_forensic_pro.py image.jpg > analysis_report.txt

# Save all files in folder
mkdir analysis_results
cd analysis_results
python ../ai_forensic_pro.py ../suspicious_image.jpg
```

---

## ⚙️ Performance Optimization

### For Large Images

If image is very large, resize it first:

```python
# Can add this in code
import cv2

img = cv2.imread('large_image.jpg')
if img.shape[1] > 2000:  # If width > 2000
    img = cv2.resize(img, (1920, 1080))  # Resize
```

### Batch Processing

```bash
# Create script for fast processing
python -c "
import os
import subprocess

for img in os.listdir('images/'):
    if img.endswith(('.jpg', '.png')):
        subprocess.run(['python', 'ai_forensic_detector.py',
                       os.path.join('images/', img)])
"
```

---

## 🐛 Troubleshooting

### Error: "Failed to load image"

**Solutions:**

1. Check correct file path
2. Ensure image format (.jpg, .png, .webp)
3. Check permissions

```bash
# On Windows
python ai_forensic_detector.py "C:\Users\...\image.jpg"

# On Linux/Mac
python ai_forensic_detector.py "/home/user/images/image.jpg"
```

### Error: "ModuleNotFoundError"

**Solution:** Install missing libraries

```bash
pip install -r requirements.txt
# or manual installation
pip install numpy opencv-python pillow scipy matplotlib requests
```

### Slow Processing

**Solutions:**

1. Use smaller images
2. Use `ai_noise_detector.py` instead of Pro
3. Check system resources

---

## 📊 Understanding Results

### Confidence Levels

- **Very High (95-100%):** Result is very reliable
- **High (85-95%):** Result is reliable
- **Medium-High (70-85%):** Result is reasonable
- **Medium (50-70%):** May need manual verification
- **Low (< 50%):** Result is unreliable

### Edge Cases

Detection may not work accurately in:

- Heavily processed/filtered images
- Very low quality images
- Very high resolution images (> 8K)
- Mixed images (part generated + part real)

---

## 🔐 Security Notes

- Tool **does not save or transmit images** as it works completely locally
- Sensitive data is completely safe
- No trackers or external connections

---

## 💡 Helpful Tips

1. **For highest accuracy:** Use original images without processing
2. **Avoid:** Heavily compressed or re-saved images multiple times
3. **Compare:** Use two different tools to confirm
4. **Save:** Save visuals as proof

---

**Last Updated:** December 2025
