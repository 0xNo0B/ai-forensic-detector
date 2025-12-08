import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys
from scipy.fft import fft2, fftshift


# -----------------------------------------------------
# Load Image (URL or local)
# -----------------------------------------------------
def load_image(src):
    if src.startswith("http"):
        data = requests.get(src, timeout=10).content
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(src)


# -----------------------------------------------------
# 1) PRNU Noise Fingerprint Detection
# -----------------------------------------------------
def extract_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    return noise


def noise_score(noise):
    std_dev = np.std(noise)
    energy = np.mean(noise ** 2)
    return std_dev, energy


# -----------------------------------------------------
# 2) FFT Spectrum Analysis (GAN Detection)
# -----------------------------------------------------
def fft_analysis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # GAN images "grid" FFT
    fft_variance = np.std(magnitude)
    return fft_variance


# -----------------------------------------------------
# 3) JPEG QTable Analysis
# -----------------------------------------------------
def jpeg_quality_analysis(path):
    try:
        img = Image.open(path)
        if "quantization" in img.info:
            qtables = img.info["quantization"]
            flat = []
            for table in qtables.values():
                flat.extend(table)
            avg_q = np.mean(flat)
            return avg_q
    except:
        return None

    return None


# -----------------------------------------------------
# 4) Texture Consistency (AI images have softer edges)
# -----------------------------------------------------
def texture_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    return edge_density


# -----------------------------------------------------
# Combine All Signals
# -----------------------------------------------------
def final_analysis(img, path=None):
    noise = extract_noise(img)
    std, energy = noise_score(noise)
    fft_var = fft_analysis(img)
    tex = texture_score(img)

    jpeg_q = jpeg_quality_analysis(path) if path and not path.startswith("http") else None

    print("\n========== FORENSIC AI ANALYSIS ==========")
    print(f"Noise STD:           {std:.4f}")
    print(f"Noise Energy:        {energy:.4f}")
    print(f"FFT Variance:        {fft_var:.4f}")
    print(f"Texture Density:     {tex:.4f}")
    print(f"JPEG Avg QTable:     {jpeg_q}")
    print("==========================================")


    ai_score = 0


    if std < 3: ai_score += 2
    elif std < 6: ai_score += 1


    if fft_var < 60:
        ai_score += 2
    elif fft_var < 80:
        ai_score += 1

    if tex < 0.007:
        ai_score += 2
    elif tex < 0.013:
        ai_score += 1

    if jpeg_q is not None and jpeg_q < 18:
        ai_score += 1

    if ai_score >= 5:
        verdict = "AI (Highly likely)"
    elif ai_score >= 3:
        verdict = "Possibly AI"
    else:
        verdict = "Real"

    print(f"\nFinal Verdict: {verdict}")
    print("==========================================\n")

    return verdict


# -----------------------------------------------------
# CLI
# -----------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_forensic_detector.py [image or URL]")
        return

    src = sys.argv[1]
    img = load_image(src)

    if img is None:
        print("Error loading image.")
        return

    final_analysis(img, src)


if __name__ == "__main__":
    main()
