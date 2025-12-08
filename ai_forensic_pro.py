import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import os


# -----------------------------------------------------
# Load Image (URL or local)
# -----------------------------------------------------
def load_image(src):
    if src.startswith("http"):
        data = requests.get(src, timeout=10).content
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(src)


# -----------------------------------------------------
# FFT Visualization
# -----------------------------------------------------
def save_fft_visualization(img, output_path="fft_visualization.png"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap="gray")
    plt.title("FFT Frequency Spectrum")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return output_path


# -----------------------------------------------------
# Noise (PRNU) extraction + heatmap
# -----------------------------------------------------
def extract_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    return noise


def save_noise_heatmap(noise, output_path="noise_heatmap.png"):
    # Normal noise for visualization
    norm_noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(norm_noise, cmap="jet")
    plt.title("Noise Heatmap (PRNU Residual)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return output_path


def noise_statistics(noise):
    mean = np.mean(noise)
    std = np.std(noise)
    energy = np.mean(noise ** 2)
    return mean, std, energy


# -----------------------------------------------------
# FFT Analysis
# -----------------------------------------------------
def fft_analysis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    fft_variance = np.std(magnitude)
    return fft_variance


# -----------------------------------------------------
# Edge / Texture analysis
# -----------------------------------------------------
def texture_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = np.sum(edges) / edges.size
    return edge_density


# -----------------------------------------------------
# Model Identifier (MJ / SD / DALL-E / REAL)
# -----------------------------------------------------
def identify_model(std_noise, fft_var, tex):

    if std_noise < 2.5 and fft_var < 70 and tex < 0.008:
        return "Midjourney"


    if 2.5 <= std_noise <= 4.5 and 70 <= fft_var <= 85:
        return "Stable Diffusion"

    if std_noise < 3.5 and fft_var > 85 and tex < 0.010:
        return "DALL-E"


    if std_noise > 6 and tex > 0.012:
        return "Real Camera"

    return "Unknown AI / Mixed"


# -----------------------------------------------------
# Final Forensic AI Analysis + Binary Verdict
# -----------------------------------------------------
def analyze_image(img, src):

    noise = extract_noise(img)
    mean_noise, std_noise, energy_noise = noise_statistics(noise)


    fft_var = fft_analysis(img)


    tex = texture_score(img)


    fft_path = save_fft_visualization(img)
    heatmap_path = save_noise_heatmap(noise)


    model_type = identify_model(std_noise, fft_var, tex)


    ai_score = 0

    if std_noise < 3:
        ai_score += 2
    elif std_noise < 6:
        ai_score += 1

    if fft_var < 60:
        ai_score += 2
    elif fft_var < 80:
        ai_score += 1

    if tex < 0.008:
        ai_score += 2
    elif tex < 0.013:
        ai_score += 1

    if ai_score >= 4:
        binary_verdict = "AI"
    else:
        binary_verdict = "Not AI"

    # ----- Print Results -----
    print("\n========== FORENSIC ANALYSIS ==========")
    print(f"Source:            {src}")
    print(f"Noise Mean:        {mean_noise:.4f}")
    print(f"Noise STD:         {std_noise:.4f}")
    print(f"Noise Energy:      {energy_noise:.4f}")
    print(f"FFT Variance:      {fft_var:.4f}")
    print(f"Texture Density:   {tex:.4f}")
    print(f"Model Signature:   {model_type}")
    print(f"FFT Image Saved:   {fft_path}")
    print(f"Noise Heatmap:     {heatmap_path}")
    print("----------------------------------------")
    print(f"Final Verdict (Binary): {binary_verdict}")
    print("========================================\n")

    return binary_verdict, model_type


# -----------------------------------------------------
# CLI
# -----------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_forensic_pro.py [image file or URL]")
        return

    src = sys.argv[1]
    img = load_image(src)

    if img is None:
        print("Error: Could not load image. Check path or URL.")
        return

    analyze_image(img, src)


if __name__ == "__main__":
    main()
