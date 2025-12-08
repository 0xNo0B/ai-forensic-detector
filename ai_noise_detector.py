import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys


def load_image(src):
    if src.startswith("http"):
        data = requests.get(src, timeout=10).content
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    img = cv2.imread(src)
    return img


def extract_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    return noise


def noise_statistics(noise):
    return np.mean(noise), np.std(noise), np.mean(noise ** 2)


def analyze_noise(img):
    noise = extract_noise(img)
    mean, std, energy = noise_statistics(noise)

    print("\n======== NOISE ANALYSIS ========")
    print(f"Noise Mean:   {mean:.4f}")
    print(f"Noise Std:    {std:.4f}")
    print(f"Noise Energy: {energy:.4f}")

    if std < 3:
        verdict = "AI (very low noise)"
    elif std < 6:
        verdict = "Possibly AI"
    else:
        verdict = "Real (camera noise detected)"

    print(f"\nConclusion: {verdict}")
    print("================================\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_noise_detector.py [image or URL]")
        return

    src = sys.argv[1]
    img = load_image(src)

    if img is None:
        print("Error: Could not load the image.")
        return

    analyze_noise(img)


if __name__ == "__main__":
    main()
