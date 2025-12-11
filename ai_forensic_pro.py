"""
🔬 AI Forensic Pro - Advanced AI-Generated Image Detector with Visualizations
========================================================

Professional version of AI-generated image detection tool including:
- PRNU analysis + Heatmap visualizations
- FFT spectrum analysis with charts
- Model type identification (Midjourney / Stable Diffusion / DALL-E / Real)
- Binary final verdict (AI / Real)

Outputs:
- FFT spectrum visualization image
- Noise heatmap visualization
- Detailed analysis report with all metrics
"""

import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_image(src: str) -> Optional[np.ndarray]:
    """
    Load image from local path or URL.
    
    Args:
        src (str): Image path or URL
        
    Returns:
        Optional[np.ndarray]: Image in BGR format or None
    """
    try:
        if src.startswith("http"):
            logger.info(f"Loading image from URL: {src}")
            data = requests.get(src, timeout=10).content
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            logger.info(f"Loading image from: {src}")
            img = cv2.imread(src)
        
        if img is None:
            logger.error("Failed to load image")
            return None
            
        logger.info(f"Image loaded successfully - Size: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


# ============================================================
# مرئيات FFT
# ============================================================

def save_fft_visualization(img: np.ndarray, output_path: str = "fft_visualization.png") -> str:
    """
    Save FFT spectrum visualization image.
    
    Args:
        img (np.ndarray): Input image
        output_path (str): Save path
        
    Returns:
        str: Path to saved file
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(magnitude, cmap="gray")
        plt.title("FFT Frequency Spectrum", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()
        
        logger.info(f"FFT visualization saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving FFT visualization: {e}")
        return ""


# ============================================================
# Extract Noise + Heatmap
# ============================================================

def extract_noise(img: np.ndarray) -> np.ndarray:
    """
    Extract PRNU fingerprint from the image.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        np.ndarray: Noise matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    return noise


<<<<<<< HEAD
def save_noise_heatmap(noise: np.ndarray, output_path: str = "noise_heatmap.png") -> str:
    """
    Save noise distribution heatmap (PRNU).
    
    Hot colors (red) = high noise
    Cold colors (blue) = low noise
    
    Args:
        noise (np.ndarray): Noise matrix
        output_path (str): Save path
        
    Returns:
        str: Path to saved file
    """
    try:
        # Normalize noise for visualization
        norm_noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
=======
def save_noise_heatmap(noise, output_path="noise_heatmap.png"):
    # Normal noise for visualization
    norm_noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8

        plt.figure(figsize=(8, 8), dpi=100)
        im = plt.imshow(norm_noise, cmap="jet")
        plt.title("Noise Heatmap (PRNU Residual)", fontsize=14)
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Noise Level", rotation=270, labelpad=20)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()
        
        logger.info(f"Noise heatmap saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving heatmap: {e}")
        return ""


def noise_statistics(noise: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate noise statistics.
    
    Args:
        noise (np.ndarray): Noise matrix
        
    Returns:
        Tuple[float, float, float]: (Mean, Standard Deviation, Energy)
    """
    mean = np.mean(noise)
    std = np.std(noise)
    energy = np.mean(noise ** 2)
    
    logger.debug(f"Noise Stats - Mean: {mean:.4f}, Std: {std:.4f}, Energy: {energy:.4f}")
    return mean, std, energy


# ============================================================
# FFT Spectrum Analysis
# ============================================================

def fft_analysis(img: np.ndarray) -> float:
    """
    Analyze FFT spectrum to detect artificial patterns.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Spectrum variance (FFT Variance)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    fft_variance = np.std(magnitude)
    
    logger.debug(f"FFT Variance: {fft_variance:.4f}")
    return fft_variance


# ============================================================
# Texture and Edge Analysis
# ============================================================

def texture_score(img: np.ndarray) -> float:
    """
    Calculate edge density in the image.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Edge density
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = np.sum(edges) / edges.size
    
    logger.debug(f"Edge Density: {edge_density:.6f}")
    return edge_density


<<<<<<< HEAD
# ============================================================
# Identify Generative Model Type
# ============================================================

def identify_model(std_noise: float, fft_var: float, tex: float) -> str:
    """
    Identify the type of generative model used to create the image.
    
    Model signatures used:
    - Midjourney: Very low noise, regular FFT, very soft edges
    - Stable Diffusion: Medium noise, medium FFT, more details
    - DALL-E: Soft colors, high FFT, few edges
    - Real Camera: High noise, random FFT, sharp edges
    
    Args:
        std_noise (float): Standard deviation of noise
        fft_var (float): FFT variance
        tex (float): Edge density
        
    Returns:
        str: Identified model name
    """
    logger.info(f"Analyzing model signature - Noise:{std_noise:.2f}, FFT:{fft_var:.2f}, Tex:{tex:.6f}")
    
    # --- Midjourney (very smooth) ---
=======
# -----------------------------------------------------
# Model Identifier (MJ / SD / DALL-E / REAL)
# -----------------------------------------------------
def identify_model(std_noise, fft_var, tex):

>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if std_noise < 2.5 and fft_var < 70 and tex < 0.008:
        logger.info("Model identified: Midjourney")
        return "Midjourney"

<<<<<<< HEAD
    # --- Stable Diffusion (good details) ---
=======

>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if 2.5 <= std_noise <= 4.5 and 70 <= fft_var <= 85:
        logger.info("Model identified: Stable Diffusion")
        return "Stable Diffusion"

<<<<<<< HEAD
    # --- DALL-E (soft colors) ---
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if std_noise < 3.5 and fft_var > 85 and tex < 0.010:
        logger.info("Model identified: DALL-E")
        return "DALL-E"

<<<<<<< HEAD
    # --- Real Camera (high noise) ---
=======

>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if std_noise > 6 and tex > 0.012:
        logger.info("Model identified: Real Camera")
        return "Real Camera"

    logger.info("Model identified: Unknown/Mixed")
    return "Unknown AI / Mixed"


<<<<<<< HEAD
# ============================================================
# Comprehensive Analysis and Final Verdict
# ============================================================

def analyze_image(img: np.ndarray, src: str) -> Tuple[str, str]:
    """
    Perform comprehensive image analysis with visualization output.
    
    This function:
    1. Extracts noise and calculates statistics
    2. Calculates FFT variance
    3. Calculates edge density
    4. Saves visualizations (FFT + Heatmap)
    5. Identifies generative model type
    6. Returns binary final verdict
    
    Args:
        img (np.ndarray): Input image
        src (str): Image source (path or URL)
        
    Returns:
        Tuple[str, str]: (Final verdict, Model type)
    """
    logger.info("🔬 Starting comprehensive image analysis...")
    
    # 1️⃣ Extract noise and calculate statistics
    noise = extract_noise(img)
    mean_noise, std_noise, energy_noise = noise_statistics(noise)

    # 2️⃣ FFT analysis
    fft_var = fft_analysis(img)

    # 3️⃣ Texture analysis
    tex = texture_score(img)

    # 4️⃣ Save visualizations
    fft_path = save_fft_visualization(img)
    heatmap_path = save_noise_heatmap(noise)

    # 5️⃣ Identify generative model
    model_type = identify_model(std_noise, fft_var, tex)

    # 6️⃣ Decision engine for final verdict
    ai_score = 0

    # Noise criterion
=======
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

>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if std_noise < 3:
        ai_score += 2
    elif std_noise < 6:
        ai_score += 1

<<<<<<< HEAD
    # FFT criterion
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if fft_var < 60:
        ai_score += 2
    elif fft_var < 80:
        ai_score += 1

<<<<<<< HEAD
    # Texture criterion
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if tex < 0.008:
        ai_score += 2
    elif tex < 0.013:
        ai_score += 1

<<<<<<< HEAD
    # Final verdict
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if ai_score >= 4:
        binary_verdict = "🤖 AI-Generated Image"
        confidence = "High"
    else:
        binary_verdict = "✅ Real Image"
        confidence = "High"

    # Print results
    print("\n" + "="*55)
    print("📊 Comprehensive Analysis Report")
    print("="*55)
    print(f"Source:              {src}")
    print(f"Mean Noise:          {mean_noise:.4f}")
    print(f"Noise Std Dev:       {std_noise:.4f}")
    print(f"Noise Energy:        {energy_noise:.4f}")
    print(f"FFT Variance:        {fft_var:.4f}")
    print(f"Edge Density:        {tex:.6f}")
    print(f"Model Type:          {model_type}")
    if fft_path:
        print(f"FFT Image Saved:     {fft_path}")
    if heatmap_path:
        print(f"Noise Heatmap:       {heatmap_path}")
    print("-"*55)
    print(f"🎯 Final Verdict:    {binary_verdict}")
    print(f"📈 Confidence:       {confidence}")
    print("="*55 + "\n")

    logger.info(f"Analysis complete - Result: {binary_verdict}")
    return binary_verdict, model_type


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Main entry point for command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python ai_forensic_pro.py [image path or URL]")
        print("\nExamples:")
        print("  python ai_forensic_pro.py image.jpg")
        print("  python ai_forensic_pro.py https://example.com/image.png")
        print("\nOutputs:")
        print("  - fft_visualization.png (FFT spectrum image)")
        print("  - noise_heatmap.png (Noise heatmap)")
        print("  - Detailed analysis report")
        return

    src = sys.argv[1]
    logger.info(f"Starting analysis: {src}")
    
    img = load_image(src)
    if img is None:
        logger.critical("Failed to load image. Check path or URL.")
        return

    analyze_image(img, src)


if __name__ == "__main__":
    main()
