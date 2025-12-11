"""
🔬 AI Forensic Detector - Advanced AI-Generated Image Detection Tool
========================================================================

Comprehensive multi-layer image analysis using advanced forensic techniques:
1. PRNU fingerprint analysis (Photo Response Non-Uniformity)
2. FFT spectrum analysis to detect artificial patterns
3. JPEG quantization table analysis
4. Texture and edge consistency analysis

This tool provides reliable final results without relying on external AI models.
"""

import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys
from scipy.fft import fft2, fftshift
from typing import Tuple, Optional, Dict
import logging

# Configure logging
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
# 1) PRNU Fingerprint Detection (Photo Response Non-Uniformity)
# ============================================================

def extract_noise(img: np.ndarray) -> np.ndarray:
    """
    Extract PRNU fingerprint from the image.
    
    PRNU is a unique fingerprint for each camera resulting from non-uniform
    sensor response. AI-generated images typically do not contain clear PRNU.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        np.ndarray: Noise/PRNU matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    return noise


def noise_score(noise: np.ndarray) -> Tuple[float, float]:
    """
    Calculate noise score.
    
    Args:
        noise (np.ndarray): Noise matrix
        
    Returns:
        Tuple[float, float]: (Standard Deviation, Energy)
    """
    std_dev = np.std(noise)
    energy = np.mean(noise ** 2)
    return std_dev, energy


# ============================================================
# 2) FFT Spectrum Analysis (Detecting Artificial Patterns)
# ============================================================

def fft_analysis(img: np.ndarray) -> float:
    """
    Analyze Fourier spectrum to detect grid patterns in AI-generated images.
    
    Generated images typically have low variance and regular grid patterns
    in the spectrum. Real images have more random distribution.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Spectrum variance (FFT Variance)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
<<<<<<< HEAD
=======

    # GAN images "grid" FFT
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    fft_variance = np.std(magnitude)
    
    logger.debug(f"FFT Variance: {fft_variance:.4f}")
    return fft_variance


# ============================================================
# 3) JPEG Quantization Table Analysis
# ============================================================

def jpeg_quality_analysis(path: str) -> Optional[float]:
    """
    Analyze JPEG quantization tables to detect unusual processing patterns.
    
    Note: This only works on local files, not on URLs.
    
    Args:
        path (str): Path to local JPEG file
        
    Returns:
        Optional[float]: Average quantization table or None
    """
    try:
        img = Image.open(path)
        if "quantization" in img.info:
            qtables = img.info["quantization"]
            flat = []
            for table in qtables.values():
                flat.extend(table)
            avg_q = np.mean(flat)
            logger.debug(f"JPEG QTable Average: {avg_q:.2f}")
            return avg_q
    except Exception as e:
        logger.debug(f"Cannot analyze JPEG QTable: {e}")
        return None
    
    return None


# ============================================================
# 4) Texture and Edge Consistency Analysis
# ============================================================

def texture_score(img: np.ndarray) -> float:
    """
    Calculate edge density in the image.
    
    AI-generated images typically have soft and less sharp edges.
    Real images (from cameras) have sharper, more detailed edges.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        float: Edge density (0-1)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size
    
    logger.debug(f"Edge Density: {edge_density:.6f}")
    return edge_density


# ============================================================
# Comprehensive Analysis and Decision Engine
# ============================================================

def final_analysis(img: np.ndarray, path: Optional[str] = None) -> str:
    """
    Comprehensive image analysis using all techniques together.
    
    Criteria used:
    - Noise STD: Very low → AI-generated image (Std < 3)
    - FFT Variance: Very low → Artificial grid patterns
    - Texture Density: Very low → Very soft edges
    - JPEG QTable: Unusual → Artificial processing
    
    Args:
        img (np.ndarray): Input image
        path (Optional[str]): Image path (for JPEG analysis)
        
    Returns:
        str: Final verdict
    """
    logger.info("Starting comprehensive image analysis...")
    
    # Extract all metrics
    noise = extract_noise(img)
    std, energy = noise_score(noise)
    fft_var = fft_analysis(img)
    tex = texture_score(img)
    jpeg_q = jpeg_quality_analysis(path) if path and not path.startswith("http") else None

    # Print detailed results
    print("\n" + "="*50)
    print("🔬 Comprehensive Analysis Results")
    print("="*50)
    print(f"Noise Std Dev:       {std:.4f}")
    print(f"Noise Energy:        {energy:.4f}")
    print(f"FFT Variance:        {fft_var:.4f}")
    print(f"Edge Density:        {tex:.6f}")
    if jpeg_q is not None:
        print(f"JPEG QTable Average: {jpeg_q:.2f}")
    print("="*50)

<<<<<<< HEAD
    # Advanced decision engine
    ai_score = 0

    # 1️⃣ Noise criterion (high weight)
    if std < 3:
        ai_score += 2
        logger.warning("⚠️  Very low noise - strong indicator of AI generation")
    elif std < 6:
        ai_score += 1
        logger.info("⚠️  Low noise - likely AI-generated")

    # 2️⃣ FFT criterion (high weight)
=======

    ai_score = 0


    if std < 3: ai_score += 2
    elif std < 6: ai_score += 1


>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if fft_var < 60:
        ai_score += 2
        logger.warning("⚠️  Very regular FFT spectrum - indicates artificial patterns")
    elif fft_var < 80:
        ai_score += 1
        logger.info("⚠️  Regular FFT spectrum - likely AI-generated")

<<<<<<< HEAD
    # 3️⃣ Texture criterion
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if tex < 0.007:
        ai_score += 2
        logger.warning("⚠️  Very soft edges - indicator of AI generation")
    elif tex < 0.013:
        ai_score += 1
        logger.info("⚠️  Relatively soft edges - likely AI-generated")

<<<<<<< HEAD
    # 4️⃣ JPEG criterion
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if jpeg_q is not None and jpeg_q < 18:
        ai_score += 1
        logger.info("⚠️  Unusual JPEG quantization tables")

<<<<<<< HEAD
    # Final verdict
=======
>>>>>>> 9429d53dcb8dd02ed69d256c0ca594b306f402a8
    if ai_score >= 5:
        verdict = "🤖 AI-Generated (very high probability)"
        confidence = "95-100%"
    elif ai_score >= 3:
        verdict = "⚠️  Likely AI-Generated"
        confidence = "70-85%"
    else:
        verdict = "✅ Real Image (from real camera)"
        confidence = "High"

    print(f"\n🎯 Final Verdict: {verdict}")
    print(f"📊 Confidence:    {confidence}")
    print("="*50 + "\n")

    logger.info(f"Final verdict: {verdict}")
    return verdict


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Main entry point for command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python ai_forensic_detector.py [image path or URL]")
        print("\nExamples:")
        print("  python ai_forensic_detector.py image.jpg")
        print("  python ai_forensic_detector.py https://example.com/image.png")
        return

    src = sys.argv[1]
    logger.info(f"Starting analysis: {src}")
    
    img = load_image(src)
    if img is None:
        logger.critical("Failed to load image")
        return

    final_analysis(img, src)


if __name__ == "__main__":
    main()
