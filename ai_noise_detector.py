"""
🔍 AI Noise Detector - Detecting AI-Generated Images
=====================================================================

This module analyzes image noise (PRNU) to detect whether images are AI-generated.

Features:
- Extract PRNU fingerprint from images
- Calculate noise statistics (Mean, Std, Energy)
- Identify AI-generated vs real images
"""

import cv2
import numpy as np
from PIL import Image
import requests
import io
import sys
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_image(src: str) -> Optional[np.ndarray]:
    """
    Load image from local path or URL.
    
    Args:
        src (str): Image path or URL
        
    Returns:
        np.ndarray: Loaded image in BGR format, or None if loading failed
        
    Raises:
        ConnectionError: If image loading from internet fails
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
            logger.error("Failed to load image - Check path or URL")
            return None
            
        logger.info(f"Image loaded successfully - Size: {img.shape}")
        return img
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading image from URL: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}")
        return None


def extract_noise(img: np.ndarray) -> np.ndarray:
    """
    Extract PRNU (Photo Response Non-Uniformity) fingerprint from image.
    
    This function works by:
    1. Converting image to grayscale
    2. Applying NLM Denoising filter
    3. Calculate difference between original and denoised
    
    Args:
        img (np.ndarray): Input image in BGR format
        
    Returns:
        np.ndarray: Noise matrix (PRNU Residual)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use fastNlMeansDenoising for efficient noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    
    logger.debug(f"Noise extracted - Shape: {noise.shape}")
    return noise


def noise_statistics(noise: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate noise statistics.
    
    Args:
        noise (np.ndarray): Noise matrix
        
    Returns:
        Tuple[float, float, float]: (Mean, Standard Deviation, Energy)
        
    Explanation:
    - Mean: Average noise values
    - Std: Standard deviation (strong indicator for AI-generated images)
    - Energy: Sum of squared noise (energy distribution)
    """
    mean = np.mean(noise)
    std = np.std(noise)
    energy = np.mean(noise ** 2)
    
    logger.debug(f"Statistics - Mean={mean:.4f}, Std={std:.4f}, Energy={energy:.4f}")
    return mean, std, energy


def analyze_noise(img: np.ndarray) -> str:
    """
    Comprehensive noise analysis to determine if image is AI-generated.
    
    Classification criteria:
    - std < 3: AI-generated image (very low noise)
    - 3 <= std < 6: Possibly AI-generated
    - std >= 6: Real image (camera noise is high)
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        str: Final result (AI or Real)
    """
    logger.info("Starting noise analysis...")
    
    noise = extract_noise(img)
    mean, std, energy = noise_statistics(noise)

    print("\n" + "="*50)
    print("📊 NOISE ANALYSIS RESULTS")
    print("="*50)
    print(f"Noise Mean:       {mean:.4f}")
    print(f"Noise Std Dev:    {std:.4f}")
    print(f"Noise Energy:     {energy:.4f}")

    # Decision engine based on Std value
    if std < 3:
        verdict = "🤖 AI-GENERATED (Very Low Noise)"
        logger.warning(f"AI-generated image detected - Std: {std:.4f}")
    elif std < 6:
        verdict = "⚠️  POSSIBLY AI-GENERATED"
        logger.info(f"Suspicious image - Std: {std:.4f}")
    else:
        verdict = "✅ REAL IMAGE (Natural Camera Noise)"
        logger.info(f"Real image detected - Std: {std:.4f}")

    print(f"\nVerdict: {verdict}")
    print("="*50 + "\n")
    
    return verdict


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python ai_noise_detector.py [image path or URL]")
        print("\nExamples:")
        print("  python ai_noise_detector.py image.jpg")
        print("  python ai_noise_detector.py https://example.com/image.png")
        return

    src = sys.argv[1]
    logger.info(f"Starting analysis: {src}")
    
    img = load_image(src)

    if img is None:
        logger.critical("Failed to load image. Exiting.")
        return

    analyze_noise(img)


if __name__ == "__main__":
    main()
