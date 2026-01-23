import cv2
import numpy as np
import os
import glob
import sys

def process_image(image_path):
    print(f"Processing {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    # 1. Preprocessing (Enhance, Denoise, Gray, Edge)
    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise (Gaussian Blur)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance (Histogram Equalization)
    enhanced = cv2.equalizeHist(denoised)
    
    # Edge Detection (Canny)
    edges = cv2.Canny(enhanced, 100, 200)
    
    print("Preprocessing done (Gray, Denoise, Enhance, Edge).")

    # 2. Feature Extraction (SIFT/SURF)
    # SIFT
    try:
        sift = cv2.SIFT_create()
        kp_sift, des_sift = sift.detectAndCompute(enhanced, None)
        print(f"SIFT: Found {len(kp_sift)} keypoints.")
    except AttributeError:
        print("SIFT not available (check opencv version).")
    except Exception as e:
        print(f"SIFT error: {e}")

    # SURF (Note: SURF is patented and may not be available in standard builds)
    try:
        # Check if xfeatures2d exists
        if hasattr(cv2, 'xfeatures2d'):
             surf = cv2.xfeatures2d.SURF_create(400)
             kp_surf, des_surf = surf.detectAndCompute(enhanced, None)
             print(f"SURF: Found {len(kp_surf)} keypoints.")
        else:
             print("SURF not available (xfeatures2d module missing).")
    except Exception as e:
        print(f"SURF error: {e}")

    return

if __name__ == "__main__":
    # Try to find a sample image in dataset
    dataset_root = r"d:\MachineLearning\ComputerVersion\Industrial_defect_detection-master\dataset"
    
    # Find first jpg file
    sample_images = glob.glob(os.path.join(dataset_root, "**", "*.jpg"), recursive=True)
    
    if sample_images:
        print(f"Found {len(sample_images)} images. Processing the first one.")
        process_image(sample_images[0])
    else:
        print("No images found in dataset folder.")
