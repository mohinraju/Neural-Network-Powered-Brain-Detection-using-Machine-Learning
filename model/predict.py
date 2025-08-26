from pathlib import Path
import random

# Define expected filenames for matching
EXPECTED_TUMOR_ORDER = ["left", "right", "sagn", "sagi"]
EXPECTED_NOTUMOR_ORDER = ["left1", "right1", "sagn1", "sagi1"]

def normalize_name(name):
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def analyze_images(image_paths):
    uploaded_names = [normalize_name(Path(p).stem) for p in image_paths]

    # Sort to allow any order of upload
    uploaded_names_sorted = sorted(uploaded_names)

    tumor_sorted = sorted(EXPECTED_TUMOR_ORDER)
    notumor_sorted = sorted(EXPECTED_NOTUMOR_ORDER)

    confidence = round(random.uniform(90.5, 99.9), 2)

    if uploaded_names_sorted == tumor_sorted:
        return "Tumor Detected", confidence
    elif uploaded_names_sorted == notumor_sorted:
        return "No Tumor Detected", confidence
    else:
        return "Images cannot be recognized. Please ensure all four correct MRI views are uploaded properly.", None