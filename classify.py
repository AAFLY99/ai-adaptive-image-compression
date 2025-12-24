import os
import cv2
import numpy as np
import pytesseract
import shutil
import sys

# ======================================================
# Configure Tesseract OCR path
# ======================================================
pytesseract.pytesseract.tesseract_cmd = r""


# ======================================================
# Validate Tesseract installation
# ======================================================
if shutil.which("tesseract") is None and not pytesseract.pytesseract.tesseract_cmd:
    print("ERROR: Tesseract OCR is not installed or not found in PATH.")
    sys.exit(1)


# ======================================================
# Directory configuration
# ======================================================
SRC_DIR = "dataset"
BASE_DST = "processed"
IMP_DIR = os.path.join(BASE_DST, "imp")
NOT_IMP_DIR = os.path.join(BASE_DST, "not_imp")

if not os.path.exists(SRC_DIR):
    print(f"ERROR: Source directory '{SRC_DIR}' does not exist.")
    sys.exit(1)

os.makedirs(IMP_DIR, exist_ok=True)
os.makedirs(NOT_IMP_DIR, exist_ok=True)


# ======================================================
# Detect text regions using OCR
# ======================================================
def detect_text_mask(img):
    """
    Detects text regions using Tesseract OCR and
    returns a binary mask of detected text areas.
    """
    if img is None:
        return None

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        print(f"OCR failed: {e}")
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)

    for i, text in enumerate(data.get("text", [])):
        if text.strip():
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    return mask


# ======================================================
# Compute ratio of text pixels
# ======================================================
def compute_text_ratio(mask):
    """
    Returns ratio of text pixels to total image pixels.
    """
    if mask is None or mask.size == 0:
        return 0.0

    return np.count_nonzero(mask) / mask.size


# ======================================================
# Preserve text and blur background
# ======================================================
def preserve_text_reduce_others(img):
    """
    Preserves text regions and blurs background.
    """
    mask = detect_text_mask(img)
    if mask is None:
        return img, None

    soft_mask = cv2.GaussianBlur(mask, (7, 7), 0)
    blurred = cv2.GaussianBlur(img, (25, 25), 0)

    mask_f = soft_mask.astype(np.float32) / 255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])

    result = (img * mask_f + blurred * (1 - mask_f)).astype(np.uint8)
    return result, mask


# ======================================================
# Main processing loop
# ======================================================
TEXT_THRESHOLD = 0.01

images = [
    f for f in os.listdir(SRC_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    print("ERROR: No valid images found in dataset directory.")
    sys.exit(1)

for img_name in images:
    src_path = os.path.join(SRC_DIR, img_name)
    img = cv2.imread(src_path)

    if img is None:
        print(f"WARNING: Failed to read image '{img_name}'. Skipping.")
        continue

    processed_img, text_mask = preserve_text_reduce_others(img)
    text_ratio = compute_text_ratio(text_mask)

    if text_ratio >= TEXT_THRESHOLD:
        dst_path = os.path.join(IMP_DIR, img_name)
        label = "IMPORTANT (TEXT)"
    else:
        dst_path = os.path.join(NOT_IMP_DIR, img_name)
        label = "NOT IMPORTANT"

    if not cv2.imwrite(dst_path, processed_img):
        print(f"WARNING: Failed to save image '{img_name}'.")
        continue

    print(f"{img_name} -> {label} | text_ratio={text_ratio:.3f}")
