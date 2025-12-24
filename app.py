# streamlit_app_pretty_rgba_high_low.py

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ======================================================
# Global configuration
# ======================================================
MODEL_PATH = "models/tile_classifier_ocr.keras"
TILE_SIZE = 64
CLASS_INPUT_SIZE = (128, 128)

INTERP_HIGH = cv2.INTER_LANCZOS4
INTERP_LOW = cv2.INTER_LINEAR

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================
# Model loading
# ======================================================
def load_trained_model(path=MODEL_PATH):
    """
    Loads the trained tile classifier model.
    """
    if not os.path.exists(path):
        st.error("Model file not found. Please train the model first.")
        return None

    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ======================================================
# Image tiling utilities
# ======================================================
def image_to_tiles(img, tile_size=TILE_SIZE):
    """
    Splits the image into fixed-size RGBA tiles.
    Pads border tiles if needed.
    """
    h, w = img.shape[:2]
    tiles, coords = [], []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            if th != tile_size or tw != tile_size:
                padded = np.zeros((tile_size, tile_size, 4), dtype=img.dtype)
                padded[:th, :tw] = tile
                tile = padded

            tiles.append(tile)
            coords.append((x, y))

    return tiles, coords, (w, h)


def classify_tiles(model, tiles):
    """
    Classifies tiles using the trained model.
    """
    if not tiles:
        return []

    batch = np.array(
        [cv2.resize(t, CLASS_INPUT_SIZE) / 255.0 for t in tiles],
        dtype=np.float32
    )

    try:
        preds = model.predict(batch, verbose=0).flatten()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return []

    return preds


# ======================================================
# Adaptive tile processing
# ======================================================
def adapt_tile(tile, prob, high_th, scale_high, scale_low):
    """
    Applies adaptive resolution scaling based on importance probability.
    """
    if prob >= high_th:
        scale, interp = scale_high, INTERP_HIGH
    else:
        scale, interp = scale_low, INTERP_LOW

    h, w = tile.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    small = cv2.resize(tile, (new_w, new_h), interpolation=interp)
    restored = cv2.resize(small, (w, h), interpolation=interp)
    return restored


def tiles_to_image(tiles, coords, orig_size):
    """
    Reconstructs the full image from processed tiles.
    """
    w, h = orig_size
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for tile, (x, y) in zip(tiles, coords):
        y_end = min(y + TILE_SIZE, h)
        x_end = min(x + TILE_SIZE, w)
        canvas[y:y_end, x:x_end] = tile[:y_end - y, :x_end - x]

    return canvas


# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="AI Adaptive Compression", layout="wide")
st.title("🧠 AI Adaptive Image Compression")
st.markdown(
    "Smart image compression that preserves important regions while reducing "
    "resolution in less significant areas."
)

model = load_trained_model()
if model is None:
    st.stop()


# ======================================================
# Sidebar controls
# ======================================================
st.sidebar.header("⚙️ Compression Settings")

high_th = st.sidebar.slider(
    "Importance Threshold",
    min_value=0.0, max_value=1.0, value=0.6
)

scale_high = st.sidebar.slider(
    "High-Importance Quality Scale",
    min_value=0.1, max_value=1.0, value=1.0
)

scale_low = st.sidebar.slider(
    "Low-Importance Quality Scale",
    min_value=0.1, max_value=1.0, value=0.4
)

jpeg_quality = st.sidebar.slider(
    "Final JPEG Quality",
    min_value=10, max_value=100, value=85
)


# ======================================================
# Image upload
# ======================================================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img_np = np.array(
            Image.open(uploaded_file).convert("RGBA")
        )
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    with st.spinner("Processing and classifying tiles..."):
        tiles, coords, orig_size = image_to_tiles(img_np)
        preds = classify_tiles(model, tiles)

        if len(preds) != len(tiles):
            st.error("Tile classification failed.")
            st.stop()

        tiles_processed = [
            adapt_tile(t, p, high_th, scale_high, scale_low)
            for t, p in zip(tiles, preds)
        ]

        result_img = tiles_to_image(
            tiles_processed, coords, orig_size
        )

    # ==================================================
    # Display comparison
    # ==================================================
    col1, col2 = st.columns(2)
    col1.image(img_np[..., :3], caption="Original Image")
    col2.image(result_img[..., :3], caption="Adaptive Compressed Image")

    # ==================================================
    # Save and analyze file sizes
    # ==================================================
    out_path = os.path.join(
        OUTPUT_DIR,
        f"compressed_{os.path.splitext(uploaded_file.name)[0]}.jpg"
    )

    temp_orig = os.path.join(OUTPUT_DIR, "temp_original.jpg")

    Image.fromarray(img_np[..., :3]).save(temp_orig, "JPEG", quality=100)
    Image.fromarray(result_img[..., :3]).save(
        out_path, "JPEG", quality=jpeg_quality
    )

    size_old = os.path.getsize(temp_orig) / 1024
    size_new = os.path.getsize(out_path) / 1024
    reduction = ((size_old - size_new) / size_old) * 100

    os.remove(temp_orig)

    # ==================================================
    # Compression statistics
    # ==================================================
    st.markdown("---")
    st.header("📊 Compression Statistics")

    m1, m2, m3 = st.columns(3)
    m1.metric("Original Size", f"{size_old:.1f} KB")
    m2.metric("Compressed Size", f"{size_new:.1f} KB", f"-{reduction:.1f}%")
    m3.metric(
        "High-Importance Tiles",
        f"{np.sum(preds >= high_th)} / {len(preds)}"
    )

    st.progress(int((size_new / size_old) * 100) / 100)
    st.caption(
        f"The compressed image uses only "
        f"{int((size_new / size_old) * 100)}% of the original size."
    )

    # ==================================================
    # Download and heatmap visualization
    # ==================================================
    with open(out_path, "rb") as f:
        st.download_button(
            "📥 Download Compressed Image",
            f,
            file_name=os.path.basename(out_path),
            mime="image/jpeg"
        )

    with st.expander("🔥 View Importance Heatmap"):
        heatmap = np.zeros_like(img_np[..., :3])
        for (x, y), p in zip(coords, preds):
            if p >= high_th:
                heatmap[y:y + TILE_SIZE, x:x + TILE_SIZE, 0] = int(p * 255)

        overlay = cv2.addWeighted(
            img_np[..., :3], 0.7, heatmap, 0.3, 0
        )

        st.image(
            overlay,
            caption="Red regions indicate high importance"
        )
