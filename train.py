import os
import sys
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    Input, Concatenate, Conv2D, MaxPooling2D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.utils import class_weight
import pytesseract

# ======================================================
# Configuration
# ======================================================
DATA_DIR = "processed"
IMP_DIR = os.path.join(DATA_DIR, "imp")
NOT_IMP_DIR = os.path.join(DATA_DIR, "not_imp")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 3
MODEL_OUT = "models/tile_classifier_ocr.keras"

os.makedirs("models", exist_ok=True)

# ======================================================
# Configure Tesseract OCR path
# (Leave empty if already in PATH)
# ======================================================
pytesseract.pytesseract.tesseract_cmd = r""


# ======================================================
# Validate Tesseract installation
# ======================================================
if shutil.which("tesseract") is None and not pytesseract.pytesseract.tesseract_cmd:
    print("ERROR: Tesseract OCR is not installed or not found in PATH.")
    sys.exit(1)


# ======================================================
# Validate dataset directories
# ======================================================
if not os.path.exists(IMP_DIR) or not os.path.exists(NOT_IMP_DIR):
    print("ERROR: Processed dataset folders not found.")
    sys.exit(1)


# ======================================================
# OCR text mask extraction
# ======================================================
def detect_text_mask(img):
    """
    Detects text regions using OCR and returns
    a normalized binary mask.
    """
    if img is None:
        return None

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

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

    mask = cv2.resize(mask, IMG_SIZE)
    return mask.astype(np.float32) / 255.0


# ======================================================
# Custom data generator with OCR channel
# ======================================================
class OCRDataGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of images with an additional OCR mask channel.
    """

    def __init__(self, file_list, labels, batch_size=BATCH_SIZE,
                 img_size=IMG_SIZE, shuffle=True):
        self.file_list = np.array(file_list)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        X, y = [], []

        for idx in indexes:
            img = cv2.imread(self.file_list[idx])
            if img is None:
                continue

            img = cv2.resize(img, self.img_size)
            mask = detect_text_mask(img)

            if mask is None:
                continue

            img_norm = img.astype(np.float32) / 255.0
            img_4ch = np.dstack([img_norm, mask])

            X.append(img_4ch)
            y.append(self.labels[idx])

        return np.array(X), np.array(y)


# ======================================================
# Model architecture
# ======================================================
def build_model(input_shape=(128, 128, 4)):
    """
    Builds a dual-branch CNN combining visual
    features and OCR spatial features.
    """
    input_tensor = Input(shape=input_shape)

    rgb_input = input_tensor[..., :3]
    ocr_input = input_tensor[..., 3:]

    # Visual branch
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3)
    )

    x = base(rgb_input)
    x = GlobalAveragePooling2D()(x)

    # OCR branch
    y = Reshape((128, 128, 1))(ocr_input)
    y = Conv2D(16, (3, 3), activation="relu")(y)
    y = MaxPooling2D((2, 2))(y)
    y = GlobalAveragePooling2D()(y)

    # Feature fusion
    combined = Concatenate()([x, y])
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation="relu")(z)
    out = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=input_tensor, outputs=out)

    # Partial fine-tuning
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ======================================================
# Training pipeline
# ======================================================
def main():
    imp_files = [
        os.path.join(IMP_DIR, f)
        for f in os.listdir(IMP_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    not_imp_files = [
        os.path.join(NOT_IMP_DIR, f)
        for f in os.listdir(NOT_IMP_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if not imp_files or not not_imp_files:
        print("ERROR: One or more classes have no images.")
        sys.exit(1)

    files = imp_files + not_imp_files
    labels = [1] * len(imp_files) + [0] * len(not_imp_files)

    # Handle class imbalance
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = {i: weights[i] for i in range(len(weights))}

    indices = np.arange(len(files))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_gen = OCRDataGenerator(
        [files[i] for i in indices[:split]],
        [labels[i] for i in indices[:split]]
    )

    val_gen = OCRDataGenerator(
        [files[i] for i in indices[split:]],
        [labels[i] for i in indices[split:]],
        shuffle=False
    )

    model = build_model()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2
            ),
            ModelCheckpoint(
                MODEL_OUT,
                save_best_only=True
            )
        ]
    )

    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
