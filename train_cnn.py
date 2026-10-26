"""
train_cnn.py
------------
Model A: Custom Convolutional Neural Network (CNN)
Developed by: Student 1

This script builds, trains, and saves a custom CNN from scratch
to classify e-waste images into 4 classes.

Architecture overview:
1. Rescaling layer: Normalizes 0-255 pixel values to [0, 1].
2. Conv2D block 1: 32 filters (3x3), ReLU activation, extracts low-level features (edges/lines).
3. MaxPooling2D: Downsamples spatial dimensions by 2x.
4. Conv2D block 2: 64 filters (3x3), ReLU activation, extracts mid-level patterns (shapes/parts).
5. MaxPooling2D: Second downsampling step.
6. Flatten layer: Converts 2D feature maps into a 1D vector.
7. Dense layer: 64 hidden neurons with ReLU activation.
8. Output Dense layer: 4 neurons with Softmax activation (outputs class probabilities).
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Hyperparameters
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
CLASSES = ["mobile_phone", "charger_cable", "earphones", "circuit_board"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_datasets():
    """Loads training and validation datasets from directory."""
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            "Dataset folders not found. Please run 'python generate_dataset.py' first!"
        )

    print("Loading datasets from 'data/' directory...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASSES,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=CLASSES,
        shuffle=False
    )

    # Cache and prefetch for smooth pipeline execution
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def build_custom_cnn():
    """Builds a simple, interpretable Custom CNN from scratch."""
    model = models.Sequential([
        # 1. Input and Rescaling Layer (normalizes [0, 255] to [0.0, 1.0])
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Rescaling(1.0 / 255),

        # 2. First Convolutional Block: extracts basic features (edges, color gradients)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),

        # 3. Second Convolutional Block: extracts composite patterns (corners, shapes)
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),

        # 4. Flattening: converts 2D feature maps (24x24x64) into 1D feature vector (36,864)
        layers.Flatten(name="flatten"),

        # 5. Fully-Connected (Dense) Layer with Dropout to reduce overfitting
        layers.Dense(64, activation="relu", name="dense_features"),
        layers.Dropout(0.2, name="dropout"),

        # 6. Classification Output Layer: 4 neurons for 4 classes with Softmax
        layers.Dense(len(CLASSES), activation="softmax", name="output_probabilities")
    ], name="Custom_CNN_Student1")

    return model

def train():
    """Compiles, trains, and evaluates the Custom CNN model."""
    print("=" * 60)
    print("ReBoot AI - Model A: Custom CNN Training (Student 1)")
    print("=" * 60)

    train_ds, val_ds = load_datasets()
    model = build_custom_cnn()
    model.summary()

    # Compile with Adam optimizer and sparse categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    print(f"\nTraining Custom CNN for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # Save trained model
    model_save_path = os.path.join(MODELS_DIR, "custom_cnn.keras")
    model.save(model_save_path)
    print(f"\n[OK] Model successfully saved to: {model_save_path}")

    # Save history metrics to JSON for evaluation and plotting
    history_dict = {
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "epochs": EPOCHS
    }

    history_path = os.path.join(RESULTS_DIR, "history_cnn.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[OK] Training history saved to: {history_path}")
    print("=" * 60)

if __name__ == "__main__":
    train()
