"""
train_mobilenet.py
------------------
Model B: Transfer Learning using MobileNetV2
Developed by: Student 2

This script implements Transfer Learning using Google's MobileNetV2 architecture
pre-trained on ImageNet.

Transfer Learning workflow:
1. Load MobileNetV2 feature extractor without the final 1000-class classification head (`include_top=False`).
2. Attempt to use ImageNet pre-trained weights (with offline fallback if unavailable).
3. Freeze all convolutional base layers (`trainable = False`) to preserve learned visual representations.
4. Preprocess incoming raw images using MobileNetV2's specific normalization [-1, 1].
5. Append a custom task-specific classification head:
   - GlobalAveragePooling2D (aggregates spatial feature maps into compact representations).
   - Dense(64, activation='relu') + Dropout(0.2).
   - Dense(4, activation='softmax') for our 4 e-waste categories.
6. Train only the newly added classification head.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

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

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def build_transfer_model():
    """Builds MobileNetV2 Transfer Learning model."""
    print("Loading MobileNetV2 base model...")
    weights_choice = "imagenet"
    weights_status = "Pre-trained ImageNet weights"

    try:
        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights=weights_choice
        )
        print("  -> Successfully loaded pre-trained ImageNet weights.")
    except Exception as e:
        print(f"  [Notice] Could not download ImageNet weights ({e}).")
        print("  -> Fallback: Initializing MobileNetV2 with randomly initialized weights (weights=None).")
        weights_choice = None
        weights_status = "Randomly Initialized (Offline fallback - NOT pre-trained)"
        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights=weights_choice
        )

    # Freeze base model weights so they are not updated during training
    base_model.trainable = False

    # Construct final model pipeline
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # MobileNetV2 requires input scaling from [0, 255] to [-1, 1]
    x = preprocess_input(inputs)

    # Pass through frozen feature extractor (training=False ensures batchnorm stays in inference mode)
    x = base_model(x, training=False)

    # Task-specific classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="output_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer_Student2")
    return model, weights_status

def train():
    """Compiles, trains, and saves the MobileNetV2 Transfer Learning model."""
    print("=" * 60)
    print("ReBoot AI - Model B: MobileNetV2 Transfer Learning (Student 2)")
    print("=" * 60)

    train_ds, val_ds = load_datasets()
    model, weights_status = build_transfer_model()
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    print(f"\nTraining MobileNetV2 classification head for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # Save model
    model_save_path = os.path.join(MODELS_DIR, "mobilenet_v2.keras")
    model.save(model_save_path)
    print(f"\n[OK] Model successfully saved to: {model_save_path}")

    # Save training history
    history_dict = {
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "weights_status": weights_status,
        "epochs": EPOCHS
    }

    history_path = os.path.join(RESULTS_DIR, "history_mobilenet.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[OK] Training history saved to: {history_path}")
    print("=" * 60)

if __name__ == "__main__":
    train()
