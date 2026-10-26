"""
evaluate.py
-----------
Fair Evaluation and Model Comparison Script for ReBoot AI

Evaluates both trained models:
1. Model A: Custom CNN (Student 1)
2. Model B: MobileNetV2 Transfer Learning (Student 2)

Evaluated on the exact same held-out test split (data/test/).
Calculates actual metrics:
- Overall Test Accuracy
- Precision, Recall, F1-Score (via scikit-learn)
- Confusion Matrix (via scikit-learn)
- Approximate single-image inference latency
- Generates publication-ready Matplotlib plots for training/validation curves and confusion matrices
- Saves results to results/metrics.json for Streamlit web app consumption
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = (96, 96)
CLASSES = ["mobile_phone", "charger_cable", "earphones", "circuit_board"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_data():
    """Loads all test images and true integer labels as numpy arrays."""
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}. Run generate_dataset.py first!")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=32,
        label_mode="int",
        class_names=CLASSES,
        shuffle=False
    )

    all_images = []
    all_labels = []

    for images, labels in test_ds:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())

    X_test = np.concatenate(all_images, axis=0)
    y_test = np.concatenate(all_labels, axis=0)
    return X_test, y_test

def get_model_size_mb(filepath):
    """Calculates model file size in Megabytes."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 2)
    return 0.0

def evaluate_single_model(model_name, model_path, X_test, y_test):
    """Evaluates one model, measuring accuracy, report, confusion matrix, and latency."""
    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}. Skipping {model_name}.")
        return None

    print(f"\nEvaluating: {model_name}...")
    model = tf.keras.models.load_model(model_path)

    # Measure inference time over multiple runs for accuracy
    start_time = time.perf_counter()
    raw_predictions = model.predict(X_test, verbose=0)
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / len(X_test)) * 1000.0

    y_pred = np.argmax(raw_predictions, axis=1)

    # Calculate actual metrics
    test_accuracy = float(np.mean(y_pred == y_test))
    report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    total_params = model.count_params()
    size_mb = get_model_size_mb(model_path)

    print(f"  Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Avg Latency per image: {avg_latency_ms:.2f} ms")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Model Size: {size_mb} MB")

    return {
        "model_name": model_name,
        "test_accuracy": round(test_accuracy * 100, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "total_parameters": total_params,
        "model_size_mb": size_mb,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "y_pred": y_pred.tolist()
    }

def plot_curves():
    """Generates and saves combined training & validation accuracy and loss curves."""
    cnn_hist_path = os.path.join(RESULTS_DIR, "history_cnn.json")
    mob_hist_path = os.path.join(RESULTS_DIR, "history_mobilenet.json")

    has_cnn = os.path.exists(cnn_hist_path)
    has_mob = os.path.exists(mob_hist_path)

    if not has_cnn and not has_mob:
        print("[Notice] No history files found to plot curves.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "cnn_train": "#1f77b4", "cnn_val": "#aec7e8",
        "mob_train": "#2ca02c", "mob_val": "#98df8a"
    }

    # 1. Accuracy Plot
    ax1 = axes[0]
    if has_cnn:
        with open(cnn_hist_path) as f:
            h_cnn = json.load(f)
        epochs = range(1, len(h_cnn["accuracy"]) + 1)
        ax1.plot(epochs, h_cnn["accuracy"], label="CNN Train Acc", color=colors["cnn_train"], lw=2)
        ax1.plot(epochs, h_cnn["val_accuracy"], label="CNN Val Acc", color=colors["cnn_val"], lw=2, linestyle="--")

    if has_mob:
        with open(mob_hist_path) as f:
            h_mob = json.load(f)
        epochs = range(1, len(h_mob["accuracy"]) + 1)
        ax1.plot(epochs, h_mob["accuracy"], label="MobileNetV2 Train Acc", color=colors["mob_train"], lw=2)
        ax1.plot(epochs, h_mob["val_accuracy"], label="MobileNetV2 Val Acc", color=colors["mob_val"], lw=2, linestyle="--")

    ax1.set_title("Training & Validation Accuracy Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Loss Plot
    ax2 = axes[1]
    if has_cnn:
        ax2.plot(epochs, h_cnn["loss"], label="CNN Train Loss", color=colors["cnn_train"], lw=2)
        ax2.plot(epochs, h_cnn["val_loss"], label="CNN Val Loss", color=colors["cnn_val"], lw=2, linestyle="--")

    if has_mob:
        ax2.plot(epochs, h_mob["loss"], label="MobileNetV2 Train Loss", color=colors["mob_train"], lw=2)
        ax2.plot(epochs, h_mob["val_loss"], label="MobileNetV2 Val Loss", color=colors["mob_val"], lw=2, linestyle="--")

    ax2.set_title("Training & Validation Loss Comparison", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "accuracy_loss_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[OK] Accuracy & Loss curves saved to: {curve_path}")

def plot_confusion_matrices(eval_results, y_test):
    """Plots confusion matrix heatmaps side-by-side."""
    models_to_plot = [m for m in eval_results if m is not None]
    if not models_to_plot:
        return

    n_models = len(models_to_plot)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, models_to_plot):
        cm = np.array(res["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.set_title(f"Confusion Matrix: {res['model_name']}\n(Acc: {res['test_accuracy']}%)", fontsize=11, fontweight="bold")

        tick_marks = np.arange(len(CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASSES, fontsize=9)

        # Print numbers inside cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold")

        ax.set_ylabel("True Category")
        ax.set_xlabel("Predicted Category")
        ax.grid(False)

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrices saved to: {cm_path}")

def evaluate_all():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("ReBoot AI - Fair Model Evaluation on Held-Out Test Set")
    print("=" * 60)

    X_test, y_test = load_test_data()
    print(f"Loaded {len(X_test)} held-out test images across 4 classes.")

    cnn_path = os.path.join(MODELS_DIR, "custom_cnn.keras")
    mob_path = os.path.join(MODELS_DIR, "mobilenet_v2.keras")

    res_cnn = evaluate_single_model("Custom CNN (Student 1)", cnn_path, X_test, y_test)
    res_mob = evaluate_single_model("MobileNetV2 (Student 2)", mob_path, X_test, y_test)

    # Save to metrics.json
    all_metrics = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_sample_count": int(len(X_test)),
        "models": {
            "custom_cnn": res_cnn,
            "mobilenet_v2": res_mob
        }
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[OK] Comprehensive evaluation metrics saved to: {metrics_path}")

    # Plot graphs
    plot_curves()
    plot_confusion_matrices([res_cnn, res_mob], y_test)

    # Print fair comparison table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'Custom CNN (Student 1)':<18} | {'MobileNetV2 (Student 2)':<18}")
    print("-" * 65)
    acc_cnn = f"{res_cnn['test_accuracy']}%" if res_cnn else "N/A"
    acc_mob = f"{res_mob['test_accuracy']}%" if res_mob else "N/A"
    print(f"{'Test Accuracy':<25} | {acc_cnn:<18} | {acc_mob:<18}")

    lat_cnn = f"{res_cnn['avg_latency_ms']:.2f} ms" if res_cnn else "N/A"
    lat_mob = f"{res_mob['avg_latency_ms']:.2f} ms" if res_mob else "N/A"
    print(f"{'Inference Latency/Image':<25} | {lat_cnn:<18} | {lat_mob:<18}")

    par_cnn = f"{res_cnn['total_parameters']:,}" if res_cnn else "N/A"
    par_mob = f"{res_mob['total_parameters']:,}" if res_mob else "N/A"
    print(f"{'Total Parameters':<25} | {par_cnn:<18} | {par_mob:<18}")

    siz_cnn = f"{res_cnn['model_size_mb']} MB" if res_cnn else "N/A"
    siz_mob = f"{res_mob['model_size_mb']} MB" if res_mob else "N/A"
    print(f"{'Model File Size':<25} | {siz_cnn:<18} | {siz_mob:<18}")
    print("=" * 65)

    print("\nIMPORTANT LAB NOTE ON SYNTHETIC DATA:")
    print("Synthetic images test how well architectures learn geometric rules.")
    print("Because the test set shares the same drawing style as the training set,")
    print("models can achieve near-perfect synthetic test accuracy.")
    print("This demonstrates pipeline correctness, NOT real-world field reliability.")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_all()
