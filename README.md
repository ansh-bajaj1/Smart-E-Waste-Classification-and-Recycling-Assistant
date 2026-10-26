# ReBoot AI — Smart E-Waste Classification and Recycling Assistant

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Lab%20Prototype-success.svg)]()

> **Identify. Understand. Recycle Responsibly.**  
> An educational Deep Learning laboratory project developed by two undergraduate students comparing a **Custom Convolutional Neural Network (CNN)** against **Transfer Learning via MobileNetV2**.

---

## 1. Project Objective

Electronic waste (e-waste) is rapidly accumulating worldwide. Proper disposal requires recognizing the item type, recognizing toxic hazards (such as lithium batteries or lead solder), and routing components toward certified recyclers.

**ReBoot AI** provides:
1. **Automated Image Classification:** Accurately classifies an image into one of four common e-waste categories:
   - 📱 `mobile_phone`
   - 🔌 `charger_cable`
   - 🎧 `earphones`
   - 🖲️ `circuit_board`
2. **Responsible Disposal & Circularity Advice:** Offers immediate next steps (data sanitation, repair/refurbish checks, certified collection points) and environmental hazard warnings.
3. **Fair Deep Learning Comparison:** Contrasts a hand-crafted CNN against a pre-trained MobileNetV2 backbone on identical held-out test data.

---

## 2. Deep Learning Concepts Demonstrated

1. **Custom CNN (From Scratch) — Student 1:**
   - Spatial convolution with 3×3 kernels and ReLU activations.
   - Max Pooling for spatial dimension downsampling and translation invariance.
   - Flattening and dense feature learning with Dropout regularization.
   - Softmax multi-class output and sparse categorical cross-entropy optimization.
2. **Transfer Learning — Student 2:**
   - Pre-trained backbone reuse using Google's **MobileNetV2**.
   - Input normalization via `preprocess_input` (scaling pixels to $[-1, 1]$).
   - Base layer freezing (`trainable = False`) to preserve ImageNet representations.
   - Custom classification head with `GlobalAveragePooling2D` and dense layers.
3. **Fair Evaluation & Metrics:**
   - Held-out test split evaluation.
   - Confusion matrices and scikit-learn classification reports (Precision, Recall, F1-Score).
   - Single-image inference latency benchmarking.
   - Epoch-by-epoch loss and accuracy learning curves.

---

## 3. Student-Wise Division of Work

| Student | Assigned Module | Key Deliverables |
| :--- | :--- | :--- |
| **Student 1** | **Model A: Custom CNN** | Designed 2D convolutional network from scratch; wrote `train_cnn.py`; tuned batch size, learning rates, and feature maps. |
| **Student 2** | **Model B: Transfer Learning** | Integrated MobileNetV2 with ImageNet weights; wrote `train_mobilenet.py`; configured headless architecture, layer freezing, and custom head. |
| **Joint Work** | **Data & Evaluation & UI** | Synthetic dataset generation (`generate_dataset.py`), fair test evaluation pipeline (`evaluate.py`), and Streamlit web application (`app.py`). |

---

## 4. Dataset Generation & Limitations

To ensure the project runs without requiring gigabytes of downloads, external API credentials, or web scrapers, a reproducible **synthetic dataset generator** is included using Python's Pillow (PIL) library:
- **Image Resolution:** 96 × 96 pixels.
- **Classes:** 4 categories (`mobile_phone`, `charger_cable`, `earphones`, `circuit_board`).
- **Data Split:**
  - 70% Training (~112 images per class)
  - 15% Validation (~24 images per class)
  - 15% Test (~24 images per class)
- **Reproducibility:** Seeded with `RANDOM_SEED = 42`.

### ⚠️ Important Notice on Synthetic Data
*Synthetic images allow students to verify pipeline functionality, tensor shapes, gradient descent, and training stability. Because the synthetic test images share geometric styling with the training images, models achieve high accuracy. This demonstrates architecture correctness, but does not represent real-world robustness on complex, physical photographs.*

---

## 5. Model Architectures Summary

### Model A: Custom CNN (Student 1)
```
Input (96, 96, 3)
   │
   ▼
Rescaling (1./255)
   │
   ▼
Conv2D (32 filters, 3x3, ReLU, Same Padding)
   │
   ▼
MaxPooling2D (2x2) ──> Spatial downsample to (48, 48, 32)
   │
   ▼
Conv2D (64 filters, 3x3, ReLU, Same Padding)
   │
   ▼
MaxPooling2D (2x2) ──> Spatial downsample to (24, 24, 64)
   │
   ▼
Flatten (36,864 vector)
   │
   ▼
Dense (64 units, ReLU) + Dropout (0.2)
   │
   ▼
Dense (4 units, Softmax) ──> Class Probabilities
```

### Model B: MobileNetV2 Transfer Learning (Student 2)
```
Input (96, 96, 3)
   │
   ▼
preprocess_input ([-1, 1] scaling)
   │
   ▼
MobileNetV2 Base (Pre-trained on ImageNet, Frozen, include_top=False)
   │
   ▼
GlobalAveragePooling2D ──> Spatial pooling to (1280 vector)
   │
   ▼
Dense (64 units, ReLU) + Dropout (0.2)
   │
   ▼
Dense (4 units, Softmax) ──> Class Probabilities
```

---

## 6. Project Folder Structure

```
reboot-ai/
├── app.py                  # Streamlit web application
├── generate_dataset.py     # Pillow-based synthetic image generator
├── train_cnn.py            # Student 1: Custom CNN training script
├── train_mobilenet.py      # Student 2: MobileNetV2 transfer learning script
├── evaluate.py             # Evaluation & comparison script
├── requirements.txt        # Minimal practical dependencies
├── README.md               # Complete project documentation
├── VIVA_NOTES.md           # Student viva questions & answers
├── data/                   # Generated dataset split into train/val/test
│   ├── train/
│   ├── val/
│   └── test/
├── models/                 # Saved trained models (.keras)
│   ├── custom_cnn.keras
│   └── mobilenet_v2.keras
└── results/                # Saved metrics and comparison plots
    ├── metrics.json
    ├── history_cnn.json
    ├── history_mobilenet.json
    ├── accuracy_loss_curves.png
    └── confusion_matrices.png
```

---

## 7. Installation & Execution Guide

### Step 1: Set Up Python Environment
Open a terminal in the project directory:

```bash
# Optional: Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate the Dataset
```bash
python generate_dataset.py
```
*Output: Creates `data/train/`, `data/val/`, and `data/test/` folders with 640 images across the 4 classes.*

### Step 4: Train Custom CNN (Student 1)
```bash
python train_cnn.py
```
*Output: Trains for 8 epochs, saves `models/custom_cnn.keras` and `results/history_cnn.json`.*

### Step 5: Train MobileNetV2 (Student 2)
```bash
python train_mobilenet.py
```
*Output: Loads MobileNetV2, trains custom head for 8 epochs, saves `models/mobilenet_v2.keras` and `results/history_mobilenet.json`.*

### Step 6: Evaluate Models & Generate Metrics
```bash
python evaluate.py
```
*Output: Evaluates both models on held-out test data, prints comparison table, and saves `results/metrics.json`, `accuracy_loss_curves.png`, and `confusion_matrices.png`.*

### Step 7: Launch the Web Application
```bash
streamlit run app.py
```
*Access the interactive interface in your web browser at `http://localhost:8501`.*

---

## 8. Responsible Disposal Knowledge Base

| Category | Key Environmental Hazard | Safe Handling Recommendation |
| :--- | :--- | :--- |
| **Mobile Phone** | Lithium-Ion battery punctures cause fires in municipal sorting facilities. | Wipe all private data; check for battery swelling; donate or use manufacturer take-back kiosks. |
| **Charger Cable** | Burning PVC jackets releases toxic dioxin gas and hydrochloric acid fumes. | Discontinue if exposed wires appear; bundle undamaged cords for reuse; deliver to electrical scrap recyclers. |
| **Earphones** | Miniature neodymium magnets and lithium coin cells contaminate scrap streams. | Clean silicone tips for reuse; re-solder broken 3.5mm leads; dispose in small-electronics bins. |
| **Circuit Board** | Lead solders and brominated flame retardants (BFRs) leach heavy metals into groundwater. | Never burn or acid-leach at home; route to certified closed-loop smelters (R2/e-Stewards) for precious metal extraction. |

---

## 9. Troubleshooting Tips

- **Missing Model Error:** If `app.py` displays a warning that models are missing, verify that you completed Steps 3, 4, and 5.
- **ImageNet Download Issue:** If your network firewall prevents downloading ImageNet weights during `train_mobilenet.py`, the script automatically switches to `weights=None` as an offline fallback.
- **Memory Overhead:** Both models are lightweight (< 50MB) and comfortably train on a standard CPU within 1–2 minutes.

---

## 10. Future Enhancements

1. **Real-World Transfer Fine-Tuning:** Collect a curated dataset of real e-waste photographs under varying room lightings and unfreeze the top layers of MobileNetV2 for gentle fine-tuning.
2. **Object Detection (Bounding Boxes):** Expand from classification to localization (e.g., using YOLO or SSD) to identify multiple e-waste items on a conveyor belt.
3. **Geo-Location Integration:** Connect the web app to a live municipal map API to guide users to the nearest certified e-waste drop-off bin.
