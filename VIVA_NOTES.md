# ReBoot AI — Deep Learning Lab Viva Preparation Notes

This guide is structured to help both undergraduate students prepare for their Deep Learning lab viva and project presentation.

---

## 👨‍🎓 Section 1: Student 1 — Custom Convolutional Neural Network (CNN)

### 1. What is Convolution in Deep Learning?
- **Answer:** A mathematical operation where a small matrix (called a **kernel** or **filter**, e.g., 3×3) slides across an input image. At each position, it computes the element-wise product between the filter weights and the overlapping pixel values, then sums them up to produce a single value.
- **Why it matters:** Unlike standard dense neural networks, convolution preserves spatial relationships between neighboring pixels and drastically reduces the number of learnable parameters through **weight sharing** (the same filter is reused across the entire image).

### 2. What are Filters and Feature Maps?
- **Filter (Kernel):** A small set of learnable weights (e.g., 3×3×3 for an RGB input) designed to detect specific visual features such as edges, color contrasts, or lines.
- **Feature Map (Activation Map):** The output 2D matrix produced after sliding a filter over the input and applying an activation function.
- **Hierarchy:** Early layers produce low-level feature maps (horizontal/vertical edges). Intermediate layers produce mid-level feature maps (corners, textures, cable curves). Deeper layers capture high-level object parts (phone screens, IC chips).

### 3. Why use ReLU (Rectified Linear Activation)?
- **Formula:** $f(x) = \max(0, x)$
- **Purpose:** 
  1. Introduces **non-linearity** into the network, allowing it to learn complex, non-linear boundaries. Without non-linear activations, stacking multiple convolution layers would merely equal a single linear transformation.
  2. Solves the **vanishing gradient problem** that plagues Sigmoid and Tanh activations for positive values.
  3. Computationally very fast to evaluate and differentiate.

### 4. Why use Max Pooling?
- **Operation:** Slides a window (e.g., 2×2) across the feature map and retains only the maximum value within that window with a stride of 2.
- **Benefits:**
  1. **Dimensionality Reduction:** Downsamples the spatial size by half (e.g., 96×96 → 48×48), cutting computation and memory usage.
  2. **Translation Invariance:** Small shifts or rotations in the input image do not drastically alter the pooled output.
  3. **Receptive Field Expansion:** Allows deeper layers to view larger context in the original input.

### 5. What is Softmax and why is it used at the output?
- **Formula:** $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$
- **Purpose:** Converts raw unnormalized network outputs (logits) into a valid probability distribution over the $K$ classes (here, 4 classes). Every predicted probability is strictly between 0 and 1, and the sum of all class probabilities equals exactly 1.0 (100%).

### 6. How are Loss and Accuracy Measured?
- **Sparse Categorical Cross-Entropy Loss:** Quantifies the error between true integer labels ($y \in \{0, 1, 2, 3\}$) and predicted probabilities ($\hat{y}$):
  $$L = -\log(\hat{y}_{\text{true}})$$
  When the model assigns a high probability to the correct class, loss approaches 0; if it assigns a low probability, loss penalizes heavily.
- **Accuracy:** The percentage of correctly predicted images out of total evaluated images:
  $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} \times 100\%$$

---

## 👨‍🎓 Section 2: Student 2 — Transfer Learning using MobileNetV2

### 1. What is Transfer Learning?
- **Answer:** A deep learning technique where a model developed for a large, comprehensive task (source task) is reused as the starting point for a different, specialized task (target task).
- **Core Intuition:** Instead of starting from randomly initialized weights and needing millions of images, we exploit generalized visual patterns (edges, shapes, textures) that the base network already learned on ImageNet (1.4 million images across 1,000 categories).

### 2. Why Choose MobileNetV2 for this Lab?
- **Lightweight & Efficient:** Developed by Google specifically for resource-constrained devices (mobile phones, embedded systems, edge devices).
- **Depthwise Separable Convolutions:** Splits standard convolution into:
  1. *Depthwise convolution:* applies a single 3×3 filter per input channel.
  2. *Pointwise convolution:* 1×1 convolution to combine channels.
  This achieves up to an 8–9× reduction in computation with only minimal accuracy loss.
- **Inverted Residuals & Linear Bottlenecks:** Memory-efficient shortcut connections between thin bottleneck layers.

### 3. What are Pre-trained Weights?
- **Answer:** The saved numerical values of all kernels and biases resulting from training MobileNetV2 on ImageNet.
- **Offline Fallback Note:** If an internet connection or proxy blocks downloading weights (`weights='imagenet'`), the code falls back to `weights=None` (random initialization) with an explicit notice. We never claim randomly initialized weights are pre-trained.

### 4. Why Freeze the Base Layers?
- **Answer:** We set `base_model.trainable = False`.
- **Reason:** The newly appended classification head has randomly initialized weights. In the initial epochs, backpropagation produces very large gradients. If the base layers were unfrozen, these volatile gradients would overwrite and destroy the delicate, well-trained feature detectors learned from ImageNet (known as *catastrophic forgetting*).

### 5. How Does Transfer Learning Differ from Training from Scratch?

| Criterion | Custom CNN (From Scratch) | MobileNetV2 (Transfer Learning) |
| :--- | :--- | :--- |
| **Weight Initialization** | Random weights (e.g., Glorot/He) | Pre-trained ImageNet weights |
| **Data Requirement** | Requires sufficient domain data | Works effectively on small datasets |
| **Training Time** | Fast per epoch for tiny CNN, but slow to converge on complex data | Rapid convergence because features already exist |
| **Architecture** | Simple, hand-crafted sequential layers | Advanced (inverted residuals, depthwise separable) |
| **Parameter Count** | Small (~2.3M or fewer) | Larger (~2.2M - 3.5M parameters) |

---

## 🤝 Section 3: Joint Knowledge for Both Students

### 1. The 4 E-Waste Classes in ReBoot AI
1. **`mobile_phone`:** Smartphones/feature phones (hazards: swollen Li-ion batteries; solutions: data wipe, factory reset, certified buyback).
2. **`charger_cable`:** Adapters and charging cords (hazards: copper burning emits toxic dioxins; solutions: repurposing, mechanical copper granulation).
3. **`earphones`:** Audio earbuds and cables (hazards: micro-batteries, magnets; solutions: re-soldering, accessory collection).
4. **`circuit_board`:** Printed circuit boards (hazards: lead solder, brominated flame retardants; solutions: closed-loop metallurgical smelting).

### 2. Evaluation Metrics Explained
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all items predicted as phones, how many were actually phones? (Avoids false alarms).
- **Recall:** $\frac{TP}{TP + FN}$ — Out of all real phones in the test set, how many did the model detect? (Avoids missed e-waste).
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean balancing precision and recall.
- **Confusion Matrix:** A square matrix where rows represent ground truth classes and columns represent model predictions. Diagonal elements represent correct classifications.

### 3. Critical Limitation: Why Synthetic Data Differs from Real Photos
- **Laboratory Prototype Nature:** The images were generated using Pillow at 96×96 pixels with geometric shapes and simulated textures to ensure 100% reproducibility without external API keys or huge downloads.
- **Domain Gap:** Models can achieve very high accuracy on synthetic test sets because test images follow the same generative rules. In the real world, lighting variations, background clutter, cracked screens, and varied camera angles create a "domain shift" requiring real-world fine-tuning.
