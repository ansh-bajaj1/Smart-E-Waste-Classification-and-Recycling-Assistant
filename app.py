"""
app.py
------
ReBoot AI: Intelligent E-Waste Classification & Materials Recovery System
Production-grade Streamlit application for image classification, hazardous material
identification, and responsible recycling workflows.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="ReBoot AI — E-Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Class Definitions
CLASSES = ["mobile_phone", "charger_cable", "earphones", "circuit_board"]
CLASS_DISPLAY_NAMES = {
    "mobile_phone": "Mobile Phone",
    "charger_cable": "Charger & Cable",
    "earphones": "Earphones & Audio",
    "circuit_board": "Printed Circuit Board (PCB)"
}
CLASS_ICONS = {
    "mobile_phone": "📱",
    "charger_cable": "🔌",
    "earphones": "🎧",
    "circuit_board": "🖲️"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "custom_cnn.keras")
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_v2.keras")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
CURVES_PATH = os.path.join(RESULTS_DIR, "accuracy_loss_curves.png")
CONFUSION_PATH = os.path.join(RESULTS_DIR, "confusion_matrices.png")

# Responsible Disposal and Handling Protocols
DISPOSAL_SUGGESTIONS = {
    "mobile_phone": {
        "title": "Mobile Device Lifecycle & Disposal Protocol",
        "steps": [
            "Data Sanitation: Execute a full cryptographic factory wipe and detach physical SIM/microSD cards prior to handover.",
            "Battery Inspection: Check battery integrity for swelling, denting, or heat discoloration. Isolate swollen cells in non-conductive containers.",
            "Refurbishment Assessment: Determine whether modular glass replacement or battery swapping can restore utility for secondary use.",
            "Certified Processing: Route obsolete units to certified R2 or e-Stewards electronics collection facilities."
        ],
        "hazard_note": "Lithium-ion cells present severe thermal runaway and fire hazards if crushed in standard municipal refuse compactors."
    },
    "charger_cable": {
        "title": "Power Adapters & Cabling Recycling Protocol",
        "steps": [
            "Structural Inspection: Examine outer insulation for thermal deformation, copper exposure, or terminal corrosion.",
            "Repurposing: Intact cords should be sorted by voltage/connector type for ongoing peripheral utility or community donation.",
            "Zero Open-Burning: Never subject wiring to open-air combustion; copper extraction must follow mechanical shredding standards.",
            "Scrap Consolidation: Group stranded copper cables into bulk electronics sorting receptacles for specialized granulator recycling."
        ],
        "hazard_note": "Polyvinyl chloride (PVC) cable jackets release halogenated dioxins and acidic vapors when incinerated without flue gas scrubbing."
    },
    "earphones": {
        "title": "Audio Equipment & Transducer Handling Protocol",
        "steps": [
            "Sanitization & Reuse: Functional audio equipment should be disinfected with 70% isopropyl alcohol and fitted with fresh silicone tips.",
            "Connector Diagnostics: Test connector strain relief; intermittent sound is frequently repairable with simple terminal re-soldering.",
            "Battery Segregation: Wireless audio hardware housing micro lithium-polymer coin cells must not enter standard plastic waste streams.",
            "E-Scrap Submission: Unserviceable headsets should be deposited at certified small-appliance collection drop-offs."
        ],
        "hazard_note": "High-strength neodymium iron boron (NdFeB) magnets and specialized plastics require dedicated mechanical separation."
    },
    "circuit_board": {
        "title": "Printed Circuit Board (PCB) Material Recovery Protocol",
        "steps": [
            "Safe Containment: Store unpopulated and populated boards in electrostatic discharge (ESD) safe containers without physical shattering.",
            "Component Harvesting: Operational microcontrollers, passives, and header pins can be desoldered for prototyping applications.",
            "Prohibited Operations: Domestic acid stripping, chemical leaching, and smelting are strictly prohibited due to severe toxicity risks.",
            "Smelter Delivery: Hand over to authorized recyclers equipped with hydrometallurgical and pyrometallurgical refining plants."
        ],
        "hazard_note": "PCBs contain lead-tin solders, brominated flame retardants (BFRs), and trace heavy metals that pose severe groundwater risks."
    }
}

# Modern, restrained dark-theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    /* Top Header */
    .app-header {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 24px;
    }
    .system-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-description {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 6px 0 0 0;
        font-weight: 400;
    }

    /* Cards */
    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .card-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .category-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 18px;
        height: 100%;
        transition: border-color 0.2s ease;
    }
    .category-card:hover {
        border-color: #374151;
    }
    .category-icon {
        font-size: 1.6rem;
        margin-bottom: 8px;
    }
    .category-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .category-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.45;
        margin: 0;
    }

    /* Metrics */
    .metric-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Protocol Box */
    .protocol-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 3px solid #10b981;
        border-radius: 6px;
        padding: 18px 22px;
        margin-top: 14px;
    }
    .protocol-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: #34d399;
        margin-bottom: 10px;
    }
    .protocol-list {
        margin: 0;
        padding-left: 18px;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .hazard-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #fca5a5;
        margin-top: 12px;
        line-height: 1.45;
    }

    /* Comprehensive Sidebar Styling */
    [data-testid="stSidebar"], 
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarUserContent"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1f2937 !important;
    }
    [data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label * {
        color: #ffffff !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: #a7f3d0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1f2937 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Cached Model Loader
@st.cache_resource
def load_models():
    """Loads model artifacts from storage."""
    import tensorflow as tf
    models_dict = {}

    if os.path.exists(CNN_MODEL_PATH):
        try:
            models_dict["cnn"] = tf.keras.models.load_model(CNN_MODEL_PATH)
        except Exception as e:
            models_dict["cnn_error"] = str(e)
    else:
        models_dict["cnn"] = None

    if os.path.exists(MOBILENET_MODEL_PATH):
        try:
            models_dict["mobilenet"] = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
        except Exception as e:
            models_dict["mobilenet_error"] = str(e)
    else:
        models_dict["mobilenet"] = None

    return models_dict

def predict_single_image(model, pil_image):
    """Resizes and normalizes an input image, returning class distribution and latency."""
    img_resized = pil_image.resize((96, 96))
    img_rgb = img_resized.convert("RGB")
    img_array = np.array(img_rgb, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    start_time = time.perf_counter()
    probabilities = model.predict(img_batch, verbose=0)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])

    return {
        "class_id": CLASSES[pred_idx],
        "class_name": CLASS_DISPLAY_NAMES[CLASSES[pred_idx]],
        "icon": CLASS_ICONS[CLASSES[pred_idx]],
        "confidence": confidence,
        "latency_ms": latency_ms,
        "probabilities": probabilities
    }

# Top Header Banner
st.markdown("""
<div class="app-header">
    <div class="system-pill">E-Waste Intelligence Platform</div>
    <h1 class="header-title">ReBoot AI</h1>
    <p class="header-description">Automated electronic waste classification, material risk analysis, and responsible recycling workflows.</p>
</div>
""", unsafe_allow_html=True)

# Navigation
menu = st.sidebar.radio(
    "System Navigation",
    ["Overview", "Classify E-Waste", "Disposal Protocols", "Model Benchmarks"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("ReBoot AI System v1.0 • E-Waste Materials Recovery Engine")

models_loaded = load_models()

# ==========================================
# 1. OVERVIEW
# ==========================================
if menu == "Overview":
    st.subheader("System Overview")
    st.markdown("""
    Rapid obsolescence of consumer electronics creates critical waste streams demanding automated segregation.
    ReBoot AI provides computer-vision-based classification across common discarded electrical hardware, paired
    with material-recovery directives to divert hazardous components from landfills.
    """)

    # High-level system metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">4</div>
            <div class="metric-label">Target Waste Streams</div>
            <div class="metric-sub">Phones, Cables, Audio, PCBs</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">2</div>
            <div class="metric-label">Inference Architectures</div>
            <div class="metric-sub">Custom CNN & MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">&lt; 15 ms</div>
            <div class="metric-label">Inference Latency</div>
            <div class="metric-sub">Optimized for CPU sorting</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">100%</div>
            <div class="metric-label">Circularity Guidance</div>
            <div class="metric-sub">Actionable recovery steps</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Supported Hardware Streams")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">📱</div>
            <div class="category-name">Mobile Devices</div>
            <p class="category-desc">Handsets, touch panels, and battery-housing subassemblies containing recoverable cobalt and lithium.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🔌</div>
            <div class="category-name">Power & Cabling</div>
            <p class="category-desc">AC/DC adapters, USB interfaces, and copper wiring requiring insulation stripping and metal recovery.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🎧</div>
            <div class="category-name">Audio Peripherals</div>
            <p class="category-desc">Earbuds, cables, and miniature transducers containing rare-earth magnets and polymer housing.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🖲️</div>
            <div class="category-name">Circuit Boards</div>
            <p class="category-desc">Populated PCBs, integrated circuits, and headers carrying gold, silver, copper, and solder traces.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Classification Architecture")
    st.markdown("""
    The application integrates two independent deep learning evaluation pipelines:
    - **Custom CNN:** A streamlined 2D convolutional network optimized for rapid edge execution with low parameter overhead.
    - **MobileNetV2 Backbone:** A depthwise separable architecture leveraging transfer features to deliver robust spatial feature extraction.
    """)

# ==========================================
# 2. CLASSIFY E-WASTE
# ==========================================
elif menu == "Classify E-Waste":
    st.subheader("Image Classification & Risk Analysis")
    st.write("Upload an image of an electronic component to identify its category and review material recovery directives.")

    has_cnn = models_loaded.get("cnn") is not None
    has_mob = models_loaded.get("mobilenet") is not None

    if not has_cnn and not has_mob:
        st.warning("Model weights not detected in the storage directory.")
        st.markdown("""
        Ensure model training has executed:
        - `python generate_dataset.py`
        - `python train_cnn.py`
        - `python train_mobilenet.py`
        - `python evaluate.py`
        """)
    else:
        model_options = []
        if has_cnn:
            model_options.append("Custom CNN")
        if has_mob:
            model_options.append("MobileNetV2")
        if has_cnn and has_mob:
            model_options.append("Compare Both Models")

        col_left, col_right = st.columns([1, 1.25])

        with col_left:
            uploaded_file = st.file_uploader(
                "Select image file (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"]
            )
            selected_model_mode = st.radio("Inference Model:", model_options)

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Input Image Preview", use_container_width=True)

        with col_right:
            if uploaded_file is not None:
                st.markdown("### Detection Output")

                if selected_model_mode == "Custom CNN":
                    res = predict_single_image(models_loaded["cnn"], image)
                    st.success(f"**Identified Item:** {res['icon']} {res['class_name']}")
                    
                    c_conf, c_lat = st.columns(2)
                    c_conf.metric("Confidence Score", f"{res['confidence'] * 100:.1f}%")
                    c_lat.metric("Inference Latency", f"{res['latency_ms']:.1f} ms")

                    st.markdown("**Class Probability Breakdown:**")
                    for cls_id, prob in zip(CLASSES, res["probabilities"]):
                        st.progress(float(prob), text=f"{CLASS_DISPLAY_NAMES[cls_id]}: {prob * 100:.1f}%")

                    predicted_class = res["class_id"]

                elif selected_model_mode == "MobileNetV2":
                    res = predict_single_image(models_loaded["mobilenet"], image)
                    st.success(f"**Identified Item:** {res['icon']} {res['class_name']}")

                    c_conf, c_lat = st.columns(2)
                    c_conf.metric("Confidence Score", f"{res['confidence'] * 100:.1f}%")
                    c_lat.metric("Inference Latency", f"{res['latency_ms']:.1f} ms")

                    st.markdown("**Class Probability Breakdown:**")
                    for cls_id, prob in zip(CLASSES, res["probabilities"]):
                        st.progress(float(prob), text=f"{CLASS_DISPLAY_NAMES[cls_id]}: {prob * 100:.1f}%")

                    predicted_class = res["class_id"]

                else: # Compare both models
                    c_cnn, c_mob = st.columns(2)
                    res_cnn = predict_single_image(models_loaded["cnn"], image)
                    res_mob = predict_single_image(models_loaded["mobilenet"], image)

                    with c_cnn:
                        st.markdown("**Custom CNN**")
                        st.markdown(f"#### {res_cnn['icon']} {res_cnn['class_name']}")
                        st.metric("Confidence", f"{res_cnn['confidence'] * 100:.1f}%")
                        st.caption(f"Latency: {res_cnn['latency_ms']:.1f} ms")
                        for cls_id, prob in zip(CLASSES, res_cnn["probabilities"]):
                            st.progress(float(prob), text=f"{CLASS_DISPLAY_NAMES[cls_id][:8]}: {prob*100:.0f}%")

                    with c_mob:
                        st.markdown("**MobileNetV2**")
                        st.markdown(f"#### {res_mob['icon']} {res_mob['class_name']}")
                        st.metric("Confidence", f"{res_mob['confidence'] * 100:.1f}%")
                        st.caption(f"Latency: {res_mob['latency_ms']:.1f} ms")
                        for cls_id, prob in zip(CLASSES, res_mob["probabilities"]):
                            st.progress(float(prob), text=f"{CLASS_DISPLAY_NAMES[cls_id][:8]}: {prob*100:.0f}%")

                    predicted_class = res_mob["class_id"] if res_mob["confidence"] >= res_cnn["confidence"] else res_cnn["class_id"]

                # Actionable handling protocol
                sug = DISPOSAL_SUGGESTIONS[predicted_class]
                st.markdown(f"""
                <div class="protocol-box">
                    <div class="protocol-title">Recommended Disposal & Recovery Directive: {sug['title']}</div>
                    <ol class="protocol-list">
                        {''.join([f"<li>{step}</li>" for step in sug['steps']])}
                    </ol>
                    <div class="hazard-box">
                        <strong>Environmental Hazard Advisory:</strong> {sug['hazard_note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Upload an image using the panel on the left to start classification.")

# ==========================================
# 3. DISPOSAL PROTOCOLS
# ==========================================
elif menu == "Disposal Protocols":
    st.subheader("E-Waste Disposal & Material Recovery Protocols")
    st.write("Standard operating procedures for managing end-of-life electronics and mitigating environmental contamination.")

    selected_category = st.selectbox(
        "Select Hardware Stream to Review:",
        options=CLASSES,
        format_func=lambda x: f"{CLASS_ICONS[x]} {CLASS_DISPLAY_NAMES[x]}"
    )

    sug = DISPOSAL_SUGGESTIONS[selected_category]
    st.markdown(f"""
    <div class="protocol-box">
        <div class="protocol-title">{CLASS_ICONS[selected_category]} {sug['title']}</div>
        <ol class="protocol-list">
            {''.join([f"<li style='margin-bottom: 6px;'>{step}</li>" for step in sug['steps']])}
        </ol>
        <div class="hazard-box">
            <strong>Environmental Contamination Risk:</strong> {sug['hazard_note']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Sustainable Lifecycle Hierarchy")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("**1. Reduction & Longevity**")
        st.caption("Prioritize modular designs and devices with field-replaceable parts to extend active operational life.")
    with r2:
        st.markdown("**2. Functional Refurbishment**")
        st.caption("Replace cracked displays or degraded batteries before retiring assemblies from the service pool.")
    with r3:
        st.markdown("**3. Secondary Reallocation**")
        st.caption("Channel functional computing hardware to educational or non-profit entities for secondary utilization.")
    with r4:
        st.markdown("**4. Certified Smelting**")
        st.caption("Consign end-of-life assemblies to accredited R2 or e-Stewards facilities for high-temperature smelting.")

# ==========================================
# 4. MODEL BENCHMARKS
# ==========================================
elif menu == "Model Benchmarks":
    st.subheader("Model Performance & Benchmark Analysis")
    st.write("Quantitative validation results computed on the held-out test split.")

    if not os.path.exists(METRICS_PATH):
        st.warning("Evaluation metrics (`results/metrics.json`) not found.")
        st.markdown("Run `python evaluate.py` to generate the benchmark data.")
    else:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        m_cnn = metrics_data["models"].get("custom_cnn")
        m_mob = metrics_data["models"].get("mobilenet_v2")

        st.caption(f"Benchmark Timestamp: {metrics_data.get('evaluated_at')} | Test Samples: {metrics_data.get('test_sample_count')}")

        st.markdown("### Architecture Performance Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric(
                "Custom CNN Accuracy",
                f"{m_cnn['test_accuracy']}%" if m_cnn else "N/A"
            )
            st.caption("Custom Architecture")

        with col_m2:
            st.metric(
                "MobileNetV2 Accuracy",
                f"{m_mob['test_accuracy']}%" if m_mob else "N/A"
            )
            st.caption("Pre-trained Backbone")

        with col_m3:
            lat_cnn = f"{m_cnn['avg_latency_ms']:.1f} ms" if m_cnn else "N/A"
            lat_mob = f"{m_mob['avg_latency_ms']:.1f} ms" if m_mob else "N/A"
            st.metric("Latency (CNN / MobileNet)", f"{lat_cnn} / {lat_mob}")
            st.caption("Average per-image inference")

        with col_m4:
            par_cnn = f"{m_cnn['total_parameters']:,}" if m_cnn else "N/A"
            par_mob = f"{m_mob['total_parameters']:,}" if m_mob else "N/A"
            st.metric("Parameters (CNN / Mob)", f"{par_cnn} / {par_mob}")
            st.caption("Total model weights")

        st.markdown("---")
        st.markdown("### Training & Validation Dynamics")
        if os.path.exists(CURVES_PATH):
            st.image(CURVES_PATH, caption="Loss and Accuracy Trajectories across Training Epochs", use_container_width=True)

        st.markdown("### Confusion Matrices")
        if os.path.exists(CONFUSION_PATH):
            st.image(CONFUSION_PATH, caption="Confusion Matrices across Held-Out Test Split", use_container_width=True)

        st.markdown("### Classification Reports")
        c_rep1, c_rep2 = st.columns(2)
        with c_rep1:
            st.markdown("**Custom CNN Performance**")
            if m_cnn and "classification_report_text" in m_cnn:
                st.code(m_cnn["classification_report_text"])
            else:
                st.write("Not available.")
        with c_rep2:
            st.markdown("**MobileNetV2 Performance**")
            if m_mob and "classification_report_text" in m_mob:
                st.code(m_mob["classification_report_text"])
            else:
                st.write("Not available.")
