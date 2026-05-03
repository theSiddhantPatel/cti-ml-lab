import streamlit as st
import json
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from nsl_kdd import (
    ATTACK_TO_CATEGORY,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    Net,
    RAW_COLUMNS,
    prepare_datasets,
)

st.set_page_config(page_title="CTI Dashboard", layout="wide")

PALETTE = {
    "paper": "#0f1115",
    "panel": "#171b22",
    "panel_alt": "#1e242d",
    "ink": "#efe6d6",
    "muted": "#b6ab98",
    "line": "#53483b",
    "accent": "#c89b5d",
    "accent_dark": "#f1d3a6",
    "accent_soft": "#34281f",
    "success": "#68c08f",
    "warning": "#d0a25f",
    "danger": "#d88676",
}

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["panel"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["line"],
        "axes.labelcolor": PALETTE["muted"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["ink"],
        "font.family": "serif",
    }
)

st.markdown(
    f"""
    <style>
    :root {{
        --paper: {PALETTE["paper"]};
        --panel: {PALETTE["panel"]};
        --panel-alt: {PALETTE["panel_alt"]};
        --ink: {PALETTE["ink"]};
        --muted: {PALETTE["muted"]};
        --line: {PALETTE["line"]};
        --accent: {PALETTE["accent"]};
        --accent-dark: {PALETTE["accent_dark"]};
        --accent-soft: {PALETTE["accent_soft"]};
    }}
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(200, 155, 93, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(120, 94, 59, 0.14), transparent 22%),
            linear-gradient(180deg, #111419 0%, var(--paper) 48%, #0b0d11 100%);
        color: var(--ink);
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1220px;
    }}
    h1, h2, h3 {{
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: 0.01em;
    }}
    p, label, .stMarkdown, .stCaption {{
        color: var(--ink);
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(36, 41, 50, 0.92), rgba(24, 28, 35, 0.96));
        border: 1px solid var(--line);
        padding: 0.9rem 1rem;
        border-radius: 18px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
    }}
    div[data-testid="stMetricLabel"] {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--ink);
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stFileUploaderDropzone"] {{
        background: rgba(23, 27, 34, 0.94);
        border-color: var(--line);
        color: var(--ink);
    }}
    div[data-testid="stFileUploader"] {{
        background: rgba(23, 27, 34, 0.88);
        border-radius: 16px;
        padding: 0.4rem;
        border: 1px dashed var(--line);
    }}
    .hero-card {{
        background: linear-gradient(135deg, rgba(29, 33, 41, 0.98), rgba(18, 22, 28, 0.98));
        border: 1px solid rgba(200, 155, 93, 0.24);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.34);
        position: relative;
        overflow: hidden;
    }}
    .hero-card::after {{
        content: "";
        position: absolute;
        inset: auto -6% -30% auto;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(200,155,93,0.18), transparent 68%);
        pointer-events: none;
    }}
    .eyebrow {{
        color: var(--accent-dark);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }}
    .hero-title {{
        font-size: 2.5rem;
        line-height: 1.1;
        margin: 0;
        max-width: 720px;
    }}
    .hero-copy {{
        margin-top: 0.8rem;
        color: var(--muted);
        font-size: 1rem;
        max-width: 760px;
    }}
    .hero-meta {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.8rem;
        margin-top: 1.2rem;
    }}
    .meta-item {{
        background: rgba(24, 28, 35, 0.86);
        border: 1px solid rgba(200, 155, 93, 0.18);
        border-radius: 16px;
        padding: 0.9rem 1rem;
    }}
    .meta-label {{
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    .meta-value {{
        margin-top: 0.25rem;
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.45;
    }}
    .section-card {{
        background: rgba(23, 27, 34, 0.94);
        border: 1px solid rgba(200, 155, 93, 0.18);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(8px);
    }}
    .section-kicker {{
        color: var(--accent-dark);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .section-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }}
    .section-copy {{
        color: var(--muted);
        margin-bottom: 1rem;
    }}
    .insight-list {{
        margin: 0;
        padding-left: 1.2rem;
        color: var(--ink);
    }}
    .insight-list li {{
        margin-bottom: 0.55rem;
    }}
    .dataset-banner {{
        background: linear-gradient(90deg, rgba(200, 155, 93, 0.16), rgba(52, 40, 31, 0.56));
        border: 1px solid rgba(200, 155, 93, 0.22);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin-bottom: 1rem;
    }}
    .dataset-banner-title {{
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .dataset-banner-copy {{
        color: var(--muted);
        margin-top: 0.25rem;
        font-size: 0.95rem;
    }}
    .result-card {{
        background: rgba(31, 36, 44, 0.92);
        border: 1px solid rgba(200, 155, 93, 0.16);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .result-label {{
        color: var(--accent-dark);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .result-title {{
        color: var(--ink);
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .result-copy {{
        color: var(--muted);
        margin-bottom: 0;
    }}
    .footer-bar {{
        margin-top: 1.4rem;
        padding: 1rem 0 0.35rem 0;
        border-top: 1px solid rgba(200, 155, 93, 0.18);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }}
    .footer-left {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        flex-wrap: wrap;
    }}
    .social-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(31, 36, 44, 0.9);
        border: 1px solid rgba(200, 155, 93, 0.16);
        color: var(--ink);
        text-decoration: none;
        font-size: 0.92rem;
    }}
    .social-icon {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--accent);
        color: #111419;
        font-weight: 700;
        font-size: 0.82rem;
        font-family: Georgia, "Times New Roman", serif;
    }}
    .footer-right {{
        color: var(--muted);
        font-size: 0.92rem;
        text-align: right;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_section_intro(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def render_result_banner(title, copy):
    st.markdown(
        f"""
        <div class="dataset-banner">
            <div class="dataset-banner-title">{title}</div>
            <div class="dataset-banner-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label, title, copy):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-title">{title}</div>
            <div class="result-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Cybersecurity Research Dashboard</div>
        <h1 class="hero-title">Federated Cyber Threat Intelligence with a modern interface and a classic academic tone.</h1>
        <div class="hero-copy">
            Monitor training performance, inspect client behavior, and run intrusion predictions from one polished workspace built around the NSL-KDD workflow.
        </div>
        <div class="hero-meta">
            <div class="meta-item">
                <div class="meta-label">Institute</div>
                <div class="meta-value">Bundelkhand Institute of Engineering and Technology</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Branch</div>
                <div class="meta-value">Computer Science and Engineering</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Team</div>
                <div class="meta-value">Siddhant Patel, Ayush Sharma, Tanya Gupta, Ranjeet Kumar, Sukhdev</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 🔹 Load Model + Preprocessing
# -------------------------------
model = Net()

MODEL_PATH = "global_model_round_3.pth"
model_loaded = False

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model_loaded = True
else:
    model_loaded = False

# Load preprocessing
_, _, feature_encoders, label_encoder, scaler = prepare_datasets()


def prepare_uploaded_dataframe(uploaded_file):
    raw_df = pd.read_csv(uploaded_file)
    actual_labels = None

    if "label" in raw_df.columns:
        actual_labels = raw_df["label"].astype(str).copy()

    if set(FEATURE_COLUMNS).issubset(raw_df.columns):
        return raw_df.loc[:, FEATURE_COLUMNS].copy(), actual_labels

    if len(raw_df.columns) == len(FEATURE_COLUMNS):
        raw_df.columns = FEATURE_COLUMNS
        return raw_df.copy(), actual_labels

    uploaded_file.seek(0)
    raw_df = pd.read_csv(uploaded_file, header=None)

    if raw_df.shape[1] == len(RAW_COLUMNS):
        raw_df.columns = RAW_COLUMNS
        actual_labels = raw_df["label"].astype(str).copy()
        return raw_df.drop(columns=["label", "difficulty"]), actual_labels

    if raw_df.shape[1] == len(FEATURE_COLUMNS):
        raw_df.columns = FEATURE_COLUMNS
        return raw_df, actual_labels

    raise ValueError(
        "Unsupported CSV format. Upload a file with the 41 NSL-KDD feature columns, "
        "or the raw 43-column dataset format including label and difficulty."
    )


def encode_categorical_features(dataframe):
    encoded_df = dataframe.copy()

    for column, encoder in feature_encoders.items():
        if column not in encoded_df.columns:
            continue

        unknown_values = sorted(
            set(encoded_df[column].astype(str)) - set(encoder.classes_)
        )
        if unknown_values:
            raise ValueError(
                f"Column '{column}' contains unsupported values: {unknown_values[:5]}"
            )

        encoded_df[column] = encoder.transform(encoded_df[column].astype(str))

    return encoded_df


def normalize_uploaded_labels(labels):
    if labels is None:
        return None

    normalized = labels.str.strip().str.lower().str.rstrip(".")
    normalized = normalized.map(lambda value: ATTACK_TO_CATEGORY.get(value, value))

    invalid_labels = sorted(set(normalized) - set(CLASS_NAMES))
    if invalid_labels:
        raise ValueError(
            "Uploaded labels contain unsupported values: "
            f"{invalid_labels[:5]}. Use NSL-KDD labels or the 5 grouped class names."
        )

    return normalized


accuracy_history = []
if os.path.exists("accuracy_history.json"):
    with open("accuracy_history.json", "r") as f:
        accuracy_history = json.load(f)

latest_accuracy = accuracy_history[-1] if accuracy_history else None
trained_rounds = len(accuracy_history)

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Model status", "Ready" if model_loaded else "Missing")
with metric_col2:
    accuracy_value = (
        f"{latest_accuracy:.2%}" if latest_accuracy is not None else "Unavailable"
    )
    st.metric("Latest global accuracy", accuracy_value)
with metric_col3:
    st.metric("Tracked rounds", trained_rounds)

# -------------------------------
# 🔹 Accuracy Graph
# -------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_section_intro(
    "Training Overview",
    "Global Accuracy Across Federated Rounds",
    "Track how the shared model improves as clients contribute updates to the global training cycle.",
)
render_result_banner(
    "Global Training Dataset Results",
    "This section summarizes the overall federated learning performance collected across every completed training round.",
)

if accuracy_history:
    chart_df = pd.DataFrame(
        {
            "Round": list(range(1, len(accuracy_history) + 1)),
            "Accuracy": accuracy_history,
        }
    ).set_index("Round")
    render_result_card(
        "Result Summary",
        f"Latest global accuracy is {latest_accuracy:.2%} after {trained_rounds} rounds.",
        "Use this chart as the primary view of how the global model is improving over time.",
    )
    st.line_chart(chart_df, color=PALETTE["accent"])
else:
    st.warning("No accuracy history found")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 🔹 Client Analysis
# -------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_section_intro(
    "Client View",
    "Client-Level Evaluation",
    "Review confusion patterns and per-class performance for each participating federated client.",
)

client_id = st.selectbox("Select Client", [0, 1, 2])
file_path = f"confusion_client_{client_id}.json"
render_result_banner(
    f"Client Dataset {client_id} Results",
    f"These results belong only to client {client_id}, so users can inspect one local dataset contribution at a time.",
)

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        confusion = torch.tensor(json.load(f))

    total_samples = int(confusion.sum().item())
    diagonal_total = int(confusion.diag().sum().item())
    client_accuracy = (diagonal_total / total_samples) if total_samples else 0.0

    render_result_card(
        "Client Summary",
        f"Client {client_id} accuracy is {client_accuracy:.2%} across {total_samples} evaluated records.",
        "The confusion matrix shows where predictions concentrate, while the bar chart highlights category-wise reliability.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        sns.heatmap(
            confusion.numpy(),
            annot=True,
            fmt=".0f",
            cmap=sns.light_palette(PALETTE["accent"], as_cmap=True),
            linewidths=0.6,
            linecolor="#f6efe4",
            ax=ax,
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Client {client_id} Classification Spread", pad=12)
        st.pyplot(fig)

    with col2:
        st.subheader("Per-Class Accuracy")

        row_sums = confusion.sum(dim=1)
        correct = confusion.diag()

        class_acc = []
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                class_acc.append((correct[i] / row_sums[i]).item())
            else:
                class_acc.append(0)

        class_chart = pd.DataFrame(
            {
                "Category": CLASS_NAMES,
                "Accuracy": class_acc,
            }
        ).set_index("Category")
        st.bar_chart(class_chart, color=PALETTE["accent"])
else:
    st.info(f"No saved confusion matrix found for client {client_id}.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
#  NEW: Upload & Predict
# -------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_section_intro(
    "Prediction Desk",
    "Upload Network Data for Intrusion Detection",
    "Bring in a CSV based on NSL-KDD features to generate grouped attack predictions and optional confusion analysis.",
)
render_result_banner(
    "Uploaded Dataset Results",
    "When you upload a CSV, the dashboard will separate row-level predictions from evaluation metrics so the outcome is easy to read.",
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        prediction_df, actual_labels = prepare_uploaded_dataframe(uploaded_file)
        encoded_df = encode_categorical_features(prediction_df)

        X = scaler.transform(encoded_df.values.astype(float))
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X_tensor)
            preds = outputs.argmax(dim=1).numpy()

        predicted_labels = label_encoder.inverse_transform(preds)
        normalized_labels = normalize_uploaded_labels(actual_labels)

        result_df = prediction_df.copy()
        result_df["Predicted Label"] = predicted_labels
        if normalized_labels is not None:
            result_df["Actual Label"] = normalized_labels.values

        predicted_distribution = (
            pd.Series(predicted_labels)
            .value_counts()
            .reindex(CLASS_NAMES, fill_value=0)
        )

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Rows processed", len(result_df))
        with summary_col2:
            st.metric(
                "Ground truth supplied",
                "Yes" if normalized_labels is not None else "No",
            )

        render_result_card(
            "Uploaded Prediction Summary",
            f"Prediction output generated for {len(result_df)} rows.",
            "The table below shows row-level results, and the class distribution chart helps users quickly see the dominant predicted categories in the uploaded dataset.",
        )

        st.caption(f"Processed {len(result_df)} rows using NSL-KDD feature names.")
        result_col1, result_col2 = st.columns([1.6, 1])
        with result_col1:
            st.subheader("Predictions")
            st.dataframe(result_df, use_container_width=True)
        with result_col2:
            st.subheader("Predicted Class Distribution")
            distribution_df = pd.DataFrame(
                {
                    "Category": CLASS_NAMES,
                    "Rows": predicted_distribution.values,
                }
            ).set_index("Category")
            st.bar_chart(distribution_df, color=PALETTE["accent"])

        if normalized_labels is not None:
            st.subheader("Uploaded Data Confusion Matrix")
            conf = confusion_matrix(
                normalized_labels,
                predicted_labels,
                labels=CLASS_NAMES,
            )

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                conf,
                annot=True,
                fmt="d",
                cmap=sns.light_palette(PALETTE["accent"], as_cmap=True),
                linewidths=0.6,
                linecolor="#f6efe4",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title("Uploaded Sample Evaluation", pad=12)
            st.pyplot(fig)
        else:
            st.info(
                "Confusion matrix is available when the uploaded CSV includes a "
                "'label' column or uses the raw 43-column NSL-KDD format."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 🔹 Insights
# -------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
render_section_intro(
    "Reference Notes",
    "Class Mapping and Key Insights",
    "Keep the label groups and modeling context visible while reviewing predictions and client behavior.",
)

st.markdown(
    """
    <ul class="insight-list">
        <li><strong>Class 0</strong> maps to <strong>Normal</strong>.</li>
        <li><strong>Class 1</strong> maps to <strong>DoS</strong>.</li>
        <li><strong>Class 2</strong> maps to <strong>Probe</strong>.</li>
        <li><strong>Class 3</strong> maps to <strong>R2L</strong>.</li>
        <li><strong>Class 4</strong> maps to <strong>U2R</strong>.</li>
        <li>Rare attacks remain harder to detect because the dataset is imbalanced.</li>
        <li>Federated learning helps simulate distributed security environments where raw data stays local.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-bar">
        <div class="footer-left">
            <a class="social-pill" href="https://bietjhs.ac.in/" target="_blank">
                <span class="social-icon"></span>
                <span>BIET Jhansi</span>
            </a>
            <a class="social-pill" href="https://www.linkedin.com/" target="_blank">
                <span class="social-icon">in</span>
                <span>LinkedIn</span>
            </a>
        </div>
        <div class="footer-right">Copyright © 2026 All Rights Reserved.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
