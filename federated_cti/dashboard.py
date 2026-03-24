import streamlit as st
import json
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nsl_kdd import Net, prepare_datasets

st.set_page_config(page_title="CTI Dashboard", layout="wide")

st.title("Federated Cyber Threat Intelligence Dashboard")

# -------------------------------
# 🔹 Load Model + Preprocessing
# -------------------------------
model = Net()

MODEL_PATH = "global_model_round_3.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    st.success("Model loaded successfully")
else:
    st.warning("Model file not found")

# Load preprocessing
_, _, feature_encoders, label_encoder, scaler = prepare_datasets()

# -------------------------------
# 🔹 Accuracy Graph
# -------------------------------
st.subheader("Global Accuracy Over Rounds")

if os.path.exists("accuracy_history.json"):
    with open("accuracy_history.json", "r") as f:
        acc_history = json.load(f)

    st.line_chart(acc_history)
else:
    st.warning("No accuracy history found")

# -------------------------------
# 🔹 Client Analysis
# -------------------------------
st.subheader("Client Analysis")

client_id = st.selectbox("Select Client", [0, 1, 2])
file_path = f"confusion_client_{client_id}.json"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        confusion = torch.tensor(json.load(f))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion.numpy(), annot=True, fmt=".0f", cmap="Blues", ax=ax)
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

        st.bar_chart(class_acc)

# -------------------------------
#  NEW: Upload & Predict
# -------------------------------
st.subheader(" Upload Network Data for Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # df = pd.read_csv(uploaded_file)
    # st.write("Uploaded Data Preview:", df.head())

    # try:
    #     # Encode categorical columns
    #     for col, encoder in feature_encoders.items():
    #         if col in df.columns:
    #             df[col] = encoder.transform(df[col])

    #     # Scale features
    #     X = scaler.transform(df.values)

    #     X_tensor = torch.tensor(X, dtype=torch.float32)

    #     with torch.no_grad():
    #         outputs = model(X_tensor)
    #         preds = outputs.argmax(dim=1).numpy()

    #     # Map labels back
    #     predicted_labels = label_encoder.inverse_transform(preds)

    #     st.subheader(" Predictions")
    #     result_df = df.copy()
    #     result_df["Predicted Label"] = predicted_labels

    #     st.write(result_df)
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)

            # Assign columns
            columns = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
            df.columns = columns

            # Drop label + difficulty
            df = df.drop(columns=["label", "difficulty"])

            # Encode categorical
            for col, encoder in feature_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[col])

            # Convert to numeric
            X = df.values.astype(float)

            # Scale
            X = scaler.transform(X)

            # Tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Predict
            with torch.no_grad():
                outputs = model(X_tensor)
                preds = outputs.argmax(dim=1).numpy()

            predicted_labels = label_encoder.inverse_transform(preds)

            df["Predicted Label"] = predicted_labels
            st.write(df)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# 🔹 Insights
# -------------------------------
st.subheader(" Insights")

st.write(
    """
- Class 0 → Normal  
- Class 1 → DoS  
- Class 2 → Probe  
- Class 3 → R2L  
- Class 4 → U2R  

 Rare attacks are harder to detect due to imbalance  
 Federated learning simulates real-world distributed environments  
"""
)
