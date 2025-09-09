# ===================================================================
# 🧠 Seizure Prediction App (Redesigned Frontend)
# ===================================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os

# -------------------
# Page Configuration
# -------------------
st.set_page_config(
    page_title="Seizure Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# Custom CSS for Styling
# -------------------
st.markdown("""
<style>
/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}

/* Card-like containers */
.st-emotion-cache-z5fcl4 {
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    transition: 0.3s;
}
.st-emotion-cache-z5fcl4:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

/* Style the buttons */
.stButton>button {
    border-radius: 20px;
    border: 1px solid #007bff;
    background-color: #007bff;
    color: white;
    padding: 10px 24px;
    transition-duration: 0.4s;
}
.stButton>button:hover {
    background-color: white;
    color: #007bff;
}

/* Custom titles */
h1, h2, h3 {
    color: #003366; /* A deep blue color */
}
</style>
""", unsafe_allow_html=True)


# -------------------
# Sidebar Content
# -------------------
with st.sidebar:
    st.title("🧠 Neuro-Predict")
    st.header("⚙️ Controls")
    model_variant = st.selectbox("Select Model Architecture", ["CNN+LSTM", "CNN+GRU"])
    threshold = st.slider("Seizure Detection Threshold", 0.0, 1.0, 0.5, 0.01)

    st.info(f"Using the **{model_variant}** model. A seizure will be detected if the probability is > **{threshold}**.")

    with st.expander("📋 Sample Data for Testing"):
        st.write("**Seizure Sample:**")
        seizure_sample = "382,356,331,320,315,307,272,244,232,237,258,212,2,-267,-605,-850,-1001,-1109,-1090,-967,-746,-464,-152,118,318,427,473,485,447,397,339,312,314,326,335,332,324,310,312,309,309,303,297,295,295,293,286,279,283,301,308,285,252,215,194,169,111,-74,-388,-679,-892,-949,-972,-1001,-1006,-949,-847,-668,-432,-153,72,226,326,392,461,495,513,511,496,479,453,440,427,414,399,385,385,404,432,444,437,418,392,373,363,365,372,385,388,383,371,360,353,334,303,252,200,153,151,143,48,-206,-548,-859,-1067,-1069,-957,-780,-597,-460,-357,-276,-224,-210,-350,-930,-1413,-1716,-1360,-662,-96,243,323,241,29,-167,-228,-136,27,146,229,269,297,307,303,305,306,307,280,231,159,85,51,43,62,63,63,69,89,123,136,127,102,95,105,131,163,168,164,150,146,152,157,156,154,143,129,1"
        st.code(seizure_sample, language=None)

# -------------------
# Load Model & Scaler
# -------------------
@st.cache_resource(show_spinner="Loading models, please wait...")
def load_artifacts():
    models = {}
    try:
        lstm_path = "seizure_detection_model.keras"
        gru_path = "seizure_detection_model_gru.keras"
        scaler_path = "scaler.save"

        models["CNN+LSTM"] = tf.keras.models.load_model(lstm_path, compile=False)
        models["CNN+GRU"] = tf.keras.models.load_model(gru_path, compile=False)
        scaler = joblib.load(scaler_path)
        return models, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.error("Please ensure all model and scaler files are in the same folder as this app.")
        return None, None

artifacts = load_artifacts()
if artifacts and artifacts[0] and artifacts[1]:
    all_models, scaler = artifacts
    model = all_models[model_variant]
    timesteps = 178
else:
    st.stop()

# -------------------
# Main App Layout
# -------------------
st.title("🧠 Epileptic Seizure Prediction Dashboard")
st.write("This application uses a deep learning model to predict the likelihood of an epileptic seizure from EEG signal data.")

# --- About this App Section ---
with st.expander("ℹ️ About this App"):
    st.write("""
        This dashboard is the user-facing component of a complete machine learning project for epileptic seizure detection.
        - **Technology:** The app is built in Python using Streamlit. The predictive models were built with TensorFlow/Keras.
        - **Models:** Two hybrid deep learning architectures are available: a CNN+LSTM and a CNN+GRU. Both are designed to capture spatial and temporal patterns in EEG data.
        - **Data:** The model was trained on the Epileptic Seizure Recognition dataset from the UCI Machine Learning Repository.
        - **Purpose:** To provide an intuitive interface for demonstrating the model's predictive capabilities on new data.
    """)

st.markdown("---")

col1, col2 = st.columns(2, gap="large")

# --- CSV Upload Section ---
with col1:
    with st.container():
        st.header("1️⃣ Upload CSV File")
        uploaded_file = st.file_uploader("Upload a file with 178 EEG features per row.", type="csv", label_visibility="collapsed")
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file, header=None).values.astype(np.float32)
                if data.shape[1] != timesteps:
                    st.error(f"CSV must have {timesteps} columns. Yours has {data.shape[1]}.")
                else:
                    st.success(f"Loaded {data.shape[0]} samples.")
                    if st.button("Predict from CSV"):
                        with st.spinner("Analyzing EEG data..."):
                            scaled = scaler.transform(data)
                            reshaped = scaled.reshape(-1, timesteps, 1)
                            probs = model.predict(reshaped, verbose=0)[:, 1]
                            predictions = ["⚠️ Seizure" if p > threshold else "✅ No Seizure" for p in probs]
                            
                            results_df = pd.DataFrame({'Prediction': predictions, 'Seizure Probability': probs})
                            st.subheader("🔍 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# --- Manual Input Section ---
with col2:
    with st.container():
        st.header("2️⃣ Manual Feature Input")
        manual_input = st.text_area(f"Enter {timesteps} comma-separated EEG features:", height=150, label_visibility="collapsed")
        
        if st.button("Predict from Manual Input"):
            num_features = len([x for x in manual_input.split(',') if x.strip()])
            if num_features != timesteps:
                st.error(f"You must enter exactly {timesteps} features. You entered {num_features}.")
            else:
                try:
                    with st.spinner("Analyzing EEG data..."):
                        features = np.array(manual_input.split(','), dtype=np.float32).reshape(1, -1)
                        scaled = scaler.transform(features)
                        reshaped = scaled.reshape(1, timesteps, 1)
                        prob = model.predict(reshaped, verbose=0)[0][1]
                        
                        st.subheader("🔍 Prediction Result")
                        if prob > threshold:
                            st.warning(f"**Prediction: Seizure Detected** (Probability: {prob:.4f})")
                            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGVqb2dmOXJ6bHZudHcxMnVyYW12amF5cmJldzFqem9objZzcDd2NSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/2XskdWuNUyqElkKe4bm/giphy.gif", width=150 , caption="Stay Calm! Help is on the way.")
                        else:
                            st.success(f"**Prediction: No Seizure** (Probability: {prob:.4f})")
                            st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3NmljcDNhOTJmamZmbWR3ZmJ2Y3J6NGdncmFsZzF5bnoya3M5Z2xkaiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xT8qBqNisx9dkXWAX6/giphy.gif", width=150)

                        # --- Re-added Probability Bar Chart ---
                        st.subheader("📊 Probability Breakdown")
                        fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
                        ax_prob.bar(["No Seizure", "Seizure"], [1 - prob, prob], color=['#28a745', '#ffc107'])
                        ax_prob.set_ylabel("Probability")
                        ax_prob.set_ylim(0, 1)
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)
                        # --- End of Added Section ---

                        st.subheader("📈 Input EEG Signal")
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(features.flatten(), color="#007bff")
                        ax.set_title("Manually Entered EEG Signal")
                        ax.set_xlabel("Time Step (Feature Index)")
                        ax.set_ylabel("EEG Value")
                        ax.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as e:
                    st.error(f"Error processing manual input: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Created  By Amritanshu Kumar | [GitHub]( ) | [LinkedIn](https://www.linkedin.com/in/amritanshu-kumar-507b6215a/) | [Portfolio](https://amritanshukumar.github.io/portfolio/) ")
