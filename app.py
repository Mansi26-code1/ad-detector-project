import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os

st.set_page_config(
    page_title="Alzheimer's Detector",
    page_icon="🧠"
)

@st.cache_resource
def load_models():
    with open("notebooks/models/voting_model.pkl", "rb") as f:
        audio_model = pickle.load(f)
    with open("notebooks/models/scaler_new.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("notebooks/models/linguistic_model.pkl", "rb") as f:
        ling_model = pickle.load(f)
    return audio_model, scaler, ling_model

audio_model, scaler, ling_model = load_models()

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mfcc = np.mean(librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        y=y, sr=sr).T, axis=0)
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y).T,
        axis=0)
    return np.hstack([mfcc, chroma, zcr])

st.title("🧠 Alzheimer's Disease Detection")
st.markdown("Multimodal detection using **Audio** and **Linguistic** analysis")
st.divider()

tab1, tab2 = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis"])

with tab1:
    st.subheader("Upload Speech Audio")
    st.info("Model analyzes MFCC, Chroma, and Zero Crossing Rate features")
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])

    if audio_file and st.button("Analyze Audio"):
        with st.spinner("Extracting features..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            features = extract_features(tmp_path)
            os.unlink(tmp_path)
            n = scaler.n_features_in_
            features = features[:n]
            features_scaled = scaler.transform(features.reshape(1, -1))
            pred = audio_model.predict(features_scaled)[0]
            prob = audio_model.predict_proba(features_scaled)[0]

        st.divider()
        if pred == 1:
            st.error(f"🔴 Alzheimer's Detected\n\nConfidence: {prob[1]*100:.1f}%")
        else:
            st.success(f"🟢 Healthy\n\nConfidence: {prob[0]*100:.1f}%")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", "Voting Classifier")
        col2.metric("Recall", "66%")
        col3.metric("F1 Score", "62%")

with tab2:
    st.subheader("Enter Speech Transcript")
    st.info("Model analyzes linguistic patterns in speech")
    text_input = st.text_area(
        "Paste speech transcript here",
        height=150,
        placeholder="Enter what the patient said..."
    )

    if st.button("Analyze Text"):
        if text_input:
            pred = ling_model.predict([text_input])[0]
            prob = ling_model.predict_proba([text_input])[0]

            st.divider()
            if pred == 1:
                st.error(f"🔴 Alzheimer's Detected\n\nConfidence: {prob[1]*100:.1f}%")
            else:
                st.success(f"🟢 Healthy\n\nConfidence: {prob[0]*100:.1f}%")

            col1, col2, col3 = st.columns(3)
            col1.metric("Model", "TF-IDF + LR")
            col2.metric("Recall", "83%")
            col3.metric("F1 Score", "79%")
        else:
            st.warning("Please enter text")

with st.sidebar:
    st.header("About Project")
    st.markdown("""
    **Disease:** Alzheimer's Detection
    
    **Modalities:**
    - 🎤 Audio — MFCC, Chroma, ZCR
    - 📝 Text — TF-IDF + Logistic Regression
    
    **Models:**
    - Voting Classifier (Audio)
    - TF-IDF + LR (Linguistic)
    
    **Dataset:** ADReSS Challenge
    INTERSPEECH 2020
    
    **Reference:** IEEE Transactions
    Industrial Informatics 2022
    """)

    