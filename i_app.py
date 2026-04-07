import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os

st.set_page_config(
    page_title="Alzheimer's Detector",
    page_icon="🧠",
    layout="centered"
)

# ─── Load Models ────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model_dir = os.path.join("notebooks", "models")
    with open(os.path.join(model_dir, "voting_model.pkl"), "rb") as f:
        audio_model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_new.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "linguistic_model.pkl"), "rb") as f:
        ling_model = pickle.load(f)
    return audio_model, scaler, ling_model

try:
    audio_model, scaler, ling_model = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run the training notebooks first "
        "to generate `notebooks/models/*.pkl` files."
    )
    st.stop()

# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_features(file_path, n_mfcc=13):
    """Extract MFCC + Chroma + ZCR from audio file."""
    y, sr = librosa.load(file_path, duration=30)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])   # pre-emphasis

    mfcc   = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr    = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

    return np.hstack([mfcc, chroma, zcr])

# ─── UI ─────────────────────────────────────────────────────────────────────

st.title("🧠 Alzheimer's Disease Detection")
st.markdown("Early detection from speech using **Audio** and **Linguistic** analysis.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis", "ℹ️ About"])

# ── Tab 1: Audio ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Speech Audio")
    st.info("Analyzes MFCC, Chroma, and Zero Crossing Rate features from the audio.")

    audio_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

    if audio_file and st.button("Analyze Audio", type="primary"):
        with st.spinner("Extracting audio features..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            try:
                features = extract_features(tmp_path)
            finally:
                os.unlink(tmp_path)

            # Clip features to scaler's expected size
            n_expected = scaler.n_features_in_
            features = features[:n_expected]
            features_scaled = scaler.transform(features.reshape(1, -1))

            pred = audio_model.predict(features_scaled)[0]
            prob = audio_model.predict_proba(features_scaled)[0]

        st.divider()
        if pred == 1:
            st.error(f"🔴 **Alzheimer's Indicators Detected**\n\nConfidence: {prob[1]*100:.1f}%")
        else:
            st.success(f"🟢 **Healthy Speech Patterns**\n\nConfidence: {prob[0]*100:.1f}%")

        st.caption("⚠️ This is a research tool only and not a medical diagnosis.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", "Voting Classifier")
        col2.metric("Recall (AD)", "76%")
        col3.metric("F1 Score", "82%")

# ── Tab 2: Text ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Enter Speech Transcript")
    st.info("Analyzes linguistic patterns — word choice, repetition, sentence structure.")

    text_input = st.text_area(
        "Paste or type the patient's speech transcript",
        height=150,
        placeholder="e.g. The woman is washing the dishes and the boy is stealing cookies..."
    )

    if st.button("Analyze Text", type="primary"):
        if text_input.strip():
            pred = ling_model.predict([text_input])[0]
            prob = ling_model.predict_proba([text_input])[0]

            st.divider()
            if pred == 1:
                st.error(f"🔴 **Alzheimer's Indicators Detected**\n\nConfidence: {prob[1]*100:.1f}%")
            else:
                st.success(f"🟢 **Healthy Speech Patterns**\n\nConfidence: {prob[0]*100:.1f}%")

            st.caption("⚠️ This is a research tool only and not a medical diagnosis.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Model", "TF-IDF + LR")
            col2.metric("Recall (AD)", "90%")
            col3.metric("F1 Score", "81%")
        else:
            st.warning("Please enter some text to analyze.")

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Problem:** Alzheimer's affects 55 million people globally. Early diagnosis requires
    expensive clinical tests. This system detects AD from simple speech recordings.

    **Approach:** Two independent models analyse different signals from speech:

    | Model | Input | Recall | F1 |
    |-------|-------|--------|----|
    | Voting Classifier (LR + RF) | Audio features (MFCC, Chroma, ZCR) | 76% | 82% |
    | TF-IDF + Logistic Regression | Speech transcript | 83% | 79% |

    **Why recall?** In medical AI, missing an actual Alzheimer's patient (false negative)
    is more costly than a false alarm. High recall = fewer patients missed.

    **Dataset:** ADReSS Challenge — INTERSPEECH 2020 (51 AD, 48 HC subjects)

    **Reference:** Li et al., *IEEE Transactions on Industrial Informatics*, Vol. 18, No. 3, 2022.

    **Author:** Mansi Pandey — B.Tech CSE, AKTU Lucknow
    """)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Guide")
    st.markdown("""
    **Audio tab:** Upload a `.wav` speech recording.

    **Text tab:** Paste a transcript of what the patient said.

    **Best results:** Use the ADReSS Cookie Theft picture description task.
    """)
    st.divider()
    st.caption("Research prototype | Not for clinical use")