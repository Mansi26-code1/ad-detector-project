# 🧠 Alzheimer's Disease Detection 

> Early-stage Alzheimer's detection using multimodal AI — acoustic + linguistic analysis of speech.  
> Inspired by: *Li et al., IEEE Transactions on Industrial Informatics, Vol. 18, No. 3, 2022*

---

## Problem

Alzheimer's disease affects **55 million people** globally. Early diagnosis typically requires expensive neuropsychological tests and specialist visits. This project explores whether a **simple speech recording** — something anyone can do on a smartphone — carries enough signal to flag early-stage AD.

---

## Solution

Two independent models analyse different aspects of speech:

| Model | Input | Recall | F1 | Notes |
|-------|-------|--------|----|-------|
| Voting Classifier (Random Forest + Logistic Regression) | Audio — MFCC, Chroma, ZCR | HC: 0.93 / AD: 0.76 | HC: 0.88 / AD: 0.82 | File-level acoustic features, 173 samples |
| TF-IDF + Logistic Regression | Speech transcript | 0.90 ± 0.06 | 0.81 ± 0.02 | 5-Fold CV on text features (Whisper ASR → NLP pipeline) |
| Combined Model | Audio + Text | HC: 0.82 / AD: 0.98 | HC: 0.88 / AD: 0.96 | 80 samples; final combined audio + linguistic probabilities |

**Other Metrics for Combined Model:**

Accuracy : 0.94
Macro avg : Recall: 0.90 | F1: 0.92
Weighted avg : Recall: 0.94 | F1: 0.94


**Why recall over accuracy?**  
In medical AI, a missed Alzheimer's patient (false negative) is far costlier than a false alarm. High recall = fewer patients slipping through undetected.

**Why TF-IDF and not BERT?**  
BERT was tested but led to overfitting on this small dataset (99 subjects). TF-IDF with bigrams generalised better — a known finding on small clinical NLP datasets.

---

## Demo
## 🚀 How to Run the Project

Follow these steps to run the Alzheimer Detection project on your system:

```bash
git clone https://github.com/Mansi26-code1/ad-detector-project.git
cd ad-detector-project
pip install -r requirements.txt
streamlit run app.py
```
---
## 📁 Project Structure

ad-detector-project/
├── app.py                          # Streamlit web app
├── requirements.txt
├── notebooks/
│   ├── 01_data_train.ipynb         # Feature extraction + audio model training
│   ├── linguistic_model.ipynb      # Whisper ASR + TF-IDF + LR training
│   ├── combined_model.ipynb        # Combined model evaluation
│   ├── models/
│   │   ├── voting_model.pkl        # Trained audio model
│   │   ├── scaler_new.pkl          # Feature scaler
│   │   └── linguistic_model.pkl    # Trained text pipeline
│   └── improved_features_with_id.csv
└── data/                           # Not committed — see Dataset section
    ├── alzheimer_audio_info.xlsx
    └── audio/
---                    
## 📊 Dataset

**ADReSS Challenge — INTERSPEECH 2020**

- **51 Alzheimer's Disease (AD) subjects**
- **48 Healthy Control (HC) subjects**
- **Task:** Cookie Theft picture description (standard neuropsychological test)
---
## 🛠 Tech Stack

**Python** · **Librosa** · **OpenAI Whisper** · **Scikit-learn** · **TF-IDF** ·  
**Pandas / Numpy** · **Imbalanced-learn (SMOTE)** · **Streamlit** · **Pickle**
---
## 🧩 Key Design Decisions

- **GroupShuffleSplit for audio model:**  
  Multiple recordings exist per subject. Without group-aware splitting, the same speaker could appear in both train and test — the model would learn *speaker identity*, not *disease patterns*.

- **Pre-emphasis filter:**  
  Applied before feature extraction to boost high-frequency components that carry crucial speech information.

- **class_weight='balanced':**  
  Used in all classifiers because the dataset has mild imbalance (51 AD vs 48 HC).

---

## ⚠️ Limitations

- **Small dataset (99 subjects)** — results should be validated on larger, more diverse cohorts.  
- **Model trained only on Cookie Theft task** — not tested on free or conversational speech.

---

## 🚀 Future Improvements

- **Fine-tune ClinicalBERT** on a larger Alzheimer's transcript dataset.  
- **Add speaker diarisation** to separate patient vs interviewer audio.  
- **Collect multi-center data** via federated learning to improve generalization.

---

## 📚 Reference

Li, J. et al. (2022). *An Alzheimer's Disease Detection Method Based on Combining Audio Features and Linguistic Information.*  
IEEE Transactions on Industrial Informatics, 18(3).

---

## 👩‍💻 Author

**Mansi Pandey** — B.Tech CSE, AKTU Lucknow

