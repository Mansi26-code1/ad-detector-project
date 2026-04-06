# 🧠 Alzheimer's Disease Detection System

Early-stage Alzheimer's detection from speech using 
multimodal AI — acoustic + linguistic analysis.

> Inspired by ADDetector — IEEE Transactions on 
> Industrial Informatics, 2022

---

## 🔴 Problem

Alzheimer's affects 55 million people globally.
Early diagnosis requires expensive clinical tests.
This system detects AD from simple speech recordings
— accessible via any smart device.

---

## ✅ Solution

Two-model multimodal approach:

| Model | Input | Recall | F1 |
|-------|-------|--------|----|
| Voting Classifier | Audio (MFCC + Chroma + ZCR) | 66% | 62% |
| TF-IDF + Logistic Regression | Speech Transcript | 83% | 79% |

**Why recall matters more than accuracy in medical AI:**
Missing an Alzheimer's patient is far costlier than
a false alarm. High recall = fewer patients missed.
---

## 🚀 Run Locally
```bash
git clone https://github.com/Mansi26-code1/ad-detector-project.git
cd ad-detector-project
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dataset

ADReSS Challenge — INTERSPEECH 2020
- 51 AD subjects
- 48 Healthy Control subjects

---

## 🛠️ Tech Stack

Python • Librosa • OpenAI Whisper • Scikit-learn
TF-IDF • Streamlit • Pickle

---

## 📖 Reference

Li et al., IEEE Transactions on Industrial 
Informatics, Vol. 18, No. 3, March 2022.

---

## 👩‍💻 Author

**Mansi Pandey** — B.Tech CSE, AKTU Lucknow

[GitHub](https://github.com/Mansi26-code1)

