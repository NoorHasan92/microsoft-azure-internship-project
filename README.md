# 🧠 AI-Based Mental Health Risk Detection System

Microsoft Azure Internship Project

------------------------------------------------------------------------

## 📌 Project Overview

This project implements a multi-stage AI system for detecting emotional
distress risk from textual input.

The system leverages:

-   🧠 DistilBERT (Transformer-based NLP models)
-   🤖 Google Gemini 2.5 Flash for empathetic explanation
-   🛡️ Multi-layer crisis escalation logic
-   📊 Risk scoring & prioritization engine

It is designed as a decision-support system, not a diagnostic tool, to
help prioritize potentially high-risk content for timely attention.

------------------------------------------------------------------------

# 🎯 Key Objectives

-   High-Precision Risk Detection (Low / Moderate / High)
-   Multi-Label Emotional Profiling
-   Crisis Escalation & Safety Overrides
-   Explainable AI (Natural-language justifications)
-   Privacy-First Architecture

------------------------------------------------------------------------

## 📚 Datasets Used

1. Mental Health Sentiment Dataset  
   https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health  

2. Suicide Detection Dataset  
   https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch  

3. Reddit-Based Multi-Label Emotion Dataset  
   https://github.com/abuBakarSiddiqurRahman/DepressionEmo/blob/main/Dataset/test.json

------------------------------------------------------------------------

# 🛠️ System Architecture

## 1️⃣ Text Input

User provides free-form text.

## 2️⃣ Risk Classification (DistilBERT -- 3 Class)

Predicts Low, Moderate, or High risk using softmax probabilities.

## 3️⃣ Multi-Label Symptom Detection

Detects emotional indicators such as hopelessness, sadness, suicide
intent, anger, loneliness, worthlessness, emptiness, and cognitive
dysfunction.

## 4️⃣ Suicide Escalation Logic

Includes strong phrase overrides and probability-based crisis
escalation.

## 5️⃣ Risk Scoring Engine

  Risk Label   Score Formula
  ------------ ------------------------
  High         70 + (confidence × 30)
  Moderate     40 + (confidence × 29)
  Low          confidence × 39

## 6️⃣ Explainability Layer

Gemini generates compassionate, supportive explanations.

------------------------------------------------------------------------

# 📊 Model Overview

-   Primary Model: DistilBERT (Fine-tuned, 3-class classification)
-   Symptom Model: DistilBERT (Multi-label classification)
-   Optimization: Mixed Precision (FP16)
-   Hardware Used: NVIDIA RTX 3050

------------------------------------------------------------------------

# 🚦 Risk Prioritization

  Risk Score   Priority   Action Level
  ------------ ---------- ---------------------
  85--100      Critical   Immediate attention
  70--84       High       Priority review
  40--69       Medium     Routine monitoring
  \<40         Low        General awareness

------------------------------------------------------------------------

## 📂 Project Structure

```
microsoft-azure-internship-project/
│
├── artifacts/ (Trained Model)
├── data/
|   ├── processed/
|   └── raw/
|       ├── mental_health.csv
|       ├── Suicide_Detection.csv
|       └── symptom_dataset.json
|
├── notebooks/
|   └──01_data_exploration.ipynb
|
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   └── inference/
│       ├── symptom_model.py
|       └── model_loader.py
|
├── ui/
|   ├── index.html
|   ├── script.js
|   └── style.css
|
├── venv (Virtual Environment)
├── .env (Environment Variables, like API Keys, etc)
├── .gitignore
├── train_distilbert.py
├── train_symptom_model.py
├── tune_thresholds.py
├── requirements.txt
├── Dockerfile
└── README.md
```

------------------------------------------------------------------------

# 🚀 Quick Start

1.  Install Dependencies\
    pip install -r requirements.txt

2.  Create a .env file\
    GEMINI_API_KEY=your_key_here

3.  Run Server\
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

------------------------------------------------------------------------

# 🔍 Ethical Considerations

-   Not a medical diagnosis tool
-   Designed for early warning & triage
-   Encourages professional consultation when needed

------------------------------------------------------------------------

# 👤 Author

Md Noor Hasan Ansari
B.Tech CSE Student
Microsoft Azure Internship Project

------------------------------------------------------------------------

# 📌 Disclaimer

This project is for educational and research purposes only.\
It is not a substitute for licensed mental health professionals.
