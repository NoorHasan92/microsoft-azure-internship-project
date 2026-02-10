---
title: Mental Health Api
sdk: docker
emoji: 🧠
colorFrom: blue
colorTo: green
---

🧠 AI-Based Mental Health Risk Detection System
Microsoft Azure Internship Project
📌 Project Overview
This project implements an AI-driven mental health risk detection system that analyzes textual data to identify early indicators of risks such as stress, anxiety, and depression.

By leveraging Transformer-based Deep Learning (DistilBERT) and Generative AI (Gemini 2.5), the system functions as a high-precision decision-support tool to prioritize individuals who may require timely attention.

-----------------------------------------------------------------------------------------------------------

🎯 Key Objectives
    -> High-Precision Detection: Identify mental health risk levels with Transformer-based accuracy.
    -> Explainable AI (XAI): Provide empathetic, natural-language justifications for every risk flag.
    -> Privacy by Design: Focus on anonymized, authorized text processing.
    -> Urgency Prioritization: Map model confidence to actionable priority levels.

-----------------------------------------------------------------------------------------------------------

🧩 Problem Statement
Mental health challenges often go unnoticed due to stigma and limited care access. However, digital text—such as journal entries or messages—often contains early warning signs. This project bridges the gap by providing a scalable, transparent way to flag high-risk content for human review.

-----------------------------------------------------------------------------------------------------------

🛠️ System Architecture

The pipeline utilizes a hybrid approach combining Natural Language Understanding (NLU) and Generative AI:
    1. Text Input: User-provided text (e.g., "I feel overwhelmed...").
    2.Inference (DistilBERT): A fine-tuned DistilBERT model classifies the text into Low, Moderate, or High risk.
    3.Risk Scoring: Softmax probabilities are mapped to a 0–100 scale for granular prioritization.
    4. Explainability (Gemini 2.5 Flash Lite): The model's prediction is passed to Gemini to generate a compassionate, context-aware explanation.
    5.Output: A structured JSON response containing the label, score, priority, and AI explanation.

-----------------------------------------------------------------------------------------------------------

🤖 Machine Learning Approach
    -> Primary Model: DistilBERT (Fine-tuned for sequence classification).
    -> Explainability Engine: Google Gemini 2.5 Flash Lite.
    -> Optimization: Mixed-precision training (fp16) for efficient GPU utilization (NVIDIA RTX 3050).

📊 Model Performance:

Metric                        Value
Model               TypeDistilBERT (Transformer)
Training Epochs                 2
Training Precision         FP16 (Mixed)
Build Status        ✅ Passed (Flake8 Clean)

-----------------------------------------------------------------------------------------------------------

🚦 Risk Scoring & Prioritization:
The system maps model confidence to urgency levels to assist triage.

Risk Score	        Priority	        Action Level
85–100	            Critical	    Immediate intervention
70–84	              High	            Priority review
40–69	             Medium	          Routine monitoring
<40	                  Low	           General awareness

-----------------------------------------------------------------------------------------------------------

🔍 Explainability & Ethics
    -> Generative Explanations: Unlike "Black Box" models, our system uses Gemini 2.5 Flash Lite to explain why a certain risk was detected, helping human reviewers understand the user's emotional context.
    -> Non-Diagnostic: This tool is an Early Warning System, not a medical diagnosis.
    -> Safety First: High-confidence crisis triggers automatically escalate the priority to "Critical."

-----------------------------------------------------------------------------------------------------------

📂 Project Structure
microsoft-azure-internship-project/
│
├── artifacts/             # Trained DistilBERT model & label encoders
├── data/                  # Datasets (Raw & Processed)
├── src/
│   └── api/
│       ├── main.py        # FastAPI Production Server
│       └── schemas.py     # Pydantic data models
├── ui/                    # Frontend (HTML/JS/CSS)
├── requirements.txt       # Optimized production dependencies
├── train_distilbert.py    # Model training & fine-tuning script
└── README.md              # Project documentation

-----------------------------------------------------------------------------------------------------------

🚀 Quick Start
    1. Install Dependencies: pip install -r requirements.txt
    2. Setup Environment: Create a .env file and add your GEMINI_API_KEY.
    3. Run Production Server: uvicorn src.api.main:app --workers 4

-----------------------------------------------------------------------------------------------------------

👤 Author Mahi - B.Tech CSE Student Microsoft Azure Internship Project

Disclaimer: This project is for educational purposes and is not a substitute for professional clinical help.