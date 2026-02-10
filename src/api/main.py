# src/api/main.py (DistilBERT Edition)
import os
import torch
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from src.api.schemas import PredictionRequest, PredictionResponse
from dotenv import load_dotenv
from google import genai

# Load env vars
load_dotenv()

# Initialize Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Mental Health Risk Detection API",
    description="Fine-tuned DistilBERT + Gemini 2.5 Flash Lite",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
label_encoder = None


@app.on_event("startup")
def load_artifacts():
    global model, tokenizer, label_encoder

    repo_id = "noor9292/mental-health-distilbert" 
    print(f"🚀 Loading DistilBERT from Hugging Face: {repo_id}...")

    try:
        # 1. Load from Hugging Face Hub
        tokenizer = DistilBertTokenizer.from_pretrained(repo_id)
        model = DistilBertForSequenceClassification.from_pretrained(repo_id)
        model.to(device)
        model.eval()

        # 2. Load the Label Encoder file from the same repo
        from huggingface_hub import hf_hub_download
        label_file_path = hf_hub_download(repo_id=repo_id, filename="label_encoder_bert.joblib")
        label_encoder = joblib.load(label_file_path)

        print("✅ DistilBERT Loaded Successfully from the Hub!")
    except Exception as e:
        print(f"❌ Error loading: {e}")
        raise e


@app.get("/")
def health_check():
    return {"status": "ok", "model": "DistilBERT (PyTorch)"}


def generate_empathetic_explanation(user_text, risk_label):
    prompt = f"""
    You are a compassionate Mental Health Support Assistant.
    A user has shared: "{user_text}"
    Detected Risk: {risk_label}.

    Task:
    1. Acknowledge their feelings deeply.
    2. Offer 2 specific, hopeful steps.
    3. Close with support.
    Max 4 sentences.
    """
    try:
        # UPDATED: Using gemini-2.5-flash-lite
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.replace("**", "").strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        fallback = ("I hear you, and I'm here for you. Please consider "
                    "reaching out to a professional for support.")
        return fallback


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 1. Preprocess & Tokenize
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to probabilities (Softmax)
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    # 3. Label Mapping
    class_names = label_encoder.classes_  # ['High', 'Low', 'Moderate']
    max_idx = np.argmax(probs)
    final_label = class_names[max_idx]
    confidence = probs[max_idx]

    # 4. Custom Risk Scoring Logic
    if final_label == "High":
        score = 70 + (confidence * 30)
        priority = "Critical"
    elif final_label == "Moderate":
        score = 40 + (confidence * 29)
        priority = "Medium"
    else:
        score = confidence * 39
        priority = "Low"

    # 5. Get AI Explanation
    explanation = generate_empathetic_explanation(request.text, final_label)

    return {
        "risk_label": final_label,
        "risk_score": round(float(score), 2),
        "priority": priority,
        "explanation": explanation
    }
