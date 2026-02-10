import os
import torch
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.api.schemas import PredictionRequest, PredictionResponse
from dotenv import load_dotenv
from google import genai

# Load env vars
load_dotenv()

# Global State Dictionary
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern replacement for @app.on_event('startup')"""
    # --- STARTUP LOGIC ---
    print("🚀 Application is starting up...")
    
    # Initialize Gemini
    state["gemini_client"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Device setup
    state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    repo_id = "noor9292/mental-health-distilbert"
    try:
        # 1. Load Model & Tokenizer
        state["tokenizer"] = DistilBertTokenizer.from_pretrained(repo_id)
        state["model"] = DistilBertForSequenceClassification.from_pretrained(repo_id)
        state["model"].to(state["device"])
        state["model"].eval()
        print("✅ DistilBERT Model and Tokenizer loaded.")

        # 2. Load Label Encoder
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo_id=repo_id, filename="label_encoder_bert.joblib")
            state["label_encoder"] = joblib.load(path)
            print("✅ Label Encoder loaded from Hub.")
        except Exception as e:
            print(f"⚠️ Warning: Using fallback label classes ({e}).")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.classes_ = np.array(['non-suicide', 'suicide'])
            state["label_encoder"] = le

        print("✅ Startup sequence finished successfully!")
    except Exception as e:
        print(f"❌ CRITICAL STARTUP ERROR: {str(e)}")
        raise e

    yield  # The app stays here while running requests

    # --- SHUTDOWN LOGIC ---
    print("🛑 Application is shutting down...")
    state.clear()

app = FastAPI(
    title="Mental Health Risk Detection API",
    description="Fine-tuned DistilBERT + Gemini 2.5 Flash Lite",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        client = state["gemini_client"]
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.replace("**", "").strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "I hear you, and I'm here for you. Please reach out to a professional for support."

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    label_encoder = state["label_encoder"]

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
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    # 3. Label Mapping
    class_names = label_encoder.classes_
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