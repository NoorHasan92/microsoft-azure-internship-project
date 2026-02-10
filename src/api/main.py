import os
import torch
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from src.api.schemas import PredictionRequest, PredictionResponse
from dotenv import load_dotenv
from google import genai

torch.set_grad_enabled(False)

# Load environment variables
load_dotenv()

# Global state
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from huggingface_hub import login, hf_hub_download

    print("🚀 Application is starting up...")

    # -------------------------
    # Hugging Face setup
    # -------------------------
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Hugging Face authenticated")
        except Exception as e:
            print(f"⚠️ Hugging Face login failed: {e}")
    else:
        print("ℹ️ Hugging Face public access (no token)")

    repo_id = "noor9292/mental-health-distilbert"

    # -------------------------
    # Device setup
    # -------------------------
    state["device"] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Gemini setup (SAFE)
    # -------------------------
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            state["gemini_client"] = genai.Client(api_key=gemini_key)
            print("✅ Gemini client initialized")
        except Exception as e:
            print(f"⚠️ Gemini init failed: {e}")
            state["gemini_client"] = None
    else:
        print("ℹ️ GEMINI_API_KEY not found, using fallback responses")
        state["gemini_client"] = None

    try:
        # -------------------------
        # Load tokenizer (CRITICAL FIX)
        # -------------------------
        state["tokenizer"] = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True
        )
        print("✅ Tokenizer loaded")

        # -------------------------
        # Load model
        # -------------------------
        #state["model"] = DistilBertForSequenceClassification.from_pretrained(
        #    repo_id,
        #)
        #state["model"].to(state["device"])
        #state["model"].eval()
        #print("✅ Model loaded (float 32)")

        # -------------------------
        # Load label encoder
        # -------------------------
        label_path = hf_hub_download(
            repo_id=repo_id,
            filename="label_encoder_bert.joblib",
            token=hf_token
        )
        state["label_encoder"] = joblib.load(label_path)
        print("✅ Label encoder loaded")

        print("✅ Model, tokenizer, and label encoder ready")

    except Exception as e:
        print(f"❌ Startup Error: {e}")
        raise e

    yield
    state.clear()


app = FastAPI(
    title="Mental Health Risk Detection API",
    description="Fine-tuned DistilBERT with optional Gemini-powered empathetic responses",
    version="4.2.0",
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


def generate_empathetic_explanation(user_text: str, risk_label: str) -> str:
    client = state.get("gemini_client")

    fallback = (
        "I hear you. You're not alone, and reaching out is a strong first step. "
        "If things feel overwhelming, consider talking to someone you trust or a professional."
    )

    if not client:
        return fallback

    prompt = f"""
    You are a compassionate Mental Health Support Assistant.
    A user has shared: "{user_text}"
    Detected Risk: {risk_label}.

    Task:
    1. Acknowledge their feelings.
    2. Offer 2 hopeful, practical steps.
    3. Close with supportive reassurance.
    Max 4 sentences.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.replace("**", "").strip()
    except Exception as e:
        print(f"⚠️ Gemini generation error: {e}")
        return fallback


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    tokenizer = state["tokenizer"]
    device = state["device"]
    label_encoder = state["label_encoder"]
    print("DEBUG: loading model with NO extra flags")

    # Load model once (CPU-safe)
    if "model" not in state:
        print("🚀 Loading model (CPU float32)...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "noor9292/mental-health-distilbert"
        )
        model.to(device)
        model.eval()
        state["model"] = model
        print("✅ Model loaded")

    model = state["model"]

    # Tokenize
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Decode
    class_names = label_encoder.classes_
    idx = int(np.argmax(probs))
    final_label = class_names[idx]
    confidence = float(probs[idx])

    if final_label == "High":
        score = 70 + (confidence * 30)
        priority = "Critical"
    elif final_label == "Moderate":
        score = 40 + (confidence * 29)
        priority = "Medium"
    else:
        score = confidence * 39
        priority = "Low"

    explanation = generate_empathetic_explanation(
        request.text, final_label
    )

    return {
        "risk_label": final_label,
        "risk_score": round(score, 2),
        "priority": priority,
        "explanation": explanation
    }
