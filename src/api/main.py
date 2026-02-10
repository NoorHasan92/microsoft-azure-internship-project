import os
import requests
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai

from src.api.schemas import PredictionRequest, PredictionResponse

# --------------------------------------------------
# Environment & global state
# --------------------------------------------------

load_dotenv()

state = {}

HF_MODEL_URL = "https://router.huggingface.co/models/noor9292/mental-health-distilbert"

# --------------------------------------------------
# Lifespan (startup / shutdown)
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application starting (HF Inference API mode)")

    # Gemini (optional)
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

    yield
    state.clear()


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Mental Health Risk Detection API",
    description="DistilBERT via Hugging Face Inference API + Gemini explanations",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "mode": "hf-inference-api"
    }

# --------------------------------------------------
# Gemini explanation helper
# --------------------------------------------------

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
        print(f"⚠️ Gemini error: {e}")
        return fallback

# --------------------------------------------------
# Prediction endpoint (HF Inference API)
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not set")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": request.text
    }

    response = requests.post(
        HF_MODEL_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error: {response.text}")

    result = response.json()
    # Expected format:
    # [
    #   {"label": "High", "score": 0.93},
    #   {"label": "Moderate", "score": 0.05},
    #   {"label": "Low", "score": 0.02}
    # ]

    top = max(result, key=lambda x: x["score"])
    final_label = top["label"]
    confidence = float(top["score"])

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
        "explanation": explanation,
    }
