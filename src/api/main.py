#src/api/main.py

import os
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from src.inference.symptom_model import SymptomModel
from src.api.schemas import PredictionRequest, PredictionResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# --------------------------------------------------
# Environment & global state
# --------------------------------------------------

load_dotenv()

state = {}

# --------------------------------------------------
# Load Risk Model (Manual Inference)
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

risk_tokenizer = AutoTokenizer.from_pretrained("artifacts/risk_model_v1")
risk_model = AutoModelForSequenceClassification.from_pretrained("artifacts/risk_model_v1")

risk_model.to(device)
risk_model.eval()


# --------------------------------------------------
# Load Symptom Model
# --------------------------------------------------

symptom_model = SymptomModel()

# --------------------------------------------------
# Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application starting (LOCAL model inference mode)")

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
    description="DistilBERT (Local Inference) + Gemini explanations",
    version="6.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="ui"), name="static")
from fastapi.responses import FileResponse

@app.get("/")
def serve_ui():
    return FileResponse("ui/index.html")


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

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "local-model-inference"}

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
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.replace("**", "").strip()
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return fallback

# --------------------------------------------------
# Prediction endpoint (LOCAL MODEL)
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    # ---------------------------
    # Risk Model Inference (Manual)
    # ---------------------------
    try:
        inputs = risk_tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = risk_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        probabilities = probs.cpu().numpy()[0]

        predicted_class_id = int(torch.argmax(probs, dim=1))
        confidence = float(probabilities[predicted_class_id])

        final_label = risk_model.config.id2label[predicted_class_id]

        print("RISK PROBS:", probabilities)
        print("Predicted:", final_label, confidence)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk model inference failed: {e}"
        )

    # ------------------------------------------
    # Symptom Profiling
    # ------------------------------------------

    if final_label == "Low" and confidence > 0.65:
        symptom_results = {
            "detected": [],
            "probabilities": {}
    }
    else:
        symptom_results = symptom_model.predict(request.text)

    detected_symptoms = []
    for label in symptom_results["detected"]:
        detected_symptoms.append({
            "label": label,
            "confidence": round(symptom_results["probabilities"][label], 4)
        })


    # ------------------------------------------
    # Symptom Severity Estimation
    # ------------------------------------------

    symptom_count = len(detected_symptoms)

    if symptom_count >= 5:
        symptom_severity = "Severe Distress"
    elif symptom_count >= 3:
        symptom_severity = "Moderate Distress"
    elif symptom_count >= 1:
        symptom_severity = "Mild Distress"
    else:
        symptom_severity = "No Significant Distress"

    # ------------------------------------------
    # Pattern Classification (Non-Diagnostic)
    # ------------------------------------------

    # Extract just labels for pattern logic
    detected_labels = [item["label"] for item in detected_symptoms]
    # ------------------------------------------
    # Strong Suicide Phrase Override
    # ------------------------------------------

    high_intent_phrases = [
        "kill myself",
        "end my life",
        "suicide tonight",
        "want to die tonight",
        "no reason to live",
        "i am going to kill myself",
        "i will kill myself"
    ]
    text_lower = request.text.lower()
    strong_phrase_detected = any(
        phrase in text_lower for phrase in high_intent_phrases
    )

    if "suicide intent" in detected_labels:
        pattern = "Crisis-Level Pattern"
    elif "hopelessness" in detected_labels and "sadness" in detected_labels:
        pattern = "Depressive Pattern"
    elif "anger" in detected_labels:
        pattern = "Emotional Regulation Pattern"
    else:
        pattern = "General Emotional Distress Pattern"


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

    # ------------------------------------------
    # Suicide Escalation Logic
    # ------------------------------------------

    suicide_prob = symptom_results["probabilities"].get("suicide intent", 0)

    if strong_phrase_detected:
        final_label = "High"
        priority = "Critical"

    elif suicide_prob >= 0.75:
        final_label = "High"
        priority = "Critical"

    elif 0.55 <= suicide_prob < 0.75 and final_label != "High":
        priority = "Medium"


    # ------------------------------------------
    # System Confidence Level
    # ------------------------------------------

    if confidence >= 0.80:
        system_confidence = "High"
    elif confidence >= 0.60:
        system_confidence = "Moderate"
    else:
        system_confidence = "Low"

    # ------------------------------------------
    # Crisis Auto-Trigger
    # ------------------------------------------

    emergency_support = None

    if priority == "Critical":
        emergency_support = {
            "india_helpline": "9152987821 (Kiran Mental Health Helpline)",
            "us_helpline": "988 (Suicide & Crisis Lifeline)",
            "message": "If you are in immediate danger, please contact local emergency services."
        }

    disclaimer = (
        "This system provides AI-assisted screening and is not a medical diagnosis. "
        "If you are experiencing distress, consider speaking with a qualified professional."
    )


    return {
        "risk_label": final_label,
        "risk_score": round(score, 2),
        "priority": priority,
        "system_confidence": system_confidence,
        "detected_symptoms": detected_symptoms,
        "symptom_severity": symptom_severity,
        "psychological_pattern": pattern,
        "explanation": explanation,
        "disclaimer": disclaimer,
        "emergency_support": emergency_support,
    }



