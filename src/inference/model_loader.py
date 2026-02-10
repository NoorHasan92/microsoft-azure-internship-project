# src/inference/model_loader.py

import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "final_svm_model.joblib"
WORD_TFIDF_PATH = ARTIFACTS_DIR / "word_tfidf.joblib"
CHAR_TFIDF_PATH = ARTIFACTS_DIR / "char_tfidf.joblib"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    word_vectorizer = joblib.load(WORD_TFIDF_PATH)
    char_vectorizer = joblib.load(CHAR_TFIDF_PATH)

    return model, word_vectorizer, char_vectorizer
