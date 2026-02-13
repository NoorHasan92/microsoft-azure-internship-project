#src/inference/symptom_model.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class SymptomModel:
    def __init__(self, model_path="artifacts/symptom_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Calibrated thresholds
        self.thresholds = [
            0.45,  # anger
            0.45,  # brain dysfunction
            0.35,  # emptiness
            0.50,  # hopelessness
            0.35,  # loneliness
            0.60,  # sadness
            0.50,  # suicide intent
            0.30   # worthlessness
        ]

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        detected = []
        probabilities = {}

        for i, label in self.model.config.id2label.items():
            prob = float(probs[i])
            probabilities[label] = prob

            if prob > self.thresholds[i]:
                detected.append(label)

        return {
            "detected": detected,
            "probabilities": probabilities
        }
