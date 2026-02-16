#root path: e:/Microsoft Azure/microsoft-azure-internship-project/train_symptom_model.py

import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # Default 0.5 threshold for now
    preds = (probs > 0.5).astype(int)

    return {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    }


def main():
    print("Starting Symptom Model Training...")

    #==========================
    # Load Dataset
    #==========================

    dataset_path = os.path.join("data", "raw", "symptom_dataset.json")
    data = load_dataset(dataset_path)

    texts = [item["text"] for item in data]
    emotion_lists = [item["emotions"] for item in data]

    #==========================
    # Label Processing
    #==========================

    unique_labels = sorted(list(set(e for sublist in emotion_lists for e in sublist)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    multi_hot_labels = []
    for emotions in emotion_lists:
        vector = [0] * len(unique_labels)
        for e in emotions:
            vector[label2id[e]] = 1
        multi_hot_labels.append(vector)

    #==========================
    # Train/Test Split
    #==========================

    X_train, X_test, y_train, y_test = train_test_split(
        texts, multi_hot_labels, test_size=0.2, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")

    from datasets import Features, Value, Sequence

    features = Features({
        "text": Value("string"),
        "labels": Sequence(Value("float32"))
    })

    train_dataset = Dataset.from_dict(
        {"text": X_train, "labels": y_train},
        features=features
    )

    test_dataset = Dataset.from_dict(
        {"text": X_test, "labels": y_test},
        features=features
    )


    #==========================
    # Tokenization
    #==========================

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    print("\nTokenization complete.")

    #==========================
    # Load Model
    #==========================

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(unique_labels),
        problem_type="multi_label_classification"
    )

    model.config.id2label = id2label
    model.config.label2id = label2id

    # Freeze lower transformer layers
    for param in model.distilbert.transformer.layer[:3].parameters():
        param.requires_grad = False

    print("\nModel loaded and lower layers frozen.")

    #==========================
    # Training Arguments (Transformers v5 compatible)
    #==========================

    training_args = TrainingArguments(
        output_dir="artifacts/symptom_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True
    )

    print("\nTraining arguments configured.")

    #==========================
    # Trainer
    #==========================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    #==========================
    # Train
    #==========================

    trainer.train()
    print("\nGenerating validation predictions for threshold tuning...")

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    np.save("artifacts/symptom_model/val_probs.npy", probs)
    np.save("artifacts/symptom_model/val_labels.npy", labels)

    print("Validation probabilities saved.")


    #==========================
    # Save Model
    #==========================

    trainer.save_model("artifacts/symptom_model")
    tokenizer.save_pretrained("artifacts/symptom_model")

    print("\nTraining complete. Model saved to artifacts/symptom_model")


if __name__ == "__main__":
    main()
