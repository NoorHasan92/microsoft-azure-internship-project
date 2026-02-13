# This script trains a DistilBERT model for mental health risk classification.
# /root folder

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# 1. Setup Device (Uses your RTX 3050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Load & Prepare Data
print("📂 Loading Data...")
# Ensure these paths match where your CSVs are!
df_old = pd.read_csv('data/raw/mental_health.csv')
df_new = pd.read_csv('data/raw/Suicide_Detection.csv')

# Standardize Labels
df_old = df_old.rename(columns={'statement': 'text', 'status': 'risk_label'})
old_mapping = {
    'Suicidal': 'High', 'Depression': 'Moderate', 'Anxiety': 'Moderate',
    'Stress': 'Moderate', 'Bi-Polar': 'Moderate', 'Personality Disorder': 'Moderate',
    'Normal': 'Low'
}
df_old['risk_label'] = df_old['risk_label'].map(old_mapping)
df_old = df_old[['text', 'risk_label']]

df_new = df_new.rename(columns={'class': 'risk_label'})
df_new['risk_label'] = df_new['risk_label'].map({'suicide': 'High', 'non-suicide': 'Low'})
df_new = df_new[['text', 'risk_label']]

# Combine & Balance
combined_df = pd.concat([df_old, df_new], ignore_index=True).dropna()

# Sampling 15k per class to fit in your 4GB VRAM comfortably
# If this works, you can try increasing it later!
df_low = combined_df[combined_df['risk_label'] == 'Low'].sample(n=15000, random_state=42)
df_high = combined_df[combined_df['risk_label'] == 'High'].sample(n=15000, random_state=42)
df_mod = combined_df[combined_df['risk_label'] == 'Moderate'] # Take all available (~21k)

balanced_df = pd.concat([df_low, df_mod, df_high]).sample(frac=1, random_state=42)

# Encode Labels (0, 1, 2)
le = LabelEncoder()
balanced_df['label'] = le.fit_transform(balanced_df['risk_label'])
label_map = dict(zip(le.transform(le.classes_), le.classes_))
print(f"✅ Data Loaded: {len(balanced_df)} rows. Labels: {label_map}")

# 3. Tokenization (The BERT way)
print("🔠 Tokenizing...")
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Truncation=True cuts long sentences to max 128 tokens
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert Pandas -> HuggingFace Dataset
hf_dataset = Dataset.from_pandas(balanced_df[['text', 'label']])
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# Split Train/Test
train_test = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test['train']
test_dataset = train_test['test']

# 4. Load Model
# FIX: Convert the label mapping to standard Python types (int and str) to avoid JSON errors
clean_id2label = {int(k): str(v) for k, v in label_map.items()}
clean_label2id = {str(v): int(k) for k, v in label_map.items()}

print(f"✅ Cleaned Mapping for Config: {clean_id2label}")

model = DistilBertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3, 
    id2label=clean_id2label, 
    label2id=clean_label2id
)
model.to(device)

# 5. Training Arguments (TUNED FOR RTX 3050 4GB)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Low batch size to fit in 4GB VRAM
    per_device_eval_batch_size=8,
    num_train_epochs=2,             # 2 epochs is enough for fine-tuning
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,                      # CRITICAL: Uses Mixed Precision (Saves VRAM)
    logging_steps=100,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer, 
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 7. Train!
print("🚀 Starting Training (DistilBERT)... This might take 30-45 mins.")
trainer.train()

# 8. Save
save_path = "artifacts/risk_model_v1"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Save the label encoder too (we need it for main.py)
import joblib
joblib.dump(le, os.path.join(save_path, "label_encoder_bert.joblib"))

print(f"💾 Model saved to {save_path}")