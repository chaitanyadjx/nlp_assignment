import json
import re
import os
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# 1. Configuration Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATA_DIR = Path("nlp_assignment")
REST_FILE = DATA_DIR / "restaurant_train.jsonl"
LAP_FILE = DATA_DIR / "laptop_train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
MODEL_DIR = Path("./aste_model")
OUTPUT_FILE = Path("predictions_v4.jsonl")

MODEL_NAME = "google/flan-t5-base"   # Not large
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128                 # generation_max_length

# 2. Data Preprocessing
def preprocess_text(text):
    text = re.sub(r'[\t\n\r\f\v]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_jsonl(filepath):
    data = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

rest_data = load_jsonl(REST_FILE)
lap_data  = load_jsonl(LAP_FILE)

for r in rest_data: r["domain"] = "restaurant"
for r in lap_data: r["domain"] = "laptop"
all_data = rest_data + lap_data

print(f"Training Samples: {len(all_data)}")

def record_to_seq2seq(record):
    text = preprocess_text(record["Text"])
    domain = record.get("domain", "")
    input_text = f"Extract aspect sentiment triplets from this {domain} review: {text}"

    triplet_strs = []
    if "Quadruplet" in record:
        for quad in record["Quadruplet"]:
            aspect  = quad.get("Aspect", "NULL")
            opinion = quad.get("Opinion", "NULL")
            va      = quad.get("VA", "5.00#5.00")
            triplet_strs.append(f"( {aspect} | {opinion} | {va} )")
    target_text = " ; ".join(triplet_strs)
    return input_text, target_text

pairs = []
for rec in all_data:
    inp, tgt = record_to_seq2seq(rec)
    pairs.append({"id": rec["ID"], "input": inp, "target": tgt, "domain": rec.get("domain", "")})

random.shuffle(pairs)
VAL_SPLIT = 0.1
split_idx = int(len(pairs) * (1 - VAL_SPLIT))
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

print(f"Train Pairs: {len(train_pairs)}, Val Pairs: {len(val_pairs)}")

# 3. Tokenization & Dataset
print("Loading Tokenizer & Model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

class ASTEDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_input_len, max_target_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_enc = self.tokenizer(
            pair["input"],
            max_length=self.max_input_len,
            truncation=True,
        )
        target_enc = self.tokenizer(
            pair["target"],
            max_length=self.max_target_len,
            truncation=True,
        )
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in target_enc["input_ids"]]
        return {
            "input_ids":      input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels":         labels,
        }

train_dataset = ASTEDataset(train_pairs, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
val_dataset   = ASTEDataset(val_pairs,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# 4. Training (Optimized for 20-30 min on RTX 2000 Ada)
training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=5,              # Sweet spot for speed vs F1
    per_device_train_batch_size=4,  # Ada handles this fine
    per_device_eval_batch_size=4,
    warmup_steps=50,
    gradient_accumulation_steps=8,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    learning_rate=5e-4,              # Faster convergence
    fp16=True,                       # ~2x speedup on Ada
    predict_with_generate=False,     # Disabled for training loop to avoid NaN eval loss
    generation_max_length=128,
    dataloader_num_workers=8,        # Exploit 36 cores
    logging_steps=20,
    save_strategy="no",              # Skip checkpointing to save time
    eval_strategy="epoch",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
trainer.save_model(str(MODEL_DIR / "final"))
tokenizer.save_pretrained(str(MODEL_DIR / "final"))
print(f"Training complete! Model saved to {MODEL_DIR / 'final'}")

# 5. Inference (Greedy Decode)
def parse_triplets(generated_text):
    triplets = []
    pattern = r"\(\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([\d.]+#[\d.]+)\s*\)"
    for match in re.finditer(pattern, generated_text):
        aspect  = match.group(1).strip()
        opinion = match.group(2).strip()
        va      = match.group(3).strip()
        try:
            v, a = va.split("#")
            v, a = float(v), float(a)
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))
            va = f"{v:.2f}#{a:.2f}"
        except:
            va = "5.00#5.00"
        triplets.append({"Aspect": aspect, "Opinion": opinion, "VA": va})
    return triplets

def predict_greedy(texts, domains=None, max_len=128, batch_size=64):
    model.eval()
    all_triplets = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_domains = domains[i : i + batch_size] if domains else ["" for _ in batch_texts]
        
        inputs = [
            f"Extract aspect sentiment triplets from this {d} review: {preprocess_text(t)}"
            for t, d in zip(batch_texts, batch_domains)
        ]
        
        encoded = tokenizer(
            inputs, max_length=MAX_INPUT_LEN, padding=True,
            truncation=True, return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_len,
                do_sample=False,     # Greedy search (no beams)
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for gen_text in decoded:
            all_triplets.append(parse_triplets(gen_text))
            
    return all_triplets

# 6. Generate Predictions on Test Set
if TEST_FILE.exists():
    print("Generating predictions on test set...")
    test_data = load_jsonl(TEST_FILE)
    test_texts = [r["Text"] for r in test_data]
    test_domains = [r.get("domain", "") for r in test_data]
    
    predictions = predict_greedy(test_texts, domains=test_domains)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec, preds in zip(test_data, predictions):
            out_rec = {"ID": rec["ID"], "Quadruplet": preds}
            f.write(json.dumps(out_rec) + "\n")
            
    print(f"Predictions saved to {OUTPUT_FILE}")
else:
    print(f"No test file found at {TEST_FILE}")

