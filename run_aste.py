import json
import re
import os
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available(): pass


# ── Paths (Kaggle) ──
DATA_DIR       = Path("nlp_assignment")
REST_FILE      = DATA_DIR / "restaurant_train.jsonl"
LAP_FILE       = DATA_DIR / "laptop_train.jsonl"
MODEL_DIR      = Path("aste_t5_model")
OUTPUT_FILE    = Path("predictions.jsonl")

# ── Hyper-parameters ──
MODEL_NAME       = "t5-small"                    # 60M params
MAX_INPUT_LEN    = 256
MAX_TARGET_LEN   = 256
BATCH_SIZE       = 4
LEARNING_RATE    = 3e-4
NUM_EPOCHS       = 10
VAL_SPLIT        = 0.1                           # 10% validation


def load_jsonl(filepath):
    """Load JSONL file -> list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

rest_data = load_jsonl(REST_FILE)
lap_data  = load_jsonl(LAP_FILE)
all_data  = rest_data + lap_data

print(f"Restaurant (REST): {len(rest_data)}")
print(f"Laptop     (LAP):  {len(lap_data)}")


def record_to_seq2seq(record):
    """
    Convert a single JSONL record to (input_text, target_text) for T5.
    Category is ignored.
    """
    text = record["Text"]
    input_text = f"extract triplets: {text}"

    triplet_strs = []
    for quad in record["Quadruplet"]:
        aspect  = quad.get("Aspect", "NULL")
        opinion = quad.get("Opinion", "NULL")
        va      = quad.get("VA", "5.00#5.00")
        triplet_strs.append(f"( {aspect} | {opinion} | {va} )")

    target_text = " ; ".join(triplet_strs)
    return input_text, target_text

# Build pairs
pairs = []
for rec in all_data:
    inp, tgt = record_to_seq2seq(rec)
    pairs.append({"id": rec["ID"], "input": inp, "target": tgt})

print(f"Total seq2seq pairs: {len(pairs)}")
print()
print("Example:")
print(f"  INPUT:  {pairs[0]['input'][:120]}...")


random.shuffle(pairs)
split_idx = int(len(pairs) * (1 - VAL_SPLIT))
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

print(f"Train: {len(train_pairs)}")


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)

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

print(f"Train samples: {len(train_dataset)}")
print(f"Val   samples: {len(val_dataset)}")

sample = train_dataset[0]
print(f"input_ids shape:  {sample['input_ids'].shape}")


model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=200,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    predict_with_generate=False, # disabled to prevent NaN val loss without compute_metrics func during Trainer loop
    generation_max_length=MAX_TARGET_LEN,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Trainer ready.")
print(f"  Device:     {DEVICE}")
print(f"  FP16:       {training_args.fp16}")
print(f"  Epochs:     {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train:      {len(train_dataset)} samples")


# ══════════════════════════════════════════════════════
#  TRAINING  (GPU: ~15-30 min  |  CPU: ~2-4 hrs)
# ══════════════════════════════════════════════════════
trainer.train()


# Save the best model
trainer.save_model(str(MODEL_DIR / "best"))
tokenizer.save_pretrained(str(MODEL_DIR / "best"))


def parse_triplets(generated_text):
    """
    Parse linearized output back into structured triplets.
    Expected format: ( Aspect | Opinion | V#A ) ; ( Aspect | Opinion | V#A ) ; ...
    """
    triplets = []
    pattern = r"\(\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([\d.]+#[\d.]+)\s*\)"
    for match in re.finditer(pattern, generated_text):
        aspect  = match.group(1).strip()
        opinion = match.group(2).strip()
        va      = match.group(3).strip()

        # Clamp VA to valid range [1.00, 9.00]
        try:
            v, a = va.split("#")
            v, a = float(v), float(a)
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))
            va = f"{v:.2f}#{a:.2f}"
        except:
            va = "5.00#5.00"

        triplets.append({
            "Aspect":  aspect,
            "Opinion": opinion,
            "VA":      va,
        })
    return triplets

# Quick test
test_str = "( Indian food | average to good | 6.75#6.38 ) ; ( delivery | terrible | 2.88#6.62 )"


def predict_triplets(texts, model, tokenizer, max_len=256, batch_size=16):
    """
    Given a list of raw review texts, generate triplet predictions.
    Returns list of lists of triplet dicts.
    """
    model.eval()
    all_triplets = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = [f"extract triplets: {t}" for t in batch_texts]

        encoded = tokenizer(
            inputs,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=max_len,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for gen_text in decoded:
            triplets = parse_triplets(gen_text)
            all_triplets.append(triplets)

    return all_triplets



# Get validation texts and ground-truth
val_texts  = [p["input"].replace("extract triplets: ", "") for p in val_pairs]
val_golds  = [p["target"] for p in val_pairs]
val_ids    = [p["id"] for p in val_pairs]

# Predict
val_preds = predict_triplets(val_texts, model, tokenizer, max_len=MAX_TARGET_LEN)

# Show some examples
for i in range(min(8, len(val_preds))):
    print(f"\n{'='*70}")
    print(f"ID:    {val_ids[i]}")
    print(f"Text:  {val_texts[i][:100]}...")
    print(f"Gold:  {val_golds[i][:100]}...")


# ── Simple metrics ──
def compute_metrics(val_pairs, val_preds):
    """Compute aspect/opinion extraction F1 and VA MAE."""
    total_gold = 0
    total_pred = 0
    correct    = 0
    va_errors  = []

    for pair, preds in zip(val_pairs, val_preds):
        gold_triplets = parse_triplets(pair["target"])
        total_gold += len(gold_triplets)
        total_pred += len(preds)

        gold_set = {(g["Aspect"].lower(), g["Opinion"].lower()): g["VA"] for g in gold_triplets}
        for p in preds:
            key = (p["Aspect"].lower(), p["Opinion"].lower())
            if key in gold_set:
                correct += 1
                try:
                    gv, ga = gold_set[key].split("#")
                    pv, pa = p["VA"].split("#")
                    va_errors.append(abs(float(gv) - float(pv)) + abs(float(ga) - float(pa)))
                except:
                    pass

    prec = correct / total_pred if total_pred else 0
    rec  = correct / total_gold if total_gold else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    avg_va = np.mean(va_errors) if va_errors else float('inf')

    print(f"\n{'='*50}")
    print(f"Span Extraction:")
    print(f"  Gold: {total_gold}  Pred: {total_pred}  Correct: {correct}")
    print(f"  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(f"VA Error (matched): {avg_va:.4f}  (n={len(va_errors)})")



# ══════════════════════════════════════════════════════
#  TEST INFERENCE
# ══════════════════════════════════════════════════════

TEST_FILE = Path("nlp_assignment/test.jsonl")   # <-- UPDATE THIS

if TEST_FILE.exists():
    # Load the best model (in case kernel was restarted)
    best_path = MODEL_DIR / "best"
    if best_path.exists():
        model = T5ForConditionalGeneration.from_pretrained(str(best_path)).to(DEVICE)
        tokenizer = T5Tokenizer.from_pretrained(str(best_path), legacy=True)
        print(f"Loaded model from: {best_path}")
    else:
        print("Using model already in memory.")

    # Load test data
    test_data = load_jsonl(TEST_FILE)
    print(f"Test records: {len(test_data)}")

    test_ids   = [r["ID"] for r in test_data]
    test_texts = [r["Text"] for r in test_data]

    # Predict
    test_preds = predict_triplets(test_texts, model, tokenizer, max_len=MAX_TARGET_LEN)

    # Write output JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rid, triplets in zip(test_ids, test_preds):
            out = {"ID": rid, "Triplet": triplets}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\n✅ Predictions written to: {OUTPUT_FILE}")
    print(f"   Total records: {len(test_ids)}")

    # Preview first 3
    with open(OUTPUT_FILE, "r") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(json.loads(line))
else:
    print(f"⚠️  Test file not found: {TEST_FILE}")


train_sample_texts = [p["input"].replace("extract triplets: ", "") for p in train_pairs[:20]]
train_sample_ids   = [p["id"] for p in train_pairs[:20]]
train_sample_golds = [p["target"] for p in train_pairs[:20]]

train_sample_preds = predict_triplets(train_sample_texts, model, tokenizer, max_len=MAX_TARGET_LEN)

for i in range(min(5, len(train_sample_preds))):
    print(f"\n{'='*70}")
    print(f"ID:    {train_sample_ids[i]}")
    print(f"Text:  {train_sample_texts[i][:100]}")
    print(f"Gold:  {train_sample_golds[i][:120]}")

