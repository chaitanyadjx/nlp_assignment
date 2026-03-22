import json
import re
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
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
MODEL_DIR = Path("./aste_pipeline_model")
OUTPUT_FILE = Path("predictions_v5.jsonl")

# Small subset for quick testing if FAST_TEST is True. Set to False for real training.
FAST_TEST = False 

MODEL_NAME = "microsoft/deberta-v3-base"

# BIO labels: B-ASP, I-ASP, B-OPN, I-OPN, O
LABELS = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}

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

print(f"Total Samples Loaded: {len(all_data)}")

# Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 3. Stage 1: Span Extraction
class SpanDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.samples = []
        for item in data:
            text = preprocess_text(item["Text"])
            enc = tokenizer(text, truncation=True, max_length=max_len,
                           return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")

            labels = [label2id["O"]] * len(enc["input_ids"])
            
            # Apply -100 to special tokens FIRST
            for idx, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    labels[idx] = -100
            if "Quadruplet" in item:
                for quad in item["Quadruplet"]:
                    for span_type, bio_b, bio_i in [
                        ("Aspect", "B-ASP", "I-ASP"),
                        ("Opinion", "B-OPN", "I-OPN")
                    ]:
                        span_text = quad.get(span_type)
                        if not span_text or span_text == "NULL":
                            continue
                        span_start = text.find(span_text)
                        if span_start == -1:
                            continue
                        span_end = span_start + len(span_text)
                        first = True
                        for idx, (s, e) in enumerate(offsets):
                            
                            if s >= span_start and e <= span_end + 1: # +1 for subword boundary edges
                                if first:
                                    labels[idx] = label2id[bio_b]
                                    first = False
                                else:
                                    labels[idx] = label2id[bio_i]

            enc["labels"] = labels
            self.samples.append({k: torch.tensor(v) for k, v in enc.items()})

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def train_stage1(data, tokenizer):
    print("--- Stage 1: Training Span Extractor ---")
    span_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    train_dataset = SpanDataset(data, tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(span_model.parameters(), lr=1e-5, weight_decay=0.01)
    
    span_model.train()
    epochs = 2 if FAST_TEST else 4 # reduced from 5 to save time/prevent over-fitting
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = span_model(**batch)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss at epoch {epoch+1} step {step}. Skipping batch.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(span_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    span_model.save_pretrained(str(MODEL_DIR / "stage1_final"))
    return span_model

# 4. Stage 2: Pair Classifier
class PairClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, asp_mask, opn_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        cls = out[:, 0]
        asp_vec = (out * asp_mask.unsqueeze(-1)).sum(1) / asp_mask.sum(1, keepdim=True).clamp(min=1)
        opn_vec = (out * opn_mask.unsqueeze(-1)).sum(1) / opn_mask.sum(1, keepdim=True).clamp(min=1)
        return self.classifier(torch.cat([cls, asp_vec, opn_vec], dim=-1))

class PairDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.samples = []
        all_aspects = []
        all_opinions = []
        for item in data:
            if "Quadruplet" in item:
                for q in item["Quadruplet"]:
                    if q.get("Aspect") and q["Aspect"] != "NULL":
                        all_aspects.append(q["Aspect"])
                    if q.get("Opinion") and q["Opinion"] != "NULL":
                        all_opinions.append(q["Opinion"])
                        
        for item in data:
            text = preprocess_text(item["Text"])
            pos_pairs = set()
            if "Quadruplet" in item:
                for q in item["Quadruplet"]:
                    asp, opn = q.get("Aspect"), q.get("Opinion")
                    if asp and asp != "NULL" and opn and opn != "NULL":
                        pos_pairs.add((asp, opn))

            neg_pairs = set()
            for _ in range(len(pos_pairs) * 2):
                neg_asp = random.choice(all_aspects) if all_aspects else "food"
                neg_opn = random.choice(all_opinions) if all_opinions else "good"
                if (neg_asp, neg_opn) not in pos_pairs:
                    neg_pairs.add((neg_asp, neg_opn))

            all_pairs = [(a, o, 1) for a, o in pos_pairs] + [(a, o, 0) for a, o in neg_pairs]
            for asp, opn, label in all_pairs:
                prompt = f"{text} [ASP] {asp} [OPN] {opn}"
                enc = tokenizer(prompt, truncation=True, max_length=max_len)
                
                input_ids = enc["input_ids"]
                asp_mask = [0]*len(input_ids)
                opn_mask = [0]*len(input_ids)
                
                asp_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[ASP]")) if "[ASP]" in tokenizer.vocab else -1
                opn_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[OPN]")) if "[OPN]" in tokenizer.vocab else -1
                
                if asp_tag_idx != -1 and opn_tag_idx != -1 and asp_tag_idx < opn_tag_idx:
                    for i in range(asp_tag_idx + 1, opn_tag_idx): asp_mask[i] = 1
                    for i in range(opn_tag_idx + 1, len(input_ids)-1): opn_mask[i] = 1

                self.samples.append({
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(enc["attention_mask"]),
                    "asp_mask": torch.tensor(asp_mask, dtype=torch.float),
                    "opn_mask": torch.tensor(opn_mask, dtype=torch.float),
                    "labels": torch.tensor(label, dtype=torch.long)
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def custom_collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    res = {}
    for k in batch[0].keys():
        if k in ["labels", "va_targets", "vad_prior"]:
            res[k] = torch.stack([b[k] for b in batch])
        else:
            padding_value = 0
            res[k] = torch.stack([
                torch.cat([b[k], torch.full((max_len - b[k].size(0),), padding_value, dtype=b[k].dtype)])
                for b in batch
            ])
    return res

def train_stage2(data, tokenizer):
    print("--- Stage 2: Training Pair Classifier ---")
    special_tokens = {"additional_special_tokens": ["[ASP]", "[OPN]"]}
    tokenizer.add_special_tokens(special_tokens)
    
    pair_model = PairClassifier(MODEL_NAME).to(DEVICE)
    pair_model.encoder.resize_token_embeddings(len(tokenizer))
    
    train_dataset = PairDataset(data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    
    optimizer = torch.optim.AdamW(pair_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    pair_model.train()
    epochs = 1 if FAST_TEST else 3
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            out = pair_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                asp_mask=batch["asp_mask"].to(DEVICE),
                opn_mask=batch["opn_mask"].to(DEVICE)
            )
            loss = criterion(out, batch["labels"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pair_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss/max(1, len(train_loader)):.4f}")
        
    torch.save(pair_model.state_dict(), MODEL_DIR / "stage2.pt")
    return pair_model

# 5. Stage 3: VA Regression
def load_vad_lexicon(path="nrc_vad.txt"):
    lexicon = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    word, v, a, d = parts[:4]
                    lexicon[word.lower()] = (float(v), float(a))
        print(f"Loaded {len(lexicon)} words from lexicon.")
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using neutral fallbacks.")
    return lexicon

def get_vad_prior(opinion_text, lexicon):
    words = opinion_text.lower().split()
    scores = [lexicon[w] for w in words if w in lexicon]
    if scores:
        v = sum(s[0] for s in scores) / len(scores)
        a = sum(s[1] for s in scores) / len(scores)
        return v * 8 + 1, a * 8 + 1
    return 5.0, 5.0  

class VARegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden + 2, 128),  
            nn.GELU(),
            nn.Linear(128, 2)            
        )

    def forward(self, input_ids, attention_mask, opn_mask, vad_prior):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        opn_vec = (out * opn_mask.unsqueeze(-1)).sum(1) / opn_mask.sum(1, keepdim=True).clamp(min=1)
        x = torch.cat([opn_vec, vad_prior], dim=-1)
        return self.head(x)

class VADataset(Dataset):
    def __init__(self, data, tokenizer, lexicon, max_len=128):
        self.samples = []
        for item in data:
            text = preprocess_text(item["Text"])
            if "Quadruplet" in item:
                for q in item["Quadruplet"]:
                    asp, opn, va = q.get("Aspect"), q.get("Opinion"), q.get("VA")
                    if not opn or opn == "NULL" or not va:
                        continue
                    try:
                        v, a = va.split("#")
                        v, a = float(v), float(a)
                    except:
                        continue

                    prompt = f"{text} [OPN] {opn}"
                    enc = tokenizer(prompt, truncation=True, max_length=max_len)
                    input_ids = enc["input_ids"]
                    
                    opn_mask = [0]*len(input_ids)
                    opn_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[OPN]")) if "[OPN]" in tokenizer.vocab else -1
                    if opn_tag_idx != -1:
                        for i in range(opn_tag_idx + 1, len(input_ids)-1):
                            opn_mask[i] = 1

                    prior_v, prior_a = get_vad_prior(opn, lexicon)

                    self.samples.append({
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.tensor(enc["attention_mask"]),
                        "opn_mask": torch.tensor(opn_mask, dtype=torch.float),
                        "vad_prior": torch.tensor([prior_v, prior_a], dtype=torch.float),
                        "va_targets": torch.tensor([v, a], dtype=torch.float)
                    })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def train_stage3(data, tokenizer, lexicon):
    print("--- Stage 3: Training VA Regressor ---")
    va_model = VARegressor(MODEL_NAME).to(DEVICE)
    va_model.encoder.resize_token_embeddings(len(tokenizer))
    
    train_dataset = VADataset(data, tokenizer, lexicon)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    
    optimizer = torch.optim.AdamW(va_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    va_model.train()
    epochs = 1 if FAST_TEST else 3
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            out = va_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                opn_mask=batch["opn_mask"].to(DEVICE),
                vad_prior=batch["vad_prior"].to(DEVICE)
            )
            loss = criterion(out, batch["va_targets"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(va_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss/max(1, len(train_loader)):.4f}")
        
    torch.save(va_model.state_dict(), MODEL_DIR / "stage3.pt")
    return va_model

# Inference Function
def predict_triplets(text, span_model, pair_model, va_model, tokenizer, lexicon):
    span_model.eval()
    pair_model.eval()
    va_model.eval()
    
    with torch.no_grad():
        # Stage 1: extract spans
        enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offsets = enc.pop("offset_mapping")[0]
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        logits = span_model(**enc).logits[0].argmax(-1)

        aspects, opinions = [], []
        for span_type, b_tag, i_tag in [("aspect","B-ASP","I-ASP"),
                                         ("opinion","B-OPN","I-OPN")]:
            spans, cur = [], None
            for idx, label_id in enumerate(logits):
                label_id_val = label_id.item()
                if label_id_val not in id2label: continue
                label = id2label[label_id_val]
                s, e = offsets[idx].tolist()
                
                if label == b_tag:
                    if cur: spans.append(text[cur[0]:cur[1]])
                    cur = [s, e]
                elif label == i_tag and cur:
                    cur[1] = e
                else:
                    if cur:
                        spans.append(text[cur[0]:cur[1]])
                        cur = None
            if cur: spans.append(text[cur[0]:cur[1]])
            
            spans = [sp.strip() for sp in spans if sp.strip()]
            if span_type == "aspect": aspects = list(set(spans))
            else: opinions = list(set(spans))

        # Stage 2: filter pairs
        valid_pairs = []
        for asp in aspects:
            for opn in opinions:
                prompt = f"{text} [ASP] {asp} [OPN] {opn}"
                enc2 = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = enc2["input_ids"][0].tolist()
                asp_mask, opn_mask = [0]*len(input_ids), [0]*len(input_ids)
                
                asp_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[ASP]")) if "[ASP]" in tokenizer.vocab else -1
                opn_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[OPN]")) if "[OPN]" in tokenizer.vocab else -1
                if asp_tag_idx != -1 and opn_tag_idx != -1 and asp_tag_idx < opn_tag_idx:
                    for i in range(asp_tag_idx + 1, opn_tag_idx): asp_mask[i] = 1
                    for i in range(opn_tag_idx + 1, len(input_ids)-1): opn_mask[i] = 1
                
                enc2 = {k:v.to(DEVICE) for k,v in enc2.items()}
                score = pair_model(**enc2, 
                                   asp_mask=torch.tensor([asp_mask], dtype=torch.float).to(DEVICE),
                                   opn_mask=torch.tensor([opn_mask], dtype=torch.float).to(DEVICE)).logits.softmax(-1)[0, 1].item()
                if score > 0.5:
                    valid_pairs.append((asp, opn))

        # Stage 3: VA for each valid pair
        triplets = []
        for asp, opn in valid_pairs:
            prior_v, prior_a = get_vad_prior(opn, lexicon)
            prior_tensor = torch.tensor([[prior_v, prior_a]], dtype=torch.float).to(DEVICE)
            
            enc3 = tokenizer(f"{text} [OPN] {opn}", return_tensors="pt", truncation=True, max_length=128)
            input_ids = enc3["input_ids"][0].tolist()
            opn_mask = [0]*len(input_ids)
            
            opn_tag_idx = input_ids.index(tokenizer.convert_tokens_to_ids("[OPN]")) if "[OPN]" in tokenizer.vocab else -1
            if opn_tag_idx != -1:
                for i in range(opn_tag_idx + 1, len(input_ids)-1): opn_mask[i] = 1
            
            enc3 = {k:v.to(DEVICE) for k,v in enc3.items()}
            out = va_model(**enc3, opn_mask=torch.tensor([opn_mask], dtype=torch.float).to(DEVICE),
                            vad_prior=prior_tensor).squeeze()
            
            v, a = out[0], out[1]
            if torch.is_tensor(v):
                v_val = v.clamp(1, 9).item()
                a_val = a.clamp(1, 9).item()
            else:
                v_val = max(1.0, min(9.0, v))
                a_val = max(1.0, min(9.0, a))
                
            triplets.append({
                "Aspect": asp,
                "Opinion": opn,
                "VA": f"{v_val:.2f}#{a_val:.2f}"
            })

        return triplets

def compute_metrics(val_data, val_preds):
    """Compute aspect/opinion extraction F1 and VA MAE."""
    total_gold = 0
    total_pred = 0
    correct    = 0
    va_errors  = []

    for item, preds in zip(val_data, val_preds):
        gold_triplets = item.get("Quadruplet", [])
        total_gold += len(gold_triplets)
        total_pred += len(preds)

        # Match lowercased (Aspect, Opinion)
        gold_set = {(g["Aspect"].lower(), g["Opinion"].lower()): g.get("VA", "5.00#5.00") for g in gold_triplets}
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
    print(f"Pipeline Evaluation (Validation Set):")
    print(f"  Gold: {total_gold}  Pred: {total_pred}  Correct: {correct}")
    print(f"  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    if va_errors:
         print(f"  VA MAE (matched): {avg_va:.4f}  (n={len(va_errors)})")
    else:
         print(f"  VA MAE (matched): N/A")
    print(f"{'='*50}\n")
    return f1

# Main execution block
if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    lexicon = load_vad_lexicon("nrc_vad.txt")
    
    # Train/Val Split
    all_data_copy = list(all_data)
    random.shuffle(all_data_copy)
    split_idx = int(len(all_data_copy) * 0.9)
    train_data = all_data_copy[:split_idx]
    val_data = all_data_copy[split_idx:]
    
    if FAST_TEST:
        train_data = train_data[:100]
        val_data = val_data[:20]
        
    print(f"Using {len(train_data)} samples for training, {len(val_data)} for validation.")
    
    span_model = train_stage1(train_data, tokenizer)
    pair_model = train_stage2(train_data, tokenizer)
    va_model = train_stage3(train_data, tokenizer, lexicon)
    
    print("\n--- Predicting on Validation Set ---")
    val_preds = []
    for rec in tqdm(val_data, desc="Evaluating"):
        text = preprocess_text(rec["Text"])
        preds = predict_triplets(text, span_model, pair_model, va_model, tokenizer, lexicon)
        val_preds.append(preds)
        
    compute_metrics(val_data, val_preds)

