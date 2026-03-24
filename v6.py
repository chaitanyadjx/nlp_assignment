"""
ASTE Pipeline v6 — DeBERTa-v3-base (Improved)
===============================================
Unified single-encoder model with BIO tagger + VA regressor.
Improvements:
  - Class-weighted BIO loss (handles tag imbalance)
  - Differential learning rates (encoder=1e-5, heads=5e-5)
  - Proximity-based aspect–opinion pair filtering
  - Early stopping with patience
"""

# ── Imports ────────────────────────────────────────────────────────────────
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
from torch.optim import AdamW

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/kaggle/input/datasets/chaitanyajx1/datasetnlp")
REST_FILE   = DATA_DIR / "restaurant_train.jsonl"
LAP_FILE    = DATA_DIR / "laptop_train.jsonl"
TEST_FILE   = DATA_DIR / "test.jsonl"
MODEL_DIR   = Path("./aste_pipeline_model")
OUTPUT_FILE = Path("predictions_v5.jsonl")

MODEL_DIR.mkdir(exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────
FAST_TEST    = False
MODEL_NAME   = "microsoft/deberta-v3-base"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 30
ENCODER_LR   = 1e-5     # slower LR for pretrained encoder
HEAD_LR      = 5e-5     # faster LR for task heads
WARMUP_RATIO = 0.1
PATIENCE     = 5         # early stopping patience
PAIR_MAX_DIST= 15        # max token distance for aspect-opinion pairing
VA_MIN, VA_MAX = 1.0, 9.0

# Class weights for BIO tags (O is ~95% of tokens, up-weight rare tags)
BIO_WEIGHTS = [0.15, 3.0, 2.0, 3.0, 2.0]  # O, B-ASP, I-ASP, B-OPN, I-OPN

# ── BIO Labels ─────────────────────────────────────────────────────────────
LABELS   = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}
NUM_TAGS = len(LABELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")
print(f"Labels: {LABELS}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file, return list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_null(val: str) -> bool:
    """Check if an Aspect or Opinion is NULL."""
    return val is None or str(val).strip().upper() == "NULL" or str(val).strip() == ""


def extract_triplets(record: Dict) -> List[Dict]:
    """
    Extract (Aspect, Opinion, VA) from a Quadruplet record.
    - Ignores 'Category' field entirely.
    - Skips entries where Aspect OR Opinion is NULL.
    Returns list of clean dicts: {"Aspect": str, "Opinion": str, "VA": str}
    """
    quads = record.get("Quadruplet", [])
    triplets = []
    for q in quads:
        asp = q.get("Aspect", "NULL")
        opn = q.get("Opinion", "NULL")
        va  = q.get("VA", "5.00#5.00")
        # Skip NULL entries — nothing to supervise span detection on
        if is_null(asp) or is_null(opn):
            continue
        triplets.append({"Aspect": asp, "Opinion": opn, "VA": va})
    return triplets


def load_and_merge_train(lap_file: Path, rest_file: Path) -> List[Dict]:
    """
    Load laptop + restaurant training data.
    Normalises each record to: {"ID": str, "Text": str, "Triplet": [...]}
    """
    combined = []
    for filepath, domain in [(lap_file, "LAP"), (rest_file, "REST")]:
        if not filepath.exists():
            print(f"  [WARN] File not found, skipping: {filepath}")
            continue
        raw = load_jsonl(filepath)
        for rec in raw:
            triplets = extract_triplets(rec)
            combined.append({
                "ID":      rec["ID"],
                "Text":    rec["Text"],
                "Domain":  domain,
                "Triplet": triplets,   # may be empty if all quads were NULL
            })
        print(f"  Loaded {len(raw):>5} records from {filepath.name}  (domain={domain})")

    print(f"  Total training records: {len(combined)}")
    return combined


def load_test(test_file: Path) -> List[Dict]:
    """
    Load test data. Test records may have Quadruplet (dev set)
    or just Text (blind test). Normalises to {"ID", "Text"}.
    """
    raw = load_jsonl(test_file)
    records = []
    for rec in raw:
        records.append({"ID": rec["ID"], "Text": rec["Text"]})
    print(f"  Loaded {len(records):>5} test records from {test_file.name}")
    return records


# ── Run Data Loading ───────────────────────────────────────────────────────
print("\n── Loading Data ─────────────────────────────────────────")
train_data = load_and_merge_train(LAP_FILE, REST_FILE)

# Load test data if available
if TEST_FILE.exists():
    test_data = load_test(TEST_FILE)
else:
    test_data = []
    print("  [INFO] test.jsonl not found — skipping test set for now")

# Optional quick sanity-check mode
if FAST_TEST:
    train_data = train_data[:200]
    test_data  = test_data[:50]
    print("  [FAST_TEST] Truncated to 200 train / 50 test")

# Validation split (10%)
np.random.seed(42)
np.random.shuffle(train_data)
val_size   = max(1, int(len(train_data) * 0.1))
val_data   = train_data[:val_size]
train_data = train_data[val_size:]
print(f"  Split → train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

# Quick stats
null_skipped = sum(
    1 for r in (train_data + val_data)
    for q in r.get("Triplet", [])  # already filtered, but count empties
) 
train_with_triplets = sum(1 for r in train_data if len(r["Triplet"]) > 0)
print(f"  Train records with ≥1 valid triplet: {train_with_triplets}/{len(train_data)}")

# ── Sample inspection ──────────────────────────────────────────────────────
print("\n── Sample Records ───────────────────────────────────────")
for rec in train_data[:3]:
    print(f"  ID     : {rec['ID']}")
    print(f"  Domain : {rec['Domain']}")
    print(f"  Text   : {rec['Text'][:80]}")
    print(f"  Triplets ({len(rec['Triplet'])}):")
    for t in rec["Triplet"]:
        print(f"    Aspect={t['Aspect']!r:30s}  Opinion={t['Opinion']!r:25s}  VA={t['VA']}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOKENIZATION & DATASET
# ════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_va(va_str: str) -> Tuple[float, float]:
    try:
        v, a = va_str.split("#")
        return float(v), float(a)
    except Exception:
        return 5.0, 5.0


def normalize_va(val: float) -> float:
    return (val - VA_MIN) / (VA_MAX - VA_MIN)


def denormalize_va(val: float) -> float:
    return val * (VA_MAX - VA_MIN) + VA_MIN


def format_va(v: float, a: float) -> str:
    v = float(np.clip(v, VA_MIN, VA_MAX))
    a = float(np.clip(a, VA_MIN, VA_MAX))
    return f"{v:.2f}#{a:.2f}"


def find_token_span(
    phrase: str,
    text: str,
    offset_mapping: List[Tuple[int, int]],
    seq_len: int,
) -> Optional[Tuple[int, int]]:
    """
    Map a phrase to token indices via character-level offset matching.
    Returns (tok_start, tok_end) inclusive, or None if not found.
    """
    phrase_lower = phrase.lower().strip()
    text_lower   = text.lower()

    char_start = text_lower.find(phrase_lower)
    if char_start == -1:
        return None
    char_end = char_start + len(phrase_lower) - 1

    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offset_mapping):
        if i >= seq_len:
            break
        if s == 0 and e == 0:
            continue  # special tokens
        if tok_start is None and s >= char_start:
            tok_start = i
        if e > 0 and e - 1 <= char_end:
            tok_end = i

    if tok_start is None or tok_end is None:
        return None
    if tok_start > tok_end:
        tok_end = tok_start
    return tok_start, tok_end


class ASTEDataset(Dataset):
    def __init__(self, records: List[Dict], is_train: bool = True):
        self.is_train = is_train
        self.samples  = [self._process(r) for r in records]

    def _process(self, rec: Dict) -> Dict:
        text     = rec["Text"]
        triplets = rec.get("Triplet", []) if self.is_train else []

        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offset_mapping = enc["offset_mapping"].squeeze(0).tolist()
        seq_len        = int(attention_mask.sum().item())

        bio_tags   = torch.zeros(MAX_LEN, dtype=torch.long)
        span_pairs = []  # (asp_s, asp_e, opn_s, opn_e, v_norm, a_norm)

        if self.is_train:
            for t in triplets:
                asp_span = find_token_span(t["Aspect"],  text, offset_mapping, seq_len)
                opn_span = find_token_span(t["Opinion"], text, offset_mapping, seq_len)
                if asp_span is None or opn_span is None:
                    continue

                asp_s, asp_e = asp_span
                opn_s, opn_e = opn_span
                v_raw, a_raw = parse_va(t["VA"])

                bio_tags[asp_s] = label2id["B-ASP"]
                for i in range(asp_s + 1, asp_e + 1):
                    bio_tags[i] = label2id["I-ASP"]

                bio_tags[opn_s] = label2id["B-OPN"]
                for i in range(opn_s + 1, opn_e + 1):
                    bio_tags[i] = label2id["I-OPN"]

                span_pairs.append((
                    asp_s, asp_e, opn_s, opn_e,
                    normalize_va(v_raw), normalize_va(a_raw),
                ))

        return {
            "id":            rec["ID"],
            "text":          text,
            "input_ids":     input_ids,
            "attention_mask":attention_mask,
            "bio_tags":      bio_tags,
            "span_pairs":    span_pairs,
            "seq_len":       seq_len,
            "offset_mapping":offset_mapping,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "ids":            [b["id"]            for b in batch],
        "texts":          [b["text"]          for b in batch],
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "bio_tags":       torch.stack([b["bio_tags"]       for b in batch]),
        "span_pairs":     [b["span_pairs"]    for b in batch],
        "seq_lens":       [b["seq_len"]       for b in batch],
        "offset_mappings":[b["offset_mapping"]for b in batch],
    }


print("\n── Building Datasets ────────────────────────────────────")
train_ds = ASTEDataset(train_data, is_train=True)
val_ds   = ASTEDataset(val_data,   is_train=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

if test_data:
    test_ds     = ASTEDataset(test_data, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"  Test  batches: {len(test_loader)}")
else:
    test_loader = None
    print("  Test: skipped (no test.jsonl)")

print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL
# ════════════════════════════════════════════════════════════════════════════

class ASTEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(MODEL_NAME)
        hidden        = self.encoder.config.hidden_size  # 768 for deberta-v3-base

        # BIO tagger
        self.dropout  = nn.Dropout(0.1)
        self.bio_head = nn.Linear(hidden, NUM_TAGS)

        # VA regression: concat(asp_repr, opn_repr) → (valence, arousal)
        self.va_head  = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),    # output in [0, 1], denormalize later
        )

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = self.dropout(out.last_hidden_state)   # (B, L, H)
        bio_logits = self.bio_head(hidden)               # (B, L, NUM_TAGS)
        return hidden, bio_logits


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ════════════════════════════════════════════════════════════════════════════

# Class-weighted BIO loss to handle severe tag imbalance
bio_class_weights = torch.tensor(BIO_WEIGHTS, dtype=torch.float32).to(DEVICE)

def bio_loss_fn(logits, targets, mask):
    B, L, C = logits.shape
    active  = mask.view(-1).bool()
    return F.cross_entropy(
        logits.view(-1, C)[active],
        targets.view(-1)[active],
        weight=bio_class_weights,
    )


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_bio, total_va, n_va = 0.0, 0.0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        bio_tags       = batch["bio_tags"].to(DEVICE)
        span_pairs_all = batch["span_pairs"]

        hidden, bio_logits = model(input_ids, attention_mask)
        b_loss = bio_loss_fn(bio_logits, bio_tags, attention_mask)

        va_loss = torch.tensor(0.0, device=DEVICE)
        batch_va = 0
        for b_idx, pairs in enumerate(span_pairs_all):
            for (as_, ae, os_, oe, vn, an) in pairs:
                if ae >= hidden.shape[1] or oe >= hidden.shape[1]:
                    continue
                asp_r = hidden[b_idx, as_:ae+1].mean(0)
                opn_r = hidden[b_idx, os_:oe+1].mean(0)
                pred  = model.va_head(torch.cat([asp_r, opn_r]).unsqueeze(0)).squeeze(0)
                gt    = torch.tensor([vn, an], device=DEVICE, dtype=torch.float32)
                va_loss = va_loss + F.mse_loss(pred, gt)
                batch_va += 1
                n_va += 1

        if batch_va > 0:
            va_loss = va_loss / batch_va

        loss = b_loss + va_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_bio += b_loss.item()
        total_va  += va_loss.item() if batch_va > 0 else 0.0

    return total_bio / len(loader), total_va / len(loader)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def spans_from_bio(seq: List[int], b_tag: int, i_tag: int) -> List[Tuple[int,int]]:
    spans, start = [], None
    for i, t in enumerate(seq):
        if t == b_tag:
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif t == i_tag:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(seq) - 1))
    return spans


def tokens_to_text(text: str, offset_mapping, start: int, end: int) -> str:
    char_s = offset_mapping[start][0]
    char_e = offset_mapping[end][1]
    return text[char_s:char_e].strip()


@torch.no_grad()
def predict(model, loader) -> List[Dict]:
    model.eval()
    results = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        ids            = batch["ids"]
        texts          = batch["texts"]
        seq_lens       = batch["seq_lens"]
        offset_maps    = batch["offset_mappings"]

        hidden, bio_logits = model(input_ids, attention_mask)
        bio_preds = bio_logits.argmax(-1).cpu().tolist()

        for b_idx in range(len(ids)):
            seq_len   = seq_lens[b_idx]
            bio_seq   = bio_preds[b_idx][:seq_len]
            text      = texts[b_idx]
            off_map   = offset_maps[b_idx]

            asp_spans = spans_from_bio(bio_seq, label2id["B-ASP"], label2id["I-ASP"])
            opn_spans = spans_from_bio(bio_seq, label2id["B-OPN"], label2id["I-OPN"])

            triplets = []
            for (as_, ae) in asp_spans:
                # Find closest opinion span by token distance (proximity filter)
                best_opn, best_dist = None, float("inf")
                for (os_, oe) in opn_spans:
                    dist = min(abs(as_ - oe), abs(os_ - ae))  # gap between spans
                    if dist < best_dist:
                        best_dist = dist
                        best_opn  = (os_, oe)

                if best_opn is None or best_dist > PAIR_MAX_DIST:
                    continue

                os_, oe = best_opn
                if ae >= hidden.shape[1] or oe >= hidden.shape[1]:
                    continue
                asp_r = hidden[b_idx, as_:ae+1].mean(0)
                opn_r = hidden[b_idx, os_:oe+1].mean(0)
                va    = model.va_head(torch.cat([asp_r, opn_r]).unsqueeze(0)).squeeze(0).cpu().tolist()

                asp_text = tokens_to_text(text, off_map, as_, ae)
                opn_text = tokens_to_text(text, off_map, os_, oe)

                if asp_text and opn_text:
                    triplets.append({
                        "Aspect":  asp_text,
                        "Opinion": opn_text,
                        "VA":      format_va(denormalize_va(va[0]), denormalize_va(va[1])),
                    })

            results.append({"ID": ids[b_idx], "Triplet": triplets})

    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def triplet_f1(preds: List[Dict], golds: List[Dict]) -> Dict:
    pred_map = {r["ID"]: r["Triplet"] for r in preds}
    gold_map = {r["ID"]: r["Triplet"] for r in golds}

    tp = fp = fn = 0
    v_errs, a_errs = [], []

    for id_, gold_trips in gold_map.items():
        pred_trips = pred_map.get(id_, [])
        gold_set   = {(t["Aspect"].lower(), t["Opinion"].lower()) for t in gold_trips}
        pred_set   = {(t["Aspect"].lower(), t["Opinion"].lower()) for t in pred_trips}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        g_va = {(t["Aspect"].lower(), t["Opinion"].lower()): t["VA"] for t in gold_trips}
        p_va = {(t["Aspect"].lower(), t["Opinion"].lower()): t["VA"] for t in pred_trips}
        for pair in (gold_set & pred_set):
            gv, ga = parse_va(g_va[pair])
            pv, pa = parse_va(p_va[pair])
            v_errs.append(abs(gv - pv))
            a_errs.append(abs(ga - pa))

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "va_mae_v": round(np.mean(v_errs) if v_errs else 0.0, 4),
        "va_mae_a": round(np.mean(a_errs) if a_errs else 0.0, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

print("\n── Initialising Model ───────────────────────────────────")
model = ASTEModel().to(DEVICE)
model = model.float()  # Force fp32 — prevents DeBERTa NaN on T4

# Differential learning rates: slower for pretrained encoder, faster for task heads
no_decay = ["bias", "LayerNorm.weight"]
encoder_params_decay    = [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
encoder_params_no_decay = [p for n, p in model.encoder.named_parameters() if     any(nd in n for nd in no_decay)]
head_params_decay       = [p for n, p in list(model.bio_head.named_parameters()) + list(model.va_head.named_parameters()) if not any(nd in n for nd in no_decay)]
head_params_no_decay    = [p for n, p in list(model.bio_head.named_parameters()) + list(model.va_head.named_parameters()) if     any(nd in n for nd in no_decay)]

params = [
    {"params": encoder_params_decay,    "lr": ENCODER_LR, "weight_decay": 0.01},
    {"params": encoder_params_no_decay, "lr": ENCODER_LR, "weight_decay": 0.0},
    {"params": head_params_decay,       "lr": HEAD_LR,    "weight_decay": 0.01},
    {"params": head_params_no_decay,    "lr": HEAD_LR,    "weight_decay": 0.0},
]
optimizer    = AdamW(params)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

CKPT_PATH = MODEL_DIR / "best_model.pt"
best_f1   = 0.0
no_improve = 0   # early stopping counter

print(f"  Encoder LR: {ENCODER_LR}  Head LR: {HEAD_LR}")
print(f"  Total steps: {total_steps}  Warmup: {warmup_steps}")
print(f"  BIO class weights: {BIO_WEIGHTS}")
print(f"  Pair max distance: {PAIR_MAX_DIST} tokens")
print(f"  Early stopping patience: {PATIENCE}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"\n── Training ─────────────────────────────────────────────")

for epoch in range(1, EPOCHS + 1):
    bio_l, va_l = train_one_epoch(model, train_loader, optimizer, scheduler)

    val_preds = predict(model, val_loader)
    metrics   = triplet_f1(val_preds, val_data)

    flag = ""
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        torch.save(model.state_dict(), CKPT_PATH)
        flag = "  ✓ saved"
        no_improve = 0
    else:
        no_improve += 1

    print(
        f"  Epoch {epoch:02d}/{EPOCHS}"
        f"  bio={bio_l:.4f}  va={va_l:.4f}"
        f"  val-F1={metrics['f1']:.4f}"
        f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
        f"  VA-MAE(V={metrics['va_mae_v']:.3f} A={metrics['va_mae_a']:.3f})"
        f"{flag}"
    )

    if no_improve >= PATIENCE:
        print(f"  ⏹ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FINAL PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════

if test_loader:
    print(f"\n── Inference (best F1={best_f1:.4f}) ───────────────────")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    predictions = predict(model, test_loader)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Saved {len(predictions)} predictions → {OUTPUT_FILE}")

    # Quick sample output
    print("\n── Sample Predictions ───────────────────────────────────")
    for rec in predictions[:3]:
        print(f"  ID: {rec['ID']}")
        for t in rec["Triplet"]:
            print(f"    {t['Aspect']!r:30s} | {t['Opinion']!r:25s} | VA={t['VA']}")
        print()
else:
    print("\n⚠️  Skipping test inference — test.jsonl not loaded")