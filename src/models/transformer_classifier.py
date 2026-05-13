"""Model 2: Fine-tuned DistilBERT classifier.

Educational note: A transformer reads the whole text with self-attention rather
than treating words independently, so it can pick up contextual cues (sarcasm,
urgency framing, modifier scope). We fine-tune the head + the body on our task
so the model adapts its representations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from src import config


class TextDataset(Dataset):
    """Minimal torch Dataset wrapping (text, label) pairs."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer,
                 max_length: int = config.TRANSFORMER_MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def fine_tune(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    *,
    model_name: str = config.TRANSFORMER_NAME,
    epochs: int = config.TRANSFORMER_EPOCHS,
    batch_size: int = config.TRANSFORMER_BATCH_SIZE,
    lr: float = config.TRANSFORMER_LR,
    save_dir: Path | None = None,
):
    """Fine-tune `model_name` on the supplied data. Returns (model, tokenizer)."""
    device = torch.device("cpu")  # coursework-realistic; CUDA path identical
    torch.manual_seed(config.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(config.LABELS_BINARY),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    ).to(device)

    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=config.TRANSFORMER_EVAL_BATCH,
                        shuffle=False, num_workers=0)

    optim = AdamW(model.parameters(), lr=lr,
                  weight_decay=config.TRANSFORMER_WEIGHT_DECAY)
    total_steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            train_loss += out.loss.item()
        train_loss /= max(1, len(train_dl))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
                preds = out.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        val_loss /= max(1, len(val_dl))
        val_acc = correct / max(1, total)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"  epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    return model, tokenizer, history


@torch.no_grad()
def predict(model, tokenizer, texts: Sequence[str], batch_size: int = 32):
    """Return (preds, proba) on a list of texts."""
    device = next(model.parameters()).device
    model.eval()
    preds_all, proba_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=config.TRANSFORMER_MAX_LEN,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        proba_all.append(proba)
        preds_all.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return np.array(preds_all), np.concatenate(proba_all, axis=0)
