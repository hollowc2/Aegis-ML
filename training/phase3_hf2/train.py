"""
training/phase3_hf2/train.py
==============================
Phase 3 (HF2 Ultra) multi-task fine-tuning.

Trains AegisMTModel (DeBERTa-v3-base) simultaneously on:
  Task 1 — binary classification: 0=BENIGN, 1=MALICIOUS  (primary)
  Task 2 — threat category (7 classes)                    (auxiliary, alpha=0.3)

Key improvements over Phase 2:
  - DeBERTa-v3-base (183M) vs v3-small (142M): more capacity for multi-task
  - threat_category column in dataset → learned rather than heuristic labels
  - --adversarial-training flag: 2-epoch fine-tuning pass on augmented hard examples
  - Post-training temperature calibration (T* saved in config.json)
  - Effective batch size 128 (gradient accumulation × 8) for multi-task stability

GPU requirements (DeBERTa-v3-base, fp32):
  Full fine-tune: ~10 GB VRAM
  QLoRA 4-bit:    ~5 GB VRAM

AMD ROCm: set HSA_OVERRIDE_GFX_VERSION=11.0.0 before running.

Run:
    python -m training.phase3_hf2.train
    python -m training.phase3_hf2.train --model deberta-base --epochs 12
    python -m training.phase3_hf2.train --adversarial-training
    python -m training.phase3_hf2.train --model modernbert
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# AMD RDNA3 (gfx1101) compatibility — must be set before any torch import.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from training.phase3_hf2.model import (
    AegisMTModel,
    BINARY_ID2LABEL,
    BINARY_LABEL2ID,
    THREAT2ID,
    ID2THREAT,
    THREAT_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = Path("data/combined_dataset.csv")
MODELS_DIR = Path("models")
HF2_MODEL_OUTPUT = MODELS_DIR / "hf2_classifier"

MODEL_NAMES = {
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-small",
    "deberta-base": "microsoft/deberta-v3-base",
    "modernbert": "answerdotai/ModernBERT-base",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data() -> DatasetDict:
    """Load and split the combined dataset into train / val / test splits."""
    if not DATASET_CSV.exists():
        logger.info("Dataset not found — generating...")
        from training.data.prepare_dataset import main as prep_main
        prep_main()

    df = pd.read_csv(DATASET_CSV).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # Map threat_category string → integer id (-100 if unknown/missing)
    if "threat_category" in df.columns:
        df["threat_label"] = df["threat_category"].map(THREAT2ID).fillna(-100).astype(int)
    else:
        # Fallback: heuristic assignment — benign → "none", malicious → "prompt_injection"
        df["threat_label"] = df["label"].map({0: THREAT2ID["none"], 1: THREAT2ID["prompt_injection"]})

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        df["threat_label"].tolist(),
        test_size=0.15,
        stratify=df["label"].tolist(),
        random_state=42,
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_train, y_train, t_train,
        test_size=0.12,
        stratify=y_train,
        random_state=42,
    )

    def make_ds(texts, labels, threat_labels):
        return Dataset.from_dict({
            "text": texts,
            "label": labels,
            "threat_label": threat_labels,
        })

    return DatasetDict({
        "train": make_ds(X_train, y_train, t_train),
        "validation": make_ds(X_val, y_val, t_val),
        "test": make_ds(X_test, y_test, t_test),
    })


def get_tokenize_fn(tokenizer, max_length: int = 512):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return tokenize


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Metrics for Trainer (operates on binary_logits = index 0 output)."""
    from sklearn.metrics import f1_score, confusion_matrix
    # eval_pred.predictions may be a tuple (binary_logits, threat_logits) or just logits
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="binary")
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


# ── Custom Trainer ────────────────────────────────────────────────────────────

class AegisMTTrainer(Trainer):
    """
    Extends Trainer to handle AegisMTModel's custom forward output.
    The model returns an _ForwardOutput object; Trainer expects (loss, logits).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        binary_labels = inputs.pop("labels")
        threat_labels = inputs.pop("threat_labels", None)
        outputs = model(
            **inputs,
            binary_labels=binary_labels,
            threat_labels=threat_labels,
        )
        loss = outputs.loss
        # Return (loss, binary_logits) so Trainer's eval loop gets the right logits
        return (loss, outputs.binary_logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        inputs.pop("threat_labels", None)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.binary_logits
        loss = None
        return loss, logits, labels


# ── Temperature Calibration ───────────────────────────────────────────────────

def fit_temperature(model, val_dataset, tokenizer, device: int, max_length: int = 512) -> float:
    """
    Find optimal temperature T* that minimises NLL on the validation set.
    Uses scipy optimise.minimize_scalar for efficiency.
    """
    from scipy.optimize import minimize_scalar
    import torch.nn.functional as F

    model.eval()
    all_logits = []
    all_labels = []

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, collate_fn=collator
    )

    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            batch.pop("threat_labels", None)
            batch = {k: v.to(f"cuda:{device}" if device >= 0 else "cpu") for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.binary_logits.cpu())
            all_labels.append(batch_labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    def nll(T):
        scaled = logits / T
        loss = F.cross_entropy(scaled, labels)
        return loss.item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_star = float(result.x)
    logger.info("Temperature calibration: T* = %.4f (val NLL = %.4f)", T_star, result.fun)
    return T_star


# ── Adversarial Augmentation ──────────────────────────────────────────────────

def augment_hard_negatives(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate adversarial augmentations of the hardest training examples.
    Applies invisible-char, token-smuggling, and many-shot patterns from
    the Phase 3 synthetic generators to create additional hard training examples.
    """
    from training.data.synthetic_gen import (
        _gen_invisible_char_injection,
        _gen_token_smuggling,
        _gen_many_shot_jailbreak,
        _gen_multilingual_evasion,
        _gen_sycophantic_setup,
        THREAT2ID as _T2I,
    )
    import random

    augmented = []
    n_aug = min(500, len(train_df) // 10)
    logger.info("Generating %d adversarial augmentation examples...", n_aug)

    generators = [
        (_gen_invisible_char_injection, "prompt_injection"),
        (_gen_token_smuggling, "prompt_injection"),
        (_gen_many_shot_jailbreak, "jailbreak"),
        (_gen_multilingual_evasion, "jailbreak"),
        (_gen_sycophantic_setup, "jailbreak"),
    ]

    for _ in range(n_aug):
        gen, cat = random.choice(generators)
        augmented.append({
            "text": gen(),
            "label": 1,
            "threat_label": THREAT2ID[cat],
        })

    return pd.concat([train_df, pd.DataFrame(augmented)], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


# ── Main Training Function ────────────────────────────────────────────────────

def train(
    model_key: str = "deberta-base",
    num_epochs: int = 12,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 512,
    use_adversarial_training: bool = False,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    base_model_name = MODEL_NAMES[model_key]
    logger.info("Phase 3 HF2 training")
    logger.info("Base model      : %s", base_model_name)
    logger.info("Epochs          : %d", num_epochs)
    logger.info("Batch size      : %d (eff. %d with grad. accum. × 8)", batch_size, batch_size * 8)
    logger.info("Adversarial aug : %s", use_adversarial_training)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_data()
    logger.info(
        "Dataset: train=%d  val=%d  test=%d",
        len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]),
    )

    # ── Resolve local snapshot ────────────────────────────────────────────────
    local_model_path = snapshot_download(base_model_name, local_files_only=True)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenize_fn = get_tokenize_fn(tokenizer, max_length=max_length)

    tokenised = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # Rename label columns for Trainer compatibility
    tokenised = tokenised.rename_column("label", "labels")
    tokenised = tokenised.rename_column("threat_label", "threat_labels")
    tokenised.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Build model ───────────────────────────────────────────────────────────
    aegis_model = AegisMTModel(base_model_name_or_path=local_model_path)
    model = aegis_model.build()

    # ── Device probe (same pattern as HFClassifier) ───────────────────────────
    device_id = -1
    if torch.cuda.is_available():
        try:
            t = torch.zeros(4).cuda()
            torch.isfinite(t)
            device_id = 0
            logger.info("GPU available — using CUDA/ROCm device 0")
        except RuntimeError as exc:
            logger.warning(
                "GPU detected but not functional (%s) — training on CPU.", exc
            )
    else:
        logger.info("No GPU detected — training on CPU (this will be very slow).")

    # ── Training arguments ─────────────────────────────────────────────────────
    # DeBERTa-v3 numerical instability in fp16/bf16 — always use fp32.
    training_args = TrainingArguments(
        output_dir=str(HF2_MODEL_OUTPUT),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="fnr",
        greater_is_better=False,
        logging_steps=50,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        gradient_accumulation_steps=8,  # effective batch = 128
        label_smoothing_factor=0.0,     # label smoothing is handled inside the model loss
        remove_unused_columns=False,    # keep threat_labels column
    )

    trainer = AegisMTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting Phase 3 main training...")
    trainer.train()

    # ── Optional adversarial fine-tuning pass ─────────────────────────────────
    if use_adversarial_training:
        logger.info("Starting adversarial fine-tuning pass (2 epochs)...")
        # Rebuild train dataset with augmented hard examples
        train_df = pd.DataFrame({
            "text": dataset["train"]["text"],
            "label": [int(l) for l in dataset["train"]["labels"]],
            "threat_label": [int(t) for t in dataset["train"]["threat_labels"]],
        })
        aug_df = augment_hard_negatives(train_df)
        aug_ds = Dataset.from_pandas(aug_df)
        aug_tokenised = aug_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        aug_tokenised = aug_tokenised.rename_column("label", "labels")
        aug_tokenised = aug_tokenised.rename_column("threat_label", "threat_labels")
        aug_tokenised.set_format("torch")

        adv_args = TrainingArguments(
            output_dir=str(HF2_MODEL_OUTPUT / "adv_checkpoint"),
            num_train_epochs=2,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate * 0.1,  # lower LR for fine-tuning
            weight_decay=0.1,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=False,
            bf16=False,
            report_to="none",
            dataloader_pin_memory=False,
            gradient_accumulation_steps=8,
            remove_unused_columns=False,
        )
        adv_trainer = AegisMTTrainer(
            model=model,
            args=adv_args,
            train_dataset=aug_tokenised,
            eval_dataset=tokenised["validation"],
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        adv_trainer.train()

    # ── Final test evaluation ──────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenised["test"])
    logger.info("Test results: %s", test_results)

    # ── Temperature calibration ───────────────────────────────────────────────
    logger.info("Fitting temperature scaling on validation set...")
    device = device_id
    T_star = fit_temperature(model, tokenised["validation"], tokenizer, device, max_length)

    # ── Save ───────────────────────────────────────────────────────────────────
    aegis_model._module = model
    aegis_model.save_pretrained(
        HF2_MODEL_OUTPUT,
        tokenizer=tokenizer,
        temperature_scaling=T_star,
    )
    logger.info("Phase 3 HF2 model saved to %s", HF2_MODEL_OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 HF2 multi-task classifier")
    parser.add_argument(
        "--model",
        choices=list(MODEL_NAMES.keys()),
        default="deberta-base",
        help="Base model to fine-tune (default: deberta-base)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length (use 128 for faster CPU runs)",
    )
    parser.add_argument(
        "--adversarial-training",
        action="store_true",
        help="Run a 2-epoch adversarial augmentation pass after main training",
    )
    args = parser.parse_args()

    train(
        model_key=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        use_adversarial_training=args.adversarial_training,
    )


if __name__ == "__main__":
    main()
