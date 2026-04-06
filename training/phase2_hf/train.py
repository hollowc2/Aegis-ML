"""
training/phase2_hf/train.py
=============================
Phase 2 training — Fine-tune DistilBERT (or DeBERTa-v3-small) with
optional 4-bit QLoRA for consumer GPU training.

Architecture:
  - Base: distilbert-base-uncased  (fast, 67M params)
  - Alt:  microsoft/deberta-v3-small (higher accuracy, 142M params)
  - Fine-tuning: full fine-tune OR QLoRA (4-bit) via PEFT
  - Labels: 0=BENIGN, 1=MALICIOUS

GPU requirements:
  - Full fine-tune DistilBERT: ~4 GB VRAM
  - QLoRA DistilBERT:          ~2 GB VRAM
  - Full fine-tune DeBERTa:    ~6 GB VRAM
  - QLoRA DeBERTa:             ~3 GB VRAM

Run:
    # Full fine-tune DistilBERT (default, fastest)
    python -m training.phase2_hf.train

    # QLoRA 4-bit (lower VRAM)
    python -m training.phase2_hf.train --qlora

    # Use DeBERTa-v3-small (higher accuracy)
    python -m training.phase2_hf.train --model deberta
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# AMD RDNA3 (gfx1101) compatibility — must be set before any torch import.
# PyTorch ROCm wheels compile for gfx1100 (11.0.0); override to use those kernels.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = Path("data/combined_dataset.csv")
MODELS_DIR = Path("models")
HF_MODEL_OUTPUT = MODELS_DIR / "hf_classifier"

MODEL_NAMES = {
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-small",
    "deberta-base": "microsoft/deberta-v3-base",
}

ID2LABEL = {0: "BENIGN", 1: "MALICIOUS"}
LABEL2ID = {"BENIGN": 0, "MALICIOUS": 1}


def load_data() -> DatasetDict:
    """Load and split the dataset into HuggingFace DatasetDict."""
    if not DATASET_CSV.exists():
        logger.info("Dataset not found — generating...")
        from training.data.prepare_dataset import main as prep_main
        prep_main()

    df = pd.read_csv(DATASET_CSV).dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.15, stratify=df["label"].tolist(), random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.12, stratify=y_train, random_state=42,
    )

    def make_ds(texts, labels):
        return Dataset.from_dict({"text": texts, "label": labels})

    return DatasetDict({
        "train": make_ds(X_train, y_train),
        "validation": make_ds(X_val, y_val),
        "test": make_ds(X_test, y_test),
    })


def get_tokenize_fn(tokenizer, max_length: int = 512):
    """Return a tokenisation function for dataset.map()."""
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,   # Dynamic padding via DataCollatorWithPadding
        )
    return tokenize


def compute_metrics(eval_pred):
    """Metrics for Trainer: accuracy, F1, FPR, FNR."""
    from sklearn.metrics import f1_score, confusion_matrix
    logits, labels = eval_pred
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


def apply_qlora(model, target_modules: list[str] | None = None):
    """Wrap the model with PEFT QLoRA configuration for 4-bit training."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    logger.info("Applying QLoRA 4-bit configuration...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Reload model with 4-bit quantisation
    # (The model was initially loaded in fp32; reload with quantisation)
    return bnb_config  # Return config to use during model load


def train(
    model_key: str = "deberta",
    use_qlora: bool = False,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 512,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    base_model_name = MODEL_NAMES[model_key]
    logger.info("Base model     : %s", base_model_name)
    logger.info("QLoRA 4-bit    : %s", use_qlora)
    logger.info("Epochs         : %d", num_epochs)
    logger.info("Batch size     : %d", batch_size)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_data()
    logger.info(
        "Dataset: train=%d  val=%d  test=%d",
        len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]),
    )

    # ── Resolve local snapshot path (avoids Hub file-list lookup on load) ────
    local_model_path = snapshot_download(base_model_name, local_files_only=True)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenize_fn = get_tokenize_fn(tokenizer, max_length=max_length)

    tokenised = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenised = tokenised.rename_column("label", "labels")
    tokenised.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ──────────────────────────────────────────────────────────────────
    model_kwargs: dict = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }

    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path, dtype=torch.float32, **model_kwargs
    )

    if use_qlora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Training arguments ─────────────────────────────────────────────────────
    # DeBERTa-v3's disentangled attention is numerically unstable in bf16/fp16;
    # run in fp32 regardless of hardware to avoid NaN gradients.
    _has_gpu = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(HF_MODEL_OUTPUT),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.1,           # increased from 0.01 — stronger L2 on small dataset
        warmup_ratio=0.06,          # ~6% warmup (was fixed 100 steps = 35% of training)
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="fnr",    # optimise for missed-injection rate directly
        greater_is_better=False,        # lower FNR is better
        logging_steps=50,
        fp16=False,   # DeBERTa-v3 is unstable in fp16/bf16; use fp32
        bf16=False,
        max_grad_norm=1.0,
        report_to="none",   # disable wandb/tensorboard by default
        dataloader_pin_memory=False,  # disabled: AMD HIP pin_memory causes kernel errors
        dataloader_num_workers=4,     # parallel CPU workers to keep GPU fed
        gradient_accumulation_steps=4 if batch_size <= 16 else 2,  # effective batch=64
        label_smoothing_factor=0.1,     # prevents overconfidence (was 0.0)
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Final test evaluation ──────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenised["test"])
    logger.info("Test results: %s", test_results)

    # ── Save ───────────────────────────────────────────────────────────────────
    trainer.save_model(str(HF_MODEL_OUTPUT))
    tokenizer.save_pretrained(str(HF_MODEL_OUTPUT))
    logger.info("HF model saved to %s", HF_MODEL_OUTPUT)

    # If QLoRA, also merge the adapter weights for easier inference
    if use_qlora:
        try:
            from peft import PeftModel
            logger.info("Merging QLoRA adapter weights into base model...")
            merged = model.merge_and_unload()
            merged_path = MODELS_DIR / "hf_classifier_merged"
            merged.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))
            logger.info("Merged model saved to %s", merged_path)
        except Exception as exc:
            logger.warning("Could not merge QLoRA adapters: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 HuggingFace classifier")
    parser.add_argument("--model", choices=["distilbert", "deberta", "deberta-base"], default="deberta")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA (lower VRAM)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for training (128 is faster on CPU)")
    args = parser.parse_args()

    train(
        model_key=args.model,
        use_qlora=args.qlora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
