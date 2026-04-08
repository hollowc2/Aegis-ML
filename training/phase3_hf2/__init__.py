"""
training/phase3_hf2
====================
Phase 3 (HF2 Ultra) multi-task DeBERTa-v3-base training pipeline.

Scripts:
  model.py       — AegisMTModel: DeBERTa encoder + binary head + threat-category head
  train.py       — Multi-task fine-tuning with adversarial augmentation
  evaluate.py    — Evaluation across all 15 adversarial categories + calibration
  export_onnx.py — FP32 + INT8-quantised ONNX export

Run after preparing the dataset:
  python -m training.data.prepare_dataset
  python -m training.phase3_hf2.train
  python -m training.phase3_hf2.evaluate --adversarial
  python -m training.phase3_hf2.export_onnx
"""
