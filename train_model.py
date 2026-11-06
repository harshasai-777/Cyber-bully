"""
Model Fine-Tuning Pipeline for Cyberbullying Detection.
Fine-tunes DistilBERT on the Surrey NLP Cyberbullying dataset using Hugging Face Trainer.

Usage:
    # Full Training (GPU recommended):
    python train_model.py --epochs 3 --batch-size 16 --lr 2e-5

    # Fast CPU Smoke-Test / Verification (100 train rows, 50 val rows):
    python train_model.py --quick-test --epochs 1 --batch-size 8
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# Add parent directory to sys.path so src imports work reliably
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    MODEL_NAME,
    DEFAULT_SAVED_MODEL_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    MAX_LENGTH
)
from src.data_utils import (
    load_raw_dataset,
    preprocess_and_tokenize,
    get_dataset_statistics
)
from src.model_utils import (
    get_model,
    get_tokenizer,
    compute_metrics
)


def set_reproducible_seed(seed: int = RANDOM_SEED):
    """Ensures deterministic behavior across Python, NumPy, PyTorch, and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    print(f"[Setup] Random seed set to {seed} for reproducibility.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Cyberbullying Detection.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size per device (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE, help="Evaluation batch size per device (default: 32)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Maximum sequence token length (default: 128)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_SAVED_MODEL_DIR), help="Directory to save the trained model")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick smoke test on a small subset (100 train, 50 val) to verify pipeline on CPU"
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit number of training samples (optional)")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit number of validation samples (optional)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    set_reproducible_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Device Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("CYBERBULLYING DETECTION MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Hardware Device     : {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Device Name     : {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory (Total)  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Note: Running on CPU. For fast training on full 35,000+ dataset, a GPU (or Google Colab) is recommended.")
    print(f"Base Pretrained Model: {MODEL_NAME}")
    print(f"Target Output Path   : {output_dir}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch Size (Train)   : {args.batch_size}")
    print(f"Learning Rate        : {args.lr}")
    print("=" * 70)

    # 2. Load Raw Dataset
    raw_dataset = load_raw_dataset()

    # Apply quick-test subset if specified
    if args.quick_test:
        print("\n[Mode] Quick-test mode enabled: Selecting small subset for pipeline verification...")
        train_n = min(100, len(raw_dataset["train"]))
        val_n = min(50, len(raw_dataset["validation"]))
        raw_dataset["train"] = raw_dataset["train"].select(range(train_n))
        raw_dataset["validation"] = raw_dataset["validation"].select(range(val_n))
    elif args.max_train_samples:
        train_n = min(args.max_train_samples, len(raw_dataset["train"]))
        val_n = min(args.max_eval_samples or max(100, int(train_n * 0.15)), len(raw_dataset["validation"]))
        raw_dataset["train"] = raw_dataset["train"].select(range(train_n))
        raw_dataset["validation"] = raw_dataset["validation"].select(range(val_n))
        print(f"[Data] Training limited to {train_n} samples, validation to {val_n} samples.")

    # Save dataset stats
    stats_df = get_dataset_statistics(raw_dataset)
    stats_path = RESULTS_DIR / "dataset_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[Data] Saved dataset statistics to '{stats_path}'.")

    # 3. Tokenize Dataset
    tokenizer = get_tokenizer(MODEL_NAME)
    tokenized_dataset, tokenizer = preprocess_and_tokenize(
        raw_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    train_data = tokenized_dataset["train"]
    val_data = tokenized_dataset["validation"]

    print(f"[Data] Final Processed Train Rows : {len(train_data):,}")
    print(f"[Data] Final Processed Val Rows   : {len(val_data):,}")

    # 4. Initialize Model
    model = get_model(MODEL_NAME)

    # 5. Configure Training Arguments
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=0 if args.quick_test else WARMUP_STEPS,
        logging_steps=LOGGING_STEPS if not args.quick_test else 5,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(), # Use mixed precision on GPU
        report_to="none"                # Keep clean, no wandb needed
    )

    # 6. Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if args.epochs > 2 else []
    )

    # 7. Execute Fine-Tuning
    print("\n[Training] Starting DistilBERT fine-tuning...")
    train_result = trainer.train()

    # 8. Run Validation Evaluation
    print("\n[Evaluation] Evaluating best model on validation split...")
    eval_metrics = trainer.evaluate()
    print("Validation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

    # 9. Save Best Model and Tokenizer
    print(f"\n[Save] Saving fine-tuned model and tokenizer to '{output_dir}'...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 10. Record Training Summary
    summary = {
        "model_name": MODEL_NAME,
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": device.type,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0.0),
        "train_loss": train_result.metrics.get("train_loss", 0.0),
        "validation_metrics": eval_metrics
    }
    summary_path = RESULTS_DIR / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[Save] Saved training summary to '{summary_path}'.")

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETED SUCCESSFULLY!")
    print(f"Trained model artifacts are stored in: {output_dir}")
    print("Next step: Run test evaluation using:")
    print(f"  python evaluate_model.py --model-dir \"{output_dir}\"")
    print("Or launch the interactive web application using:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
