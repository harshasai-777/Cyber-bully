"""
Model Evaluation Pipeline for Cyberbullying Detection.
Evaluates the fine-tuned DistilBERT model on the held-out test split.
Generates classification reports, confusion matrix visual, and metrics JSON/CSV.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model-dir models/distilbert-cyberbullying --max-samples 500
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_SAVED_MODEL_DIR,
    RESULTS_DIR,
    DATASET_NAME,
    TEXT_COLUMN,
    LABEL_COLUMN,
    LABELS,
    LABEL2ID,
    ID2LABEL,
    MAX_LENGTH
)
from src.data_utils import load_raw_dataset, clean_and_validate_split


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DistilBERT model on test set.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_SAVED_MODEL_DIR),
        help="Path to the saved model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save evaluation artifacts"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of test samples to evaluate (useful for quick evaluation on CPU)"
    )
    return parser.parse_args()


def plot_and_save_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    """Generates and saves a high-resolution Confusion Matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax, colorbar=True)
    
    plt.title("Confusion Matrix - Cyberbullying Detection (Test Set)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Results] Confusion matrix saved to: {output_path}")


def main():
    args = parse_arguments()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CYBERBULLYING DETECTION MODEL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Model Directory     : {model_dir}")
    print(f"Results Directory   : {output_dir}")

    # Check model directory existence
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"\n[Error] Model directory not found or empty: {model_dir}")
        print("Please fine-tune the model first by running:")
        print("    python train_model.py")
        print("Or run a quick smoke-test training:")
        print("    python train_model.py --quick-test")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device   : {device.type.upper()}")

    # 1. Load Model and Tokenizer
    print("\n[Loading] Loading saved model and tokenizer...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[Error] Failed to load model from '{model_dir}': {e}")
        sys.exit(1)

    # 2. Load and Clean Held-Out Test Split
    raw_dataset = load_raw_dataset(DATASET_NAME)
    test_split = clean_and_validate_split(raw_dataset["test"], "test")

    if args.max_samples:
        test_split = test_split.select(range(min(args.max_samples, len(test_split))))
        print(f"[Data] Limiting evaluation to {len(test_split)} samples for speed.")

    test_texts = [ex[TEXT_COLUMN] for ex in test_split]
    test_true_labels_str = [ex[LABEL_COLUMN] for ex in test_split]
    test_true_ids = [LABEL2ID[lbl] for lbl in test_true_labels_str]

    total_samples = len(test_texts)
    print(f"[Data] Total test samples to evaluate: {total_samples:,}")

    # 3. Batch Inference
    print("\n[Inference] Running batch inference across test set...")
    all_pred_ids = []
    all_confidences = []

    for i in range(0, total_samples, args.batch_size):
        batch_texts = test_texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            confs = torch.max(probs, dim=-1).values.cpu().numpy()

        all_pred_ids.extend(preds)
        all_confidences.extend(confs)

    all_pred_ids = np.array(all_pred_ids)
    test_true_ids = np.array(test_true_ids)

    # 4. Compute Metrics
    acc = accuracy_score(test_true_ids, all_pred_ids)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_true_ids, all_pred_ids, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        test_true_ids, all_pred_ids, average="weighted", zero_division=0
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy       : {acc * 100:.2f}%")
    print(f"Macro Precision        : {macro_precision * 100:.2f}%")
    print(f"Macro Recall           : {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score         : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score      : {weighted_f1 * 100:.2f}%")
    print("-" * 70)

    # 5. Classification Report
    target_names = [ID2LABEL[i] for i in range(len(LABELS))]
    report_dict = classification_report(
        test_true_ids,
        all_pred_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        test_true_ids,
        all_pred_ids,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print("Per-Category Detailed Classification Report:")
    print(report_text)

    # 6. Save Artifacts
    # JSON Metrics
    metrics_summary = {
        "dataset_name": DATASET_NAME,
        "test_samples_evaluated": total_samples,
        "overall_accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_category": {
            lbl: {
                "precision": report_dict[lbl]["precision"],
                "recall": report_dict[lbl]["recall"],
                "f1-score": report_dict[lbl]["f1-score"],
                "support": report_dict[lbl]["support"]
            }
            for lbl in target_names
        }
    }
    metrics_json_path = output_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Results] Metrics summary saved to: {metrics_json_path}")

    # CSV Classification Report
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"[Results] Classification report saved to: {report_csv_path}")

    # Confusion Matrix Image
    cm_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(test_true_ids, all_pred_ids, target_names, cm_path)

    print("=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
