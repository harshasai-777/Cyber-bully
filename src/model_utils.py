"""
Model initialization, evaluation metrics, and inference utility functions.
Supports model building, training metric computations, and single-text prediction.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from src.config import (
        MODEL_NAME,
        NUM_LABELS,
        ID2LABEL,
        LABEL2ID,
        LABELS,
        MAX_LENGTH
    )
except ImportError:
    from config import (
        MODEL_NAME,
        NUM_LABELS,
        ID2LABEL,
        LABEL2ID,
        LABELS,
        MAX_LENGTH
    )


def get_model(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None
) -> AutoModelForSequenceClassification:
    """
    Initializes a pretrained DistilBERT sequence classification model.
    Attaches a custom classification head corresponding to the number of classes.
    """
    if id2label is None:
        id2label = ID2LABEL
    if label2id is None:
        label2id = LABEL2ID

    print(f"[Model] Initializing classification model '{model_name}' with {num_labels} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model


def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """
    Loads the pretrained DistilBERT tokenizer.
    """
    print(f"[Tokenizer] Loading tokenizer for '{model_name}'...")
    return AutoTokenizer.from_pretrained(model_name)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Computes classification metrics for evaluation during training:
    - Overall Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-Score
    Macro metrics are essential for multi-class classification to ensure fair evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }


def predict_single_comment(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: Optional[torch.device] = None,
    max_length: int = MAX_LENGTH
) -> Dict[str, Any]:
    """
    Executes inference for a single input text comment.
    Returns:
    - predicted_class: string label
    - confidence: float probability (0.0 to 1.0)
    - probabilities: dict of {category_name: probability}
    - tokens: tokenized input representation for educational inspection
    """
    if not text or text.strip() == "":
        raise ValueError("Input text cannot be empty or solely whitespace.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_id = int(np.argmax(probabilities))
    predicted_label = model.config.id2label.get(predicted_id, str(predicted_id))
    confidence = float(probabilities[predicted_id])

    prob_dict = {
        model.config.id2label.get(i, str(i)): float(prob)
        for i, prob in enumerate(probabilities)
    }

    # Extract tokens for educational viva walkthrough
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    return {
        "text": text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "tokens": tokens,
        "device": str(device)
    }
