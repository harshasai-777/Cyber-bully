"""
Configuration module for the Cyberbullying Detection Project.
Defines model parameters, dataset paths, label mappings, and directory structures.
"""

import os
from pathlib import Path

# Base Paths (dynamically resolved relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure essential output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Default path for the fine-tuned model checkpoint
DEFAULT_SAVED_MODEL_DIR = MODELS_DIR / "distilbert-cyberbullying"

# Dataset Configuration
DATASET_NAME = "surrey-nlp/Cyberbullying-Detection-CB1"
TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "cyberbullying_type"

# Verified 6 categories from surrey-nlp/Cyberbullying-Detection-CB1
LABELS = [
    "not_cyberbullying",
    "gender",
    "religion",
    "other_cyberbullying",
    "age",
    "ethnicity"
]

# Consistent ID mappings
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# User-friendly descriptions for UI presentation and college defense
LABEL_DESCRIPTIONS = {
    "not_cyberbullying": "Neutral, supportive, or conversational content with no hostile intent.",
    "gender": "Hostile, harassing, or sexist remarks targeted on the basis of gender or sexual orientation.",
    "religion": "Hate speech, intolerance, or derogatory attacks targeting religious beliefs or communities.",
    "other_cyberbullying": "General cyberbullying, toxic personal insults, or trolling not tied to specific demographic groups.",
    "age": "Discriminatory or bullying comments targeting someone based on their age (ageism).",
    "ethnicity": "Racist remarks, slurs, or derogatory comments targeting racial or ethnic identity."
}

# Model & Tokenizer Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128  # Covers over 95% of social media comments without excessive padding

# Training Hyperparameters
RANDOM_SEED = 42
LEARNING_RATE = 2e-5
BATCH_SIZE = 16          # Recommended for GPU (reduce to 8 or 4 on low VRAM / CPU)
EVAL_BATCH_SIZE = 32
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 2
