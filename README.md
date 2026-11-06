# Design and Development of a Working Prototype for Cyberbullying Detection Using a Fine-Tuned Transformer Classifier

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Academic-College%20Mini--Project-green.svg)]()

A complete, production-grade academic mini-project prototype that fine-tunes **DistilBERT** to detect and classify cyberbullying in social media comments across 6 granular categories. Includes an automated training pipeline, rigorous evaluation suite, and a modern, interactive **Streamlit** web application.

---

## 1. Project Overview & Problem Statement

Cyberbullying on digital platforms causes severe psychological distress. Traditional rule-based filters and classic machine learning models (TF-IDF + SVM) fail to recognize context, sarcasm, and disguised harassment. 

This project solves this by fine-tuning **DistilBERT** (`distilbert-base-uncased`) on the standardized benchmark dataset [`surrey-nlp/Cyberbullying-Detection-CB1`](https://huggingface.co/datasets/surrey-nlp/Cyberbullying-Detection-CB1). It categorizes comments into six mutually exclusive classes:
1. `not_cyberbullying` (Safe / Conversational)
2. `gender` (Sexist & gender harassment)
3. `religion` (Religious hate speech)
4. `other_cyberbullying` (General insults and toxicity)
5. `age` (Ageist harassment)
6. `ethnicity` (Racist and xenophobic remarks)

---

## 2. Key Features

- **Fine-Tuned Transformer Engine:** DistilBERT sequence classifier with subword WordPiece tokenization and custom classification head.
- **Reproducible Pipeline:** Deterministic random seeds across Python, NumPy, and PyTorch.
- **Evaluation Suite:** Computes Accuracy, Macro Precision, Macro Recall, Macro F1, generates confusion matrix visualization and classification reports.
- **Interactive Web UI:** Modern Streamlit application with:
  - Real-time comment analysis with input validation.
  - Probability bar charts across all 6 classes.
  - Model confidence metrics.
  - Interactive demo presets for instant college viva demonstrations.
  - Inspection of underlying token IDs and attention tokens (`[CLS]`, `[SEP]`).
  - Clear academic disclaimers and human-in-the-loop guidelines.
- **College Documentation Included:** Full mini-project report (`docs/project_report.md`) and a 25-question viva examination guide (`docs/viva_questions.md`).

---

## 3. Technology Stack

- **Deep Learning:** PyTorch, Hugging Face `transformers`, `datasets`, `accelerate`
- **Model Architecture:** `distilbert-base-uncased` (6 layers, 66M parameters)
- **Web Interface:** Streamlit
- **Metrics & Analysis:** Scikit-learn, Pandas, NumPy, Matplotlib

---

## 4. Repository Structure

```text
cyberbullying-detection/
├── app.py                   # Interactive Streamlit Web Application
├── train_model.py           # Training & fine-tuning pipeline with Trainer
├── evaluate_model.py        # Independent evaluation on held-out test split
├── inspect_dataset.py       # Quick inspection utility for dataset schema
├── requirements.txt         # Project dependencies
├── README.md                # Main documentation & walkthrough
├── .gitignore               # Excludes caches, large binaries, checkpoints
├── data/
│   └── README.md            # Dataset documentation, schema, and splits
├── models/
│   └── .gitkeep             # Destination for trained model checkpoints
├── results/
│   └── .gitkeep             # Destination for evaluation reports and confusion matrix
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configurations, hyperparams, label maps
│   ├── data_utils.py        # Dataset loading, cleaning, and tokenization
│   └── model_utils.py       # Model instantiation, metrics, and inference logic
└── docs/
    ├── project_report.md    # Formal college mini-project report
    └── viva_questions.md    # 25 Viva examination Q&A preparation guide
```

---

## 5. Installation & Setup (Windows)

### Prerequisites
- Python 3.10+ installed and added to `PATH`.

### Step 1: Open Terminal in Project Directory
```powershell
cd C:\Cyber-bully
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. How to Run

### Step 1: Inspect Dataset (Optional Verification)
Verify dataset access and view schema:
```powershell
python inspect_dataset.py
```

---

### Step 2: Train the Model

#### Option A: Quick Smoke-Test (Recommended for fast CPU demonstration / 2 minutes)
To verify the full pipeline end-to-end on a local CPU without waiting hours:
```powershell
python train_model.py --quick-test --epochs 1 --batch-size 8
```

#### Option B: Full Dataset Training (Recommended for GPU / Google Colab)
Fine-tunes DistilBERT across all 35,769 training rows for 3 epochs:
```powershell
python train_model.py --epochs 3 --batch-size 16 --lr 2e-5
```
*Trained model artifacts will be saved to `models/distilbert-cyberbullying/`.*

---

### Step 3: Evaluate on Held-Out Test Set
Evaluate the saved checkpoint against the 9,923 test split samples:
```powershell
# Full evaluation on all test samples:
python evaluate_model.py

# Or fast test evaluation (e.g., 200 samples on CPU):
python evaluate_model.py --max-samples 200
```
This generates:
- `results/metrics.json` (Overall accuracy and macro metrics)
- `results/classification_report.csv` (Per-category precision, recall, F1)
- `results/confusion_matrix.png` (High-resolution confusion matrix plot)

---

### Step 4: Launch the Streamlit Web Application
Start the interactive UI:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Model directory not found or empty` | The model has not been trained yet. | Run `python train_model.py --quick-test` to generate the checkpoint. |
| `CUDA out of memory` | Batch size too large for GPU VRAM. | Reduce batch size: `python train_model.py --batch-size 8`. |
| `Training is very slow on CPU` | CPU lacks tensor parallelism. | Use `--quick-test` locally or run training on a free Google Colab GPU instance. |
| `Hugging Face download timeout` | Unstable internet connection. | Re-run script; Hugging Face will resume download from cache. |

---

## 8. Limitations & Future Scope

- **Nuance & Sarcasm:** Deep models may struggle with indirect, sarcastic, or veiled insults.
- **Multimodal Content:** The current system analyzes text only; extending to memes and video moderation is a future enhancement.
- **Human-in-the-Loop:** High model confidence should assist human review teams rather than acting as an autonomous censor.

---

## 9. Viva Preparation
Review [`docs/viva_questions.md`](file:///C:/Cyber-bully/docs/viva_questions.md) for 25 technical questions and answers covering transformers, DistilBERT, evaluation metrics, and ethical challenges.
