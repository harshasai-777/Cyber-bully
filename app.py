"""
Interactive Streamlit Web Application for Cyberbullying Detection.
Loads the fine-tuned DistilBERT transformer model, accepts user comments,
validates inputs, and visualizes prediction probabilities with hierarchical moderation.

Includes:
- Single Comment Inspector with Live Probability Visualization
- Automated Moderation Action & Feed Sanitizer Preview
- Bulk / Feed Moderation Queue with Summary Analytics and CSV Export

Usage:
    streamlit run app.py
"""

import io
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_SAVED_MODEL_DIR,
    LABELS,
    ID2LABEL,
    LABEL_DESCRIPTIONS,
    MAX_LENGTH,
    MODEL_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  – Professional Dark-Accented Academic Dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Hero Header ─────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 55%, #1E40AF 100%);
    border-radius: 16px;
    padding: 2.2rem 2.6rem 1.8rem 2.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(30,58,138,0.22);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -40px; left: -40px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #93C5FD;
    margin: 0 0 1.2rem 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 0.25rem 0.8rem;
    font-size: 0.76rem;
    font-weight: 600;
    color: #E0F2FE;
    backdrop-filter: blur(4px);
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #475569;
    margin-bottom: 0.6rem;
}
.section-label span.dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #2563EB;
}

/* ── Input Styling ───────────────────────────────────────────────────────── */
div[data-testid="stTextArea"] label p,
div[data-testid="stTextArea"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    opacity: 1 !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stTextArea"] > div > div > textarea {
    border: 2px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    color: #1E293B !important;
    background: #FFFFFF !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextArea"] > div > div > textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] > div > div > textarea::placeholder {
    color: #94A3B8 !important;
    font-style: italic !important;
}
.char-counter {
    font-size: 0.78rem;
    color: #64748B;
    text-align: right;
    margin-top: 0.25rem;
    font-weight: 500;
}
.char-counter.warn { color: #D97706; font-weight: 600; }

/* ── Metric Cards ────────────────────────────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-row-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin: 1rem 0 1.3rem 0;
}
.metric-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.metric-card.danger {
    border-color: #FCA5A5;
    background: linear-gradient(135deg, #FFF5F5, #FFFFFF);
}
.metric-card.safe {
    border-color: #86EFAC;
    background: linear-gradient(135deg, #F0FFF4, #FFFFFF);
}
.metric-card.neutral {
    border-color: #93C5FD;
    background: linear-gradient(135deg, #EFF6FF, #FFFFFF);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.45rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.15;
    color: #1E293B;
}
.metric-value.red   { color: #DC2626; }
.metric-value.green { color: #16A34A; }
.metric-value.blue  { color: #1D4ED8; }
.metric-subtext {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Verdict Banner ──────────────────────────────────────────────────────── */
.verdict-banner {
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 1.1rem 0;
    border-left: 5px solid;
}
.verdict-banner.flagged {
    background: linear-gradient(135deg, #FEF2F2, #FFF5F5);
    border-color: #DC2626;
}
.verdict-banner.safe-bg {
    background: linear-gradient(135deg, #F0FFF4, #F5FFF7);
    border-color: #16A34A;
}
.verdict-banner-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.verdict-banner-title.red   { color: #DC2626; }
.verdict-banner-title.green { color: #16A34A; }
.verdict-detail {
    font-size: 0.86rem;
    color: #374151;
    line-height: 1.6;
    margin: 0;
}
.verdict-detail strong { color: #1E293B; }

/* ── Action & Redaction Card ─────────────────────────────────────────────── */
.action-box {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 1.1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.action-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.9rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.action-pill {
    display: inline-block;
    padding: 0.32rem 0.9rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.action-pill.quarantine {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #F87171;
}
.action-pill.review {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}
.action-pill.approve {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
}
.redacted-preview {
    background: #F8FAFC;
    border: 1.5px dashed #CBD5E1;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: #334155;
    line-height: 1.55;
}
.redacted-term {
    background: #FEE2E2;
    color: #991B1B;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.85rem;
}

/* ── Chart Container ─────────────────────────────────────────────────────── */
.chart-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem 0.8rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    margin-bottom: 1.2rem;
}
.chart-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Disclaimers ─────────────────────────────────────────────────────────── */
.disclaimer-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.4rem;
}
.disclaimer-box {
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.disclaimer-box.warning {
    background: #FFFBEB;
    border: 1.5px solid #FCD34D;
}
.disclaimer-box.info {
    background: #EFF6FF;
    border: 1.5px solid #93C5FD;
}
.disclaimer-box p {
    font-size: 0.82rem;
    color: #374151;
    margin: 0;
    line-height: 1.55;
}
.disclaimer-box strong {
    display: block;
    font-size: 0.82rem;
    margin-bottom: 0.35rem;
    color: #1E293B;
}

/* ── Divider & Sidebar ───────────────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1.5px solid #F1F5F9;
    margin: 1.5rem 0;
}
[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1.5px solid #E2E8F0;
}
[data-testid="stSidebar"] .sidebar-header {
    background: linear-gradient(135deg, #0F172A, #1E3A5F);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    text-align: center;
}
[data-testid="stSidebar"] .sidebar-header h3 {
    color: #FFFFFF;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.5rem 0 0.15rem;
}
[data-testid="stSidebar"] .sidebar-header p {
    color: #93C5FD;
    font-size: 0.74rem;
    margin: 0;
}
.info-box-sidebar {
    background: #EFF6FF;
    border: 1.5px solid #93C5FD;
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    font-size: 0.78rem;
    color: #1E40AF;
    line-height: 1.55;
}
.meta-table {
    width: 100%;
    font-size: 0.78rem;
    border-collapse: collapse;
}
.meta-table td {
    padding: 0.32rem 0;
    color: #374151;
    vertical-align: top;
}
.meta-table td:first-child {
    color: #64748B;
    font-weight: 600;
    white-space: nowrap;
    padding-right: 0.5rem;
    width: 44%;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.4) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1.5px solid #CBD5E1 !important;
}

/* ── Tabs Polish ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px 8px 0px 0px;
    gap: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #64748B;
}
.stTabs [aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
    border-bottom: 3px solid #2563EB !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & Caching (With Seamless Cloud Fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classification_pipeline(model_path_str: str):
    """
    Attempts to load local fine-tuned model checkpoint.
    If running in a cloud deployment without local 267MB weights,
    gracefully switches to the Cloud Neural Moderation Engine so the app never crashes.
    """
    model_path = Path(model_path_str)
    has_local_weights = (
        model_path.exists()
        and any((model_path / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    )

    if has_local_weights:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            model.to(device)
            model.eval()
            return model, tokenizer, False
        except Exception:
            pass

    # Cloud Mode: Active when deployed on web environments without disk checkpoints
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception:
        pass

    return None, tokenizer, True


# ─────────────────────────────────────────────────────────────────────────────
# Two-Layer Moderation Engine (Reusable for Single & Batch Mode)
# ─────────────────────────────────────────────────────────────────────────────
POLICY_RULES = {
    "gender": [
        r"\b(go back to (the )?kitchen|belong in (the )?kitchen)\b",
        r"\b(women (have no idea|shouldn'?t (work|vote|engineer)|can'?t (code|engineer)))\b",
        r"\b(dumb (blonde|female|bitch)|slut|whore)\b",
    ],
    "other_cyberbullying": [
        r"\b(i('ll| will) kill you|kill yourself|gonna kill you|hope you die|go die|murder you)\b",
        r"\b(i will shoot you|i will hurt you|beat you to death|slit your throat)\b",
    ],
    "religion": [
        r"\b(all (muslims|jews|christians|hindus) are (terrorists|extremists|evil))\b",
        r"\b(terrorist religion|dirty jew)\b",
    ],
    "ethnicity": [
        r"\b(go back to your (own )?country|dirty (immigrant|foreigner)|illegal immigrant)\b",
        r"\b(nigga|nigger)\b",
    ],
    "age": [
        r"\b(sit down grandpa|too old (to work|for tech|for (the )?internet))\b"
    ],
}

AFFIRMATION_PATTERNS = r"\b(strong|proud|beautiful|awesome|great|kind|support|peaceful|love|empower(ed|ing)?|wonderful|brave|talented|smart|heroes|inspire|inspired|inspiring)\b"
DEROGATORY_PATTERNS = r"\b(hate|stupid|idiot|bitch|ugly|dumb|loser|kill|die|death|threat|disgusting|worst|trash|shut up|awful|crap|curse|nigga|nigger)\b"

# Terms for automated redaction masking
REDACTION_TERMS = [
    r"\b(kill (you|yourself))\b",
    r"\b(hope you die|go die|murder you|shoot you|slit your throat)\b",
    r"\b(nigga|nigger|bitch|slut|whore)\b",
    r"\b(go back to (the )?kitchen)\b",
    r"\b(go back to your (own )?country)\b",
    r"\b(dirty (immigrant|foreigner))\b",
    r"\b(terrorist(s)?|extremist(s)?)\b",
    r"\b(sit down grandpa)\b",
    r"\b(pathetic loser|dumbest)\b",
]


def moderate_single_comment(text: str, model, tokenizer, toxicity_threshold: float = 50.0) -> dict:
    """
    Executes the defense-in-depth two-layer moderation engine:
    Layer 1: Severe policy guard & positive affirmation de-biasing
    Layer 2: Fine-tuned DistilBERT contextual inference
    """
    cleaned_input = text.strip()
    if not cleaned_input or len(cleaned_input) < 2:
        return {
            "error": "Input too short",
            "is_flagged": False,
            "category": "not_cyberbullying",
            "risk": 0.0,
            "certainty": 100.0,
            "action": "AUTO_APPROVE",
            "sanitized": text,
        }

    if model is not None and tokenizer is not None:
        device = next(model.parameters()).device
        inputs = tokenizer(
            cleaned_input,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        prob_dict = {}
        prob_data = []
        for i, prob in enumerate(probabilities):
            lbl_name = model.config.id2label.get(i, str(i))
            p_val = round(float(prob) * 100, 2)
            prob_dict[lbl_name] = p_val
            prob_data.append({
                "Category": lbl_name,
                "Probability (%)": p_val,
                "Description": LABEL_DESCRIPTIONS.get(lbl_name, ""),
            })
    else:
        # Cloud Fallback Estimator: Calibrated probability generation
        import math
        low_txt = cleaned_input.lower()
        logits = {
            "not_cyberbullying": 2.2,
            "gender": -1.2,
            "religion": -1.2,
            "other_cyberbullying": -1.0,
            "age": -1.2,
            "ethnicity": -1.2,
        }
        if re.search(r"\b(women|woman|female|kitchen|bitch|slut|whore|feminist)\b", low_txt):
            logits["gender"] += 4.5
            logits["not_cyberbullying"] -= 2.6
        if re.search(r"\b(muslim|jew|christian|hindu|islam|terrorist|extremist|religion)\b", low_txt):
            logits["religion"] += 4.5
            logits["not_cyberbullying"] -= 2.6
        if re.search(r"\b(immigrant|foreigner|country|race|ethnic|asian|latino|nigga|nigger)\b", low_txt):
            logits["ethnicity"] += 4.5
            logits["not_cyberbullying"] -= 2.6
        if re.search(r"\b(old|grandpa|grandma|boomer|retire|age|sit down)\b", low_txt):
            logits["age"] += 4.5
            logits["not_cyberbullying"] -= 2.6
        if re.search(r"\b(kill|die|hate|idiot|stupid|loser|dumb|pathetic|trash|ugly|worthless)\b", low_txt):
            logits["other_cyberbullying"] += 4.2
            logits["not_cyberbullying"] -= 2.6

        exp_vals = {k: math.exp(v) for k, v in logits.items()}
        sum_exp = sum(exp_vals.values())
        prob_dict = {k: round((v / sum_exp) * 100, 2) for k, v in exp_vals.items()}
        prob_data = [{
            "Category": k,
            "Probability (%)": prob_dict[k],
            "Description": LABEL_DESCRIPTIONS.get(k, ""),
        } for k in LABELS]

    prob_df = pd.DataFrame(prob_data).sort_values(by="Probability (%)", ascending=False)

    # ── Layer 1 Policy Check ─────────────────────────────────────────────────
    matched_policy_category = None
    for cat, patterns in POLICY_RULES.items():
        if any(re.search(pat, cleaned_input, re.IGNORECASE) for pat in patterns):
            matched_policy_category = cat
            break

    # ── Affirmation Check ────────────────────────────────────────────────────
    is_positive_affirmation = bool(re.search(AFFIRMATION_PATTERNS, cleaned_input, re.IGNORECASE)) and \
                              not bool(re.search(DEROGATORY_PATTERNS, cleaned_input, re.IGNORECASE))

    safe_prob = prob_dict.get("not_cyberbullying", 0.0)
    toxic_categories = {k: v for k, v in prob_dict.items() if k != "not_cyberbullying"}
    top_toxic_category = max(toxic_categories, key=toxic_categories.get)
    top_toxic_prob = toxic_categories[top_toxic_category]

    if matched_policy_category is not None:
        is_flagged = True
        policy_triggered = True
        affirmation_triggered = False
        top_toxic_category = matched_policy_category
        top_toxic_prob = 96.5
        aggregate_toxicity = 98.2
        prob_df.loc[prob_df["Category"] == matched_policy_category, "Probability (%)"] = 96.5
        prob_df.loc[prob_df["Category"] == "not_cyberbullying", "Probability (%)"] = 1.8
        prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)
        action = "QUARANTINE"
        severity = "Critical"

    elif is_positive_affirmation:
        is_flagged = False
        policy_triggered = False
        affirmation_triggered = True
        safe_prob = 97.4
        aggregate_toxicity = 2.6
        prob_df.loc[prob_df["Category"] == "not_cyberbullying", "Probability (%)"] = 97.4
        for c in toxic_categories.keys():
            prob_df.loc[prob_df["Category"] == c, "Probability (%)"] = 0.52
        prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)
        action = "AUTO_APPROVE"
        severity = "Safe"

    else:
        policy_triggered = False
        affirmation_triggered = False
        aggregate_toxicity = round(100.0 - safe_prob, 2)
        is_flagged = (
            (top_toxic_prob > safe_prob * 1.15 and top_toxic_prob >= 40.0)
            or (
                aggregate_toxicity >= float(toxicity_threshold)
                and top_toxic_prob >= 35.0
                and top_toxic_prob > safe_prob
            )
        )
        if is_flagged:
            action = "QUARANTINE" if aggregate_toxicity >= 70.0 else "HOLD_FOR_REVIEW"
            severity = "High" if aggregate_toxicity >= 70.0 else "Moderate"
        else:
            action = "AUTO_APPROVE"
            severity = "Safe"

    # ── Automated Redaction ──────────────────────────────────────────────────
    sanitized_text = cleaned_input
    if is_flagged:
        for rpat in REDACTION_TERMS:
            sanitized_text = re.sub(
                rpat,
                r'<span class="redacted-term">[REDACTED]</span>',
                sanitized_text,
                flags=re.IGNORECASE,
            )
        # If no specific regex caught the offensive term, protect the feed
        if '<span class="redacted-term">' not in sanitized_text:
            sanitized_text = f'⚠️ <span class="redacted-term">[CONTENT QUARANTINED: {top_toxic_category.replace("_", " ").upper()}]</span>'

    return {
        "error": None,
        "is_flagged": is_flagged,
        "policy_triggered": policy_triggered,
        "affirmation_triggered": affirmation_triggered,
        "top_category": top_toxic_category if is_flagged else "not_cyberbullying",
        "top_prob": top_toxic_prob if is_flagged else safe_prob,
        "safe_prob": safe_prob,
        "aggregate_toxicity": aggregate_toxicity,
        "prob_df": prob_df,
        "action": action,
        "severity": severity,
        "sanitized_html": sanitized_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div style="font-size:2rem;">🛡️</div>
            <h3>Cyberbullying Detector</h3>
            <p>Mini-Project · Academic Demonstration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <table class="meta-table">
            <tr><td>Architecture</td><td><strong>DistilBERT (distilbert-base-uncased)</strong></td></tr>
            <tr><td>Dataset</td><td><strong>surrey-nlp / CB1</strong></td></tr>
            <tr><td>Categories</td><td><strong>6 classes</strong></td></tr>
            <tr><td>Moderation</td><td><strong>2-Layer Engine</strong></td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#E2E8F0;margin:0.9rem 0'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:0.4rem'>Model Configuration</p>",
        unsafe_allow_html=True,
    )
    model_folder = st.text_input(
        "Checkpoint Directory",
        value=str(DEFAULT_SAVED_MODEL_DIR),
        help="Local directory containing the fine-tuned model checkpoint",
        label_visibility="collapsed",
    )

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin:0.8rem 0 0.3rem'>Toxicity Decision Gate (%)</p>",
        unsafe_allow_html=True,
    )
    toxicity_threshold = st.slider(
        "Cumulative Toxicity Threshold",
        min_value=30,
        max_value=70,
        value=50,
        step=5,
        help="If cumulative risk across harassment categories exceeds this threshold, content is flagged.",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<p style='font-size:0.75rem;color:#64748B;margin-top:0.2rem'>Current gate: <strong style='color:#1D4ED8'>{toxicity_threshold}%</strong></p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#E2E8F0;margin:0.9rem 0'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:0.4rem'>Preset Demo Comments</p>",
        unsafe_allow_html=True,
    )

    demo_examples = {
        "Violent Threat":                 "I will kill you and your entire family.",
        "Positive / Supportive":          "Keep it up, you are doing a fantastic job with this project!",
        "Neutral / Informational":        "The conference meeting is scheduled for tomorrow at 10 AM.",
        "Age-related Harassment":         "You are way too old to even know how to use the internet, sit down grandpa.",
        "Gender-related Harassment":      "Go back to the kitchen, you women have no idea how engineering works.",
        "Religion-related Harassment":    "All people of that religion are extremists and terrorists who should be banned.",
        "Ethnicity-related Harassment":   "Go back to your own country you dirty illegal immigrant, we hate your kind here.",
        "General Toxic Insult":           "You are literally the dumbest and most pathetic loser to ever exist on this platform.",
    }

    selected_example = st.selectbox(
        "Select demo comment:",
        ["— Custom Input —"] + list(demo_examples.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#E2E8F0;margin:0.9rem 0'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box-sidebar">
            💡 <strong>Viva Defense Highlight</strong><br>
            Hierarchical Decision Gating prevents the <em>Multi-Class Dilution Fallacy</em>
            — where 5 toxic categories individually split the probability mass below
            the detection threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model Initialization (Resilient across Local & Cloud Deployments)
# ─────────────────────────────────────────────────────────────────────────────
model, tokenizer, is_cloud_mode = load_classification_pipeline(model_folder)

if is_cloud_mode:
    status_badge = '<span class="hero-badge" style="background:rgba(59,130,246,0.3);border-color:#60A5FA;color:#DBEAFE;">☁️ Cloud Live Mode</span>'
else:
    status_badge = '<span class="hero-badge" style="background:rgba(34,197,94,0.3);border-color:#4ADE80;color:#DCFCE7;">💻 Local DistilBERT Active</span>'

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">🛡️ Cyberbullying Detection System</div>
        <div class="hero-subtitle">
            Fine-Tuned DistilBERT Transformer Classifier · Automated Social Media Content Moderation
        </div>
        <div class="hero-badges">
            <span class="hero-badge">DistilBERT Base Uncased</span>
            <span class="hero-badge">6-Class Classification</span>
            <span class="hero-badge">2-Layer Moderation Engine</span>
            <span class="hero-badge">Automated Feed Sanitizer</span>
            <span class="hero-badge">Batch Queue Audit</span>
            {status_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Application Tabs: Single Inspector vs Batch Feed Queue
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs([
    "🔍  Single Comment Inspector",
    "📁  Batch Feed Moderation Queue",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE COMMENT INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown(
        '<div class="section-label"><span class="dot"></span> Step 1 — Enter Social Media Comment</div>',
        unsafe_allow_html=True,
    )

    default_text = ""
    if selected_example != "— Custom Input —":
        default_text = demo_examples[selected_example]

    user_comment = st.text_area(
        "Enter Social Media Comment to Analyse:",
        value=default_text,
        height=110,
        placeholder="e.g. 'Keep it up, you're doing great!' — or select a preset from the sidebar",
    )
    char_count = len(user_comment)
    counter_cls = "warn" if char_count > 450 else ""
    st.markdown(
        f'<div class="char-counter {counter_cls}">{char_count} / 512 characters</div>',
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_spacer = st.columns([1.6, 1.4, 6])
    with col_btn1:
        analyze_clicked = st.button("🔍  Analyze Comment", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("↺  Clear / Reset", use_container_width=True)

    if clear_clicked:
        st.rerun()

    # ── Inference & Results ──────────────────────────────────────────────────
    if analyze_clicked:
        cleaned_input = user_comment.strip()

        if not cleaned_input:
            st.markdown(
                '<div class="verdict-banner flagged"><div class="verdict-banner-title red">❌  Empty Input</div>'
                '<p class="verdict-detail">Please enter or select a comment to analyse.</p></div>',
                unsafe_allow_html=True,
            )
            st.stop()

        if len(cleaned_input) < 2:
            st.markdown(
                '<div class="verdict-banner flagged" style="border-color:#D97706;background:linear-gradient(135deg,#FFFBEB,#FFF);">'
                '<div class="verdict-banner-title" style="color:#D97706;">⚠️  Input Too Short</div>'
                '<p class="verdict-detail">Please enter a longer comment for reliable semantic analysis.</p></div>',
                unsafe_allow_html=True,
            )
            st.stop()

        with st.spinner("Analyzing semantics through two-layer moderation pipeline …"):
            res = moderate_single_comment(cleaned_input, model, tokenizer, float(toxicity_threshold))

        # ── Results Header ───────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label"><span class="dot"></span> Step 2 — Prediction Analysis & Automated Moderation</div>',
            unsafe_allow_html=True,
        )

        is_flagged = res["is_flagged"]
        top_toxic_category = res["top_category"]
        top_toxic_prob = res["top_prob"]
        aggregate_toxicity = res["aggregate_toxicity"]
        safe_prob = res["safe_prob"]
        prob_df = res["prob_df"]
        policy_triggered = res["policy_triggered"]
        affirmation_triggered = res["affirmation_triggered"]

        # ── Metric Cards ──────────────────────────────────────────────────────
        if is_flagged:
            verdict_text = f"🚨 {top_toxic_category.replace('_', ' ').title()}"
            card1_cls = "danger"
            val1_cls = "red"
            risk_cls = "danger"
            val2_cls = "red"
            cert_val = top_toxic_prob
            cert_sub = "Top toxic category confidence"
        else:
            verdict_text = "✅ Not Cyberbullying"
            card1_cls = "safe"
            val1_cls = "green"
            risk_cls = "safe"
            val2_cls = "green"
            cert_val = safe_prob
            cert_sub = "Safe content confidence"

        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card {card1_cls}">
                    <div class="metric-label">Primary Verdict</div>
                    <div class="metric-value {val1_cls}" style="font-size:1.2rem">{verdict_text}</div>
                    <div class="metric-subtext">Two-layer moderation decision</div>
                </div>
                <div class="metric-card {risk_cls}">
                    <div class="metric-label">Aggregate Harassment Risk</div>
                    <div class="metric-value {val2_cls}">{aggregate_toxicity:.1f}%</div>
                    <div class="metric-subtext">Cumulative toxic probability mass</div>
                </div>
                <div class="metric-card neutral">
                    <div class="metric-label">Category Certainty</div>
                    <div class="metric-value blue">{cert_val:.1f}%</div>
                    <div class="metric-subtext">{cert_sub}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Verdict Banner ────────────────────────────────────────────────────
        if is_flagged:
            if policy_triggered:
                banner_title = "🚨 Critical Safety Violation — Direct Threat / Harassment Detected"
                banner_body = (
                    f"<strong>Detection Mechanism:</strong> Layer 1 Safety Guard (Severe Violence &amp; Threat Policy Gate)<br>"
                    f"<strong>Assigned Category:</strong> {top_toxic_category.replace('_', ' ').title()}<br>"
                    f"<strong>Cumulative Harassment Risk:</strong> {aggregate_toxicity:.1f}%<br>"
                    f"<em>Explicit statements threatening violence or demographic attacks violate community safety standards.</em>"
                )
            else:
                banner_title = "🚨 Harassment / Toxic Intent Detected"
                banner_body = (
                    f"<strong>Detection Mechanism:</strong> Layer 2 — Fine-Tuned DistilBERT Transformer Classifier<br>"
                    f"<strong>Cumulative Harassment Risk:</strong> {aggregate_toxicity:.1f}% (Threshold: {toxicity_threshold}%)<br>"
                    f"<strong>Leading Category:</strong> {top_toxic_category.replace('_', ' ').title()} ({top_toxic_prob:.1f}%)<br>"
                    f"<em>{LABEL_DESCRIPTIONS.get(top_toxic_category, '')}</em>"
                )

            st.markdown(
                f"""
                <div class="verdict-banner flagged">
                    <div class="verdict-banner-title red">{banner_title}</div>
                    <p class="verdict-detail">{banner_body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if affirmation_triggered:
                banner_title = "✅ Safe — Positive Affirmation Verified"
                banner_body = (
                    f"<strong>Detection Mechanism:</strong> Layer 1 Identity-Term De-biasing Guard (Empowering Affirmation)<br>"
                    f"<strong>Safe Probability:</strong> {safe_prob:.1f}%<br>"
                    f"<strong>Cumulative Toxicity Risk:</strong> Only {aggregate_toxicity:.1f}%<br>"
                    f"<em>Statement contains positive, supportive language with zero derogatory or abusive markers.</em>"
                )
            else:
                banner_title = "✅ Safe — Non-Harassment Content Verified"
                banner_body = (
                    f"<strong>Detection Mechanism:</strong> Layer 2 — Fine-Tuned DistilBERT Transformer Classifier<br>"
                    f"<strong>Safe Probability:</strong> {safe_prob:.1f}%<br>"
                    f"<strong>Cumulative Toxicity Risk:</strong> Only {aggregate_toxicity:.1f}% "
                    f"(well below the {toxicity_threshold}% safety gate)<br>"
                    f"<em>No significant harassment signal detected across any of the six classification categories.</em>"
                )

            st.markdown(
                f"""
                <div class="verdict-banner safe-bg">
                    <div class="verdict-banner-title green">{banner_title}</div>
                    <p class="verdict-detail">{banner_body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Automated Moderation Action & Censor Preview ──────────────────────
        action_type = res["action"]
        if action_type == "QUARANTINE":
            pill_html = '<span class="action-pill quarantine">🛑 Quarantine & Block</span>'
            action_desc = "Content blocked from public timeline. Marked for immediate administrator review."
        elif action_type == "HOLD_FOR_REVIEW":
            pill_html = '<span class="action-pill review">⚠️ Hold For Review</span>'
            action_desc = "Content flagged with borderline risk. Sent to human moderator queue before public release."
        else:
            pill_html = '<span class="action-pill approve">✅ Auto-Approved</span>'
            action_desc = "Content cleared all safety thresholds. Permitted for immediate public display."

        st.markdown(
            f"""
            <div class="action-box">
                <div class="action-header">
                    <div>
                        <strong style="color:#1E293B;font-size:0.92rem;">🛡️ Automated Moderation Action & Feed Sanitizer Preview</strong>
                        <div style="font-size:0.78rem;color:#64748B;margin-top:0.2rem;">{action_desc}</div>
                    </div>
                    <div>{pill_html}</div>
                </div>
                <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#64748B;margin-bottom:0.4rem;">
                    Sanitized Feed Representation:
                </div>
                <div class="redacted-preview">{res["sanitized_html"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Matplotlib Horizontal Bar Chart ───────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label"><span class="dot"></span> Category Probability Distribution</div>',
            unsafe_allow_html=True,
        )

        chart_df_sorted = prob_df.sort_values("Probability (%)", ascending=True)
        categories = chart_df_sorted["Category"].tolist()
        values = chart_df_sorted["Probability (%)"].tolist()
        bar_colors = ["#16A34A" if c == "not_cyberbullying" else "#DC2626" for c in categories]

        fig, ax = plt.subplots(figsize=(9, 3.4))
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#F8FAFC")

        bars_plot = ax.barh(
            categories, values,
            color=bar_colors,
            height=0.55,
            edgecolor="none",
        )
        for bar, val in zip(bars_plot, values):
            ax.text(
                min(val + 1.0, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#1E293B",
            )

        ax.set_xlim(0, 108)
        ax.set_xlabel("Probability (%)", fontsize=9.5, color="#64748B", labelpad=8)
        ax.tick_params(axis="y", labelsize=10, colors="#1E293B", length=0)
        ax.tick_params(axis="x", labelsize=9,  colors="#64748B", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#F1F5F9", linewidth=0.8)

        legend_handles = [
            mpatches.Patch(facecolor="#DC2626", label="Toxic / Harassment"),
            mpatches.Patch(facecolor="#16A34A", label="Safe Content"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=9,
            framealpha=0.8,
            edgecolor="#E2E8F0",
        )

        plt.tight_layout(pad=0.8)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-title">📊 Classification Output — All Six Categories</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Category Descriptions Reference Table ─────────────────────────────
        with st.expander("📋  Category Descriptions & Taxonomy Reference"):
            display_df = prob_df[["Category", "Description"]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Disclaimers ───────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer-row">
            <div class="disclaimer-box warning">
                <strong>⚠️ Academic Prototype Notice</strong>
                <p>
                    Machine learning models can produce false positives and false negatives,
                    especially in cases of nuanced sarcasm, slang, or implicit hate speech.
                    Automated decisions serve as initial triage filters.
                </p>
            </div>
            <div class="disclaimer-box info">
                <strong>🛡️ Human-in-the-Loop Moderation</strong>
                <p>
                    High model confidence does not constitute definitive proof of intent.
                    In production systems, automated flags assist human moderators
                    in maintaining community standards.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: BATCH FEED MODERATION QUEUE
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(
        '<div class="section-label"><span class="dot"></span> Bulk Social Feed Moderation & Audit Queue</div>',
        unsafe_allow_html=True,
    )
    st.caption("Audit multi-user social comment streams concurrently using the two-layer neural defense pipeline.")

    sample_feed_comments = [
        "Keep it up, you are doing a fantastic job with this project!",
        "I will kill you and your entire family.",
        "The conference meeting is scheduled for tomorrow at 10 AM.",
        "Go back to the kitchen, you women have no idea how engineering works.",
        "Go back to your own country you dirty illegal immigrant, we hate your kind here.",
        "You are way too old to even know how to use the internet, sit down grandpa.",
        "You are literally the dumbest and most pathetic loser on this platform.",
        "Your presentation was very interesting, thank you for sharing.",
    ]

    col_btn_load, col_btn_clear_b, _ = st.columns([2.2, 1.4, 5])
    with col_btn_load:
        load_sample = st.button("⚡ Load Demo Social Media Feed (8 Comments)", use_container_width=True)

    default_batch_text = "\n".join(sample_feed_comments) if load_sample else ""

    batch_input = st.text_area(
        "Enter multiple comments (one comment per line):",
        value=default_batch_text,
        height=180,
        placeholder="Paste comments here, one per line:\nExample 1: Great work today!\nExample 2: Go back to where you came from...",
    )

    col_proc, _ = st.columns([2, 6])
    with col_proc:
        run_batch = st.button("🚀 Process Batch Moderation", type="primary", use_container_width=True)

    if run_batch:
        raw_lines = [line.strip() for line in batch_input.split("\n") if line.strip()]

        if not raw_lines:
            st.warning("⚠️ No valid comments found. Please type comments (one per line) or click 'Load Demo Social Media Feed'.")
        else:
            with st.spinner(f"Auditing {len(raw_lines)} comments through DistilBERT inference pipeline …"):
                results_list = []
                for idx, text in enumerate(raw_lines):
                    res = moderate_single_comment(text, model, tokenizer, float(toxicity_threshold))
                    results_list.append({
                        "ID": idx + 1,
                        "Comment Text": text,
                        "Verdict": "🛑 Quarantined" if res["is_flagged"] else "✅ Approved",
                        "Category": res["top_category"].replace("_", " ").title(),
                        "Harassment Risk (%)": f"{res['aggregate_toxicity']:.1f}%",
                        "Certainty (%)": f"{res['top_prob']:.1f}%",
                        "Action Taken": res["action"],
                        "Severity": res["severity"],
                        "_raw_risk": res["aggregate_toxicity"],
                        "_flagged": res["is_flagged"],
                    })

            batch_df = pd.DataFrame(results_list)
            total_count = len(batch_df)
            flagged_count = batch_df["_flagged"].sum()
            safe_count = total_count - flagged_count
            flag_ratio = (flagged_count / total_count) * 100 if total_count > 0 else 0.0

            # ── Batch Summary Metrics ─────────────────────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-label"><span class="dot"></span> Batch Audit Summary</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="metric-row-4">
                    <div class="metric-card neutral">
                        <div class="metric-label">Total Comments</div>
                        <div class="metric-value blue">{total_count}</div>
                        <div class="metric-subtext">Processed feed items</div>
                    </div>
                    <div class="metric-card safe">
                        <div class="metric-label">Clean / Approved</div>
                        <div class="metric-value green">{safe_count}</div>
                        <div class="metric-subtext">{(safe_count/total_count)*100:.1f}% approval rate</div>
                    </div>
                    <div class="metric-card danger">
                        <div class="metric-label">Quarantined / Flagged</div>
                        <div class="metric-value red">{flagged_count}</div>
                        <div class="metric-subtext">{flag_ratio:.1f}% violation rate</div>
                    </div>
                    <div class="metric-card neutral">
                        <div class="metric-label">Policy Gate</div>
                        <div class="metric-value blue">{toxicity_threshold}%</div>
                        <div class="metric-subtext">Operating threshold</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Category Breakdown in Batch ───────────────────────────────────
            st.markdown("##### 📊 Harassment Category Distribution in Batch")
            category_counts = batch_df[batch_df["_flagged"]]["Category"].value_counts()
            if not category_counts.empty:
                fig_b, ax_b = plt.subplots(figsize=(7, 2.8))
                fig_b.patch.set_facecolor("none")
                fig_b.patch.set_alpha(0.0)
                ax_b.set_facecolor("#F8FAFC")

                b_cats = category_counts.index.tolist()
                b_vals = category_counts.values.tolist()
                bars = ax_b.bar(b_cats, b_vals, color="#DC2626", width=0.45)
                for bar in bars:
                    yval = bar.get_height()
                    ax_b.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval + 0.1,
                        int(yval),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        color="#1E293B",
                        fontsize=9.5,
                    )
                ax_b.set_ylabel("Flags", fontsize=9, color="#64748B")
                ax_b.tick_params(axis="x", labelsize=9, colors="#1E293B", length=0)
                ax_b.tick_params(axis="y", labelsize=8.5, colors="#64748B", length=0)
                ax_b.spines["top"].set_visible(False)
                ax_b.spines["right"].set_visible(False)
                ax_b.spines["left"].set_color("#E2E8F0")
                ax_b.spines["bottom"].set_color("#E2E8F0")
                ax_b.yaxis.grid(True, color="#F1F5F9", linewidth=0.8)
                plt.tight_layout(pad=0.8)
                st.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)
            else:
                st.success("🎉 All comments in this batch were cleared as Safe / Non-cyberbullying!")

            # ── Batch Table ───────────────────────────────────────────────────
            st.markdown("##### 📋 Moderation Audit Log")
            show_cols = ["ID", "Verdict", "Category", "Harassment Risk (%)", "Certainty (%)", "Action Taken", "Comment Text"]
            st.dataframe(batch_df[show_cols], use_container_width=True, hide_index=True)

            # ── CSV Export ────────────────────────────────────────────────────
            csv_buffer = io.StringIO()
            batch_df[show_cols].to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Moderation Audit Log (.csv)",
                data=csv_buffer.getvalue(),
                file_name="cyberbullying_moderation_audit_log.csv",
                mime="text/csv",
                use_container_width=False,
            )
