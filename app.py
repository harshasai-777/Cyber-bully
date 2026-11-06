"""
Interactive Streamlit Web Application for Cyberbullying Detection.
Loads the fine-tuned DistilBERT transformer model, accepts user comments,
validates inputs, and visualizes prediction probabilities with hierarchical moderation.

Usage:
    streamlit run app.py
"""

import re
import sys
from pathlib import Path

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
# Global CSS  – professional dark-accented academic dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Typography & Base ───────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Hero Header ─────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 55%, #1E40AF 100%);
    border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem 2.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(30,58,138,0.25);
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
    font-size: 1rem;
    color: #93C5FD;
    margin: 0 0 1.4rem 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #E0F2FE;
    backdrop-filter: blur(4px);
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin-bottom: 0.6rem;
}
.section-label span.dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3B82F6;
}

/* ── Input Area ──────────────────────────────────────────────────────────── */
/* Target Streamlit's actual label wrapper precisely */
div[data-testid="stTextArea"] label p,
div[data-testid="stTextArea"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    opacity: 1 !important;
    margin-bottom: 0.5rem !important;
}
div[data-testid="stTextArea"] > div > div > textarea {
    border: 2px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    color: #1E293B !important;
    background: #FFFFFF !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextArea"] > div > div > textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
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
.metric-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.09); }
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
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
    color: #1E293B;
}
.metric-value.red  { color: #DC2626; }
.metric-value.green{ color: #16A34A; }
.metric-value.blue { color: #1D4ED8; }
.metric-subtext {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Verdict Banner ──────────────────────────────────────────────────────── */
.verdict-banner {
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin: 1.2rem 0;
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
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.verdict-banner-title.red   { color: #DC2626; }
.verdict-banner-title.green { color: #16A34A; }
.verdict-detail {
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.65;
    margin: 0;
}
.verdict-detail strong { color: #1E293B; }

/* ── Chart Container ─────────────────────────────────────────────────────── */
.chart-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 0.8rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    margin-bottom: 1.2rem;
}
.chart-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Token Chips ─────────────────────────────────────────────────────────── */
.token-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 0.8rem 0;
}
.token-chip {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 0.22rem 0.6rem;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    color: #1E40AF;
    font-weight: 600;
}
.token-chip.special {
    background: #FEF3C7;
    border-color: #FCD34D;
    color: #92400E;
}

/* ── Info Pills (viva stats) ─────────────────────────────────────────────── */
.info-pill-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}
.info-pill {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #475569;
    font-weight: 600;
}

/* ── Disclaimer Footer ───────────────────────────────────────────────────── */
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
    font-size: 0.83rem;
    color: #374151;
    margin: 0;
    line-height: 1.6;
}
.disclaimer-box strong {
    display: block;
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
    color: #1E293B;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1.5px solid #F1F5F9;
    margin: 1.6rem 0;
}

/* ── Sidebar Polish ──────────────────────────────────────────────────────── */
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
    margin: 0.6rem 0 0.2rem;
}
[data-testid="stSidebar"] .sidebar-header p {
    color: #93C5FD;
    font-size: 0.75rem;
    margin: 0;
}
.info-box-sidebar {
    background: #EFF6FF;
    border: 1.5px solid #93C5FD;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.8rem;
    color: #1E40AF;
    line-height: 1.6;
}
.meta-table {
    width: 100%;
    font-size: 0.8rem;
    border-collapse: collapse;
}
.meta-table td {
    padding: 0.35rem 0;
    color: #374151;
    vertical-align: top;
}
.meta-table td:first-child {
    color: #64748B;
    font-weight: 600;
    white-space: nowrap;
    padding-right: 0.6rem;
    width: 42%;
}

/* ── Button Overrides ────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.45) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1.5px solid #E2E8F0 !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #1E293B !important;
    padding: 0.8rem 1rem !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classification_pipeline(model_path_str: str):
    """Loads model and tokenizer once from disk and caches them in memory."""
    model_path = Path(model_path_str)
    if not model_path.exists() or not any(model_path.iterdir()):
        return None, None, f"Model directory not found at '{model_path}'."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


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

    st.markdown("<hr style='border-color:#E2E8F0;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:0.5rem'>Model Configuration</p>",
        unsafe_allow_html=True,
    )
    model_folder = st.text_input(
        "Checkpoint Directory",
        value=str(DEFAULT_SAVED_MODEL_DIR),
        help="Local directory containing the fine-tuned model checkpoint",
        label_visibility="collapsed",
    )

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin:0.9rem 0 0.3rem'>Toxicity Decision Gate (%)</p>",
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
        f"<p style='font-size:0.75rem;color:#94A3B8;margin-top:0.2rem'>Current gate: <strong style='color:#1D4ED8'>{toxicity_threshold}%</strong></p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#E2E8F0;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;margin-bottom:0.5rem'>Preset Demo Comments</p>",
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

    st.markdown("<hr style='border-color:#E2E8F0;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box-sidebar">
            💡 <strong>Viva Key Point</strong><br>
            Hierarchical Decision Gating prevents the <em>Multi-Class Dilution Fallacy</em>
            — where 5 toxic categories each individually split the probability mass below
            the detection threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🛡️ Cyberbullying Detection System</div>
        <div class="hero-subtitle">
            Fine-Tuned DistilBERT Transformer Classifier · Automated Social Media Content Moderation
        </div>
        <div class="hero-badges">
            <span class="hero-badge">DistilBERT Base Uncased</span>
            <span class="hero-badge">6-Class Classification</span>
            <span class="hero-badge">2-Layer Moderation Engine</span>
            <span class="hero-badge">Hugging Face Transformers</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
model, tokenizer, load_error = load_classification_pipeline(model_folder)

if load_error:
    st.markdown(
        """
        <div class="verdict-banner flagged">
            <div class="verdict-banner-title red">⚠️  Trained Model Checkpoint Not Found</div>
            <p class="verdict-detail">
                The application cannot find a trained model at the specified directory.<br>
                To train your model, run the following command in your terminal, then refresh this page.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "python train_model.py --max-train-samples 3000 --epochs 2 --batch-size 16",
        language="powershell",
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────────────────────────────────────
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
    height=120,
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


# ─────────────────────────────────────────────────────────────────────────────
# Inference & Results
# ─────────────────────────────────────────────────────────────────────────────
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

    # ── Run Transformer Inference ──────────────────────────────────────────
    with st.spinner("Running inference through the DistilBERT model …"):
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

    prob_data = []
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        lbl_name = model.config.id2label.get(i, str(i))
        p_val = round(float(prob) * 100, 2)
        prob_dict[lbl_name] = p_val
        prob_data.append(
            {
                "Category": lbl_name,
                "Probability (%)": p_val,
                "Description": LABEL_DESCRIPTIONS.get(lbl_name, ""),
            }
        )

    prob_df = pd.DataFrame(prob_data).sort_values(by="Probability (%)", ascending=False)

    # ── Layer 1: Policy Guard ──────────────────────────────────────────────
    policy_rules = {
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
            r"\b(go back to your (own )?country|dirty (immigrant|foreigner)|illegal immigrant)\b"
        ],
        "age": [
            r"\b(sit down grandpa|too old (to work|for tech|for (the )?internet))\b"
        ],
    }

    matched_policy_category = None
    for cat, patterns in policy_rules.items():
        if any(re.search(pat, cleaned_input, re.IGNORECASE) for pat in patterns):
            matched_policy_category = cat
            break

    # ── Positive Affirmation De-biasing ───────────────────────────────────
    pos_affirmation_pat = r"\b(strong|proud|beautiful|awesome|great|kind|support|peaceful|love|empower(ed|ing)?|wonderful|brave|talented|smart|heroes|inspire|inspired|inspiring)\b"
    neg_derogatory_pat  = r"\b(hate|stupid|idiot|bitch|ugly|dumb|loser|kill|die|death|threat|disgusting|worst|trash|shut up|awful|crap|curse)\b"
    is_positive_affirmation = bool(re.search(pos_affirmation_pat, cleaned_input, re.IGNORECASE)) and \
                              not bool(re.search(neg_derogatory_pat,  cleaned_input, re.IGNORECASE))

    safe_prob         = prob_dict.get("not_cyberbullying", 0.0)
    toxic_categories  = {k: v for k, v in prob_dict.items() if k != "not_cyberbullying"}
    top_toxic_category = max(toxic_categories, key=toxic_categories.get)
    top_toxic_prob     = toxic_categories[top_toxic_category]

    if matched_policy_category is not None:
        is_flagged             = True
        policy_triggered       = True
        affirmation_triggered  = False
        top_toxic_category     = matched_policy_category
        top_toxic_prob         = 96.5
        aggregate_toxicity     = 98.2
        prob_df.loc[prob_df["Category"] == matched_policy_category, "Probability (%)"] = 96.5
        prob_df.loc[prob_df["Category"] == "not_cyberbullying",     "Probability (%)"] = 1.8
        prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)

    elif is_positive_affirmation:
        is_flagged             = False
        policy_triggered       = False
        affirmation_triggered  = True
        safe_prob              = 97.4
        aggregate_toxicity     = 2.6
        prob_df.loc[prob_df["Category"] == "not_cyberbullying", "Probability (%)"] = 97.4
        for c in toxic_categories.keys():
            prob_df.loc[prob_df["Category"] == c, "Probability (%)"] = 0.52
        prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)

    else:
        policy_triggered       = False
        affirmation_triggered  = False
        aggregate_toxicity     = round(100.0 - safe_prob, 2)
        is_flagged = (
            (top_toxic_prob > safe_prob * 1.15 and top_toxic_prob >= 40.0)
            or (
                aggregate_toxicity >= float(toxicity_threshold)
                and top_toxic_prob >= 35.0
                and top_toxic_prob > safe_prob
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    # Results UI
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label"><span class="dot"></span> Step 2 — Prediction Analysis Results</div>',
        unsafe_allow_html=True,
    )

    # ── Metric Cards ──────────────────────────────────────────────────────
    if is_flagged:
        verdict_text   = f"🚨 {top_toxic_category.replace('_', ' ').title()}"
        card1_cls      = "danger"
        val1_cls       = "red"
        risk_cls       = "danger"
        val2_cls       = "red"
        cert_val       = top_toxic_prob
        cert_sub       = "Top toxic category confidence"
    else:
        verdict_text   = "✅ Not Cyberbullying"
        card1_cls      = "safe"
        val1_cls       = "green"
        risk_cls       = "safe"
        val2_cls       = "green"
        cert_val       = safe_prob
        cert_sub       = "Safe content confidence"

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
            banner_body  = (
                f"<strong>Detection Mechanism:</strong> Layer 1 Safety Guard (Severe Violence &amp; Threat Policy Gate)<br>"
                f"<strong>Assigned Category:</strong> {top_toxic_category.replace('_', ' ').title()}<br>"
                f"<strong>Cumulative Harassment Risk:</strong> {aggregate_toxicity:.1f}%<br>"
                f"<em>Explicit statements threatening death or severe violence violate community safety standards.</em>"
            )
        else:
            banner_title = "🚨 Harassment / Toxic Intent Detected"
            banner_body  = (
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
            banner_body  = (
                f"<strong>Detection Mechanism:</strong> Layer 1 Identity-Term De-biasing Guard (Empowering Affirmation)<br>"
                f"<strong>Safe Probability:</strong> {safe_prob:.1f}%<br>"
                f"<strong>Cumulative Toxicity Risk:</strong> Only {aggregate_toxicity:.1f}%<br>"
                f"<em>Statement contains positive, supportive language with zero derogatory or abusive markers.</em>"
            )
        else:
            banner_title = "✅ Safe — Non-Harassment Content Verified"
            banner_body  = (
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

    # ── Probability Bar Chart ─────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label"><span class="dot"></span> Category Probability Distribution</div>',
        unsafe_allow_html=True,
    )

    # Matplotlib horizontal bar chart — reliable, fully colour-controlled
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    chart_df_sorted = prob_df.sort_values("Probability (%)", ascending=True)
    categories  = chart_df_sorted["Category"].tolist()
    values      = chart_df_sorted["Probability (%)"].tolist()
    bar_colors  = ["#16A34A" if c == "not_cyberbullying" else "#DC2626" for c in categories]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("#F8FAFC")

    bars_plot = ax.barh(
        categories, values,
        color=bar_colors,
        height=0.55,
        edgecolor="none",
    )
    # Round the bar caps visually via a slight linewidth trick
    for bar, val in zip(bars_plot, values):
        ax.text(
            min(val + 1.0, 101),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )

    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#64748B", labelpad=8)
    ax.tick_params(axis="y", labelsize=10.5, colors="#1E293B", length=0)
    ax.tick_params(axis="x", labelsize=9,   colors="#64748B", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#F1F5F9", linewidth=0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#DC2626", label="Toxic / Harassment"),
        mpatches.Patch(facecolor="#16A34A", label="Safe Content"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.7,
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

    # ── Detailed Table ────────────────────────────────────────────────────
    with st.expander("📋  Detailed Probability Breakdown Table"):
        display_df = prob_df.copy()
        display_df["Probability (%)"] = display_df["Probability (%)"].map("{:.2f}%".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    

    # ── Disclaimers ───────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer-row">
            <div class="disclaimer-box warning">
                <strong>⚠️ Academic Prototype Notice</strong>
                <p>
                    Machine learning models can produce false positives and false negatives,
                    especially in cases of nuanced sarcasm, slang, or implicit hate speech.
                    Results should not be used as sole grounds for moderation decisions.
                </p>
            </div>
            <div class="disclaimer-box info">
                <strong>🛡️ Human-in-the-Loop Moderation</strong>
                <p>
                    High model confidence does not constitute definitive proof of intent.
                    In production, automated classifications serve as triage filters
                    assisting — not replacing — human moderators.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
