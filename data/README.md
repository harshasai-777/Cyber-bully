# Dataset Documentation: Surrey NLP Cyberbullying Detection (CB1)

## 1. Overview
This project uses the publicly available academic dataset:
- **Repository:** [`surrey-nlp/Cyberbullying-Detection-CB1`](https://huggingface.co/datasets/surrey-nlp/Cyberbullying-Detection-CB1)
- **Hosted On:** Hugging Face Hub
- **Domain:** Social media microblogging comments (Twitter / X corpus)
- **Task:** Multi-class text classification for online harassment identification

---

## 2. Dataset Schema & Structure

Each record contains the following features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `tweet_text` | `string` | The raw text content of the social media comment |
| `cyberbullying_type` | `string` | The target ground-truth categorical label (6 distinct classes) |

---

## 3. Split Breakdown

The dataset provides clean, partitioned splits ensuring no overlap or data leakage between training and testing:

| Split Name | Row Count | Purpose in Pipeline |
| :--- | :--- | :--- |
| `train` | 35,769 | Model fine-tuning and weight optimization |
| `validation` | 2,000 | Hyperparameter tuning and model checkpoint selection (`load_best_model_at_end`) |
| `test` | 9,923 | Final independent unbiased evaluation (`evaluate_model.py`) |
| **Total** | **47,692** | Full corpus |

---

## 4. Class Distribution & Taxonomy

The dataset contains six balanced categories:

1. **`not_cyberbullying`**: Conversational, constructive, neutral, or positive statements with no hostile intent.
2. **`gender`**: Sexist, misogynistic, or gender-based harassment, slurs, or exclusionary remarks.
3. **`religion`**: Hostile derogatory comments, hate speech, or stereotypes targeting religious beliefs or communities.
4. **`other_cyberbullying`**: General toxicity, personal insults, name-calling, or trolling without demographic targeting.
5. **`age`**: Ageist slurs, harassment, or prejudice targeting elderly individuals or specific age brackets.
6. **`ethnicity`**: Racial slurs, xenophobic hate speech, or harassment targeting race, ethnic origins, or nationality.

---

## 5. Ethical & Licensing Notice
- **Academic Use:** This dataset is provided for academic, educational, and safety research purposes under standard Hugging Face and academic open data terms.
- **Privacy & Storage:** Do not redistribute raw user identities or sensitive personal data. Dataset cache is handled automatically by the Hugging Face `datasets` library under `~/.cache/huggingface/datasets/` and excluded from Git commits.
