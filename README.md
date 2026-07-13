# CommonLit — Evaluate Student Summaries

An NLP regression system that scores the quality of student-written summaries, built for the Kaggle [**CommonLit - Evaluate Student Summaries**](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) competition.

---

## Overview

Teachers need scalable ways to assess how well students summarise what they read. This project trains a transformer model to automatically evaluate a student's summary of a source text along two dimensions — **how well it captures the content** and **how well it's written** — producing scores that track expert human ratings.

Given a summary (plus the source text it was written about, the prompt question the student was answering, and the passage title), the model predicts two continuous scores:

| Target | What it measures |
| --- | --- |
| **content** | How accurately and completely the summary captures the main ideas, details, and structure of the source text |
| **wording** | Clarity, objectivity, precision of language, and fluency of the writing — independent of content coverage |

Both are real-valued (roughly standardised around 0, spanning negative to positive values), so this is a **two-target regression** problem, not classification.

---

## Evaluation metric

Submissions are ranked by **MCRMSE** — Mean Columnwise Root Mean Squared Error:

$$\text{MCRMSE} = \frac{1}{2}\sum_{j=1}^{2} \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{ij} - \hat{y}_{ij})^2}$$

The RMSE is computed separately for `content` and `wording`, then averaged. Because the two columns are weighted equally, a model that nails content but writes off wording is penalised just as hard as the reverse — both heads have to be good.

---

## The core modelling challenge: generalising to unseen prompts

The single most important fact about this dataset is that it contains **only a handful of distinct prompts** (each prompt = one source passage + one question that many students summarised). The hidden test set is built from **prompts the model has never seen during training**.

This has one decisive consequence: **a random train/validation split is misleading.** If summaries from the same prompt land in both train and validation, the model memorises prompt-specific signals (topic vocabulary, the source passage's phrasing) and reports a validation score far better than it will achieve on the leaderboard.

The fix — and the backbone of this solution's validation — is **grouped cross-validation by prompt**: every fold holds out entire prompts, so validation always measures performance on prompts the model was not trained on. This mirrors the real test condition and makes local scores trustworthy.

---

## Methodology

### 1. Input construction

Each training example is assembled into a single text sequence that gives the model the full context it needs to judge a summary *relative to what was being summarised*. The pieces combined are:

- the **prompt question** (what the student was asked to do)
- the **source text** (what they were summarising)
- the **student summary** (what we're scoring)

<!-- verify: confirm the exact field order and any separator tokens you used, e.g. [SEP] between segments -->

Feeding the source alongside the summary is what lets the model assess *content* — it can compare what the summary says against what the source actually contains, rather than judging the summary in isolation.

### 2. Model architecture

A pretrained transformer encoder is fine-tuned as a regressor:

- **Backbone:** a DeBERTa-family encoder <!-- verify: replace with the exact checkpoint you used, e.g. microsoft/deberta-v3-base -->
- **Pooling:** the token representations are pooled into a single fixed-length vector <!-- verify: mean pooling / CLS token / attention pooling — state which -->
- **Head:** a lightweight linear regression head maps the pooled vector to the **two outputs** (`content`, `wording`) predicted jointly

Predicting both targets from one shared encoder lets the model exploit signal common to both (a garbled summary tends to score low on both axes) while the two output units specialise.

### 3. Handling long inputs

Source texts can be long — often longer than the encoder's maximum token limit. <!-- verify: describe what you actually did — truncation to N tokens, sliding window, or summary-focused truncation that always keeps the student summary intact -->

### 4. Training setup

- **Loss:** the model optimises regression error on both targets <!-- verify: MSE / SmoothL1 / RMSE — state which -->
- **Validation:** grouped K-fold by `prompt_id` (see above) <!-- verify: number of folds -->
- **Optimiser & schedule:** AdamW with a warmup + linear decay schedule <!-- verify: learning rate, epochs, batch size, warmup ratio -->
- **Regularisation:** <!-- verify: dropout, weight decay, layer-wise learning rate decay, early stopping — list what you used -->

### 5. Inference

At test time, each summary is assembled into the same input format, passed through the fine-tuned model, and the two predicted scores are written to `submission.csv` in the competition format (`student_id, content, wording`).

---

## Repository structure

| File | Description |
| --- | --- |
| `commonlit-evaluate-student-summaries.ipynb` | End-to-end notebook: data loading → input construction → tokenisation → fine-tuning with grouped CV → inference → submission |
| `README.md` | This file |

---

## Running it

The notebook is built to run as a **Kaggle notebook with a GPU** enabled:

1. Attach the competition dataset (`commonlit-evaluate-student-summaries`).
2. Enable GPU in the notebook settings. <!-- verify: note if internet must be off for submission -->
3. Run all cells — training runs, then inference writes `submission.csv`.

Core dependencies (`torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`) are preinstalled in the Kaggle image.

---

## Key takeaways

- **Prompt-grouped validation is the whole game.** The gap between a naive random split and grouped CV is the difference between a score that looks great locally and one that survives the leaderboard.
- **Context matters for content scoring.** Giving the model the source text — not just the summary — is what makes the `content` target learnable.
- **Joint two-target regression** shares representation across the two correlated scores while letting each specialise.

---

## Ideas for improvement

- **Hand-crafted features** appended to the transformer embedding: summary/source length ratio, n-gram overlap with the source, readability indices, spelling/grammar error counts — cheap signals that correlate with `wording`.
- **Ensembling** across multiple encoder checkpoints, seeds, or max-length settings to reduce MCRMSE variance.
- **Larger backbones** (deberta-v3-large) or **multi-sample dropout** for a typically reliable bump on this kind of text-regression task.
- **Pseudo-labelling** additional summary data to expand the small effective training set.
