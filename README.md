# Painting Response Classifier — CSC311 ML Challenge

A machine learning classifier that predicts viewer responses to paintings, built as part of the CSC311 (Introduction to Machine Learning) challenge project at the University of Toronto. The system uses a Logistic Regression model with a custom 322-feature representation and ships with a **scikit-learn-free inference pipeline** for lightweight, dependency-minimal deployment.

---

## Overview

Given a dataset of paintings paired with viewer metadata and response attributes, the model predicts how a viewer will respond to a given painting. The project emphasizes thoughtful feature engineering over model complexity — a well-tuned Logistic Regression achieves **~90.5% validation accuracy**, outperforming more complex approaches that overfit to the training distribution.

---

## Feature Engineering (322 Features)

The feature representation combines three encoding strategies into a single dense vector per sample:

| Feature Type | Description | Details |
|---|---|---|
| **Standardized Numerics** | Continuous and ordinal fields | Zero-mean, unit-variance scaling applied to numeric columns to ensure equal contribution across features with different magnitudes. |
| **Multi-Hot Encodings** | Categorical fields with multiple values | Categories such as viewer demographics or painting attributes are encoded as binary indicator vectors, supporting multi-label fields where a single sample can belong to multiple categories. |
| **Bag-of-Words (BoW)** | Free-text response fields | Text fields are tokenized and represented as token-count vectors. A consistent tokenizer is used across training and inference to prevent vocabulary mismatches. |

All three representations are concatenated horizontally into a single 322-dimensional feature vector per sample.

---

## Model

- **Algorithm:** Logistic Regression (one-vs-rest for multiclass)
- **Regularization:** L2 with `C=0.1` (inverse regularization strength), selected via cross-validation to balance bias and variance.
- **Validation Accuracy:** ~90.5%
- **Training:** Fit using scikit-learn during development, then model weights exported for standalone inference.

### Why Logistic Regression?

Simpler models with strong feature engineering tend to generalize better on tabular datasets of moderate size. Logistic Regression with L2 regularization provides a well-understood decision boundary, is less prone to overfitting than tree ensembles or neural networks on this data scale, and produces interpretable weight vectors that reveal which features drive predictions.

---

## Sklearn-Free Inference Pipeline

To meet the challenge's deployment constraints (minimal dependencies), the trained model parameters are exported to a `.npz` file and inference is performed using only the NumPy library features:

```
Training Phase (sklearn allowed)          Inference Phase (sklearn-free)
┌──────────────────────────┐              ┌──────────────────────────┐
│  Load & preprocess data  │              │  Load model_params.npz   │
│  Fit LogisticRegression  │  ──export──▶ │  (weights, bias, vocab,  │
│  Validate (~90.5% acc)   │              │   scaler params, etc.)   │
│  Export to .npz          │              │  Reconstruct features    │
└──────────────────────────┘              │  Manual dot product +    │
                                          │   softmax / argmax       │
                                          └──────────────────────────┘
```

### Exported Parameters (`model_params.npz`)
- **Model weights** — Coefficient matrix and intercept vector from the fitted Logistic Regression.
- **Scaler parameters** — Per-feature mean and standard deviation for reapplying standardization at inference time.
- **BoW vocabulary** — The exact token-to-index mapping used during training, ensuring tokenization consistency.
- **Encoding mappings** — Category-to-index dictionaries for reconstructing multi-hot vectors.

### Inference Logic (NumPy only)
```python
# Reconstruct the 322-dim feature vector
x = build_feature_vector(sample, scaler_params, vocab, encodings)

# Predict: logits = Wx + b
logits = x @ weights.T + bias
predicted_class = np.argmax(logits, axis=1)
```

## Project Structure

```
painting-classifier/
├── train.py             # Data loading, feature engineering, model training & export
├── predict.py           # Sklearn-free inference pipeline (NumPy only)
├── model_params.npz     # Exported model weights, scaler params, vocab, encodings
├── features.py          # Feature construction utilities (scaling, multi-hot, BoW)
├── data/                # Training and validation datasets
└── README.md
```

---

## Usage

### Training
```bash
python train.py --data data/train.csv --output model_params.npz
```

### Inference
```bash
python predict.py --model model_params.npz --input data/test.csv --output predictions.csv
```

---

## Technical Highlights

- **322-feature representation** combining three encoding strategies into a unified dense vector — no deep learning, no complex ensembles.
- **~90.5% validation accuracy** with a single Logistic Regression model (`C=0.1`, L2 regularization).
- **Fully portable inference** — the prediction pipeline depends only on NumPy, with all preprocessing parameters serialized into a single `.npz` artifact.
- Diagnosed and resolved subtle **pandas Copy-on-Write** bugs and **tokenizer consistency** issues that silently degraded model performance.

---

## Technologies

- **Language:** Python
- **Training:** scikit-learn, pandas, NumPy
- **Inference:** NumPy only (sklearn-free)
- **Concepts:** Logistic Regression, feature engineering, bag-of-words, multi-hot encoding, standardization, model serialization, deployment-constrained inference

---

## Acknowledgements

Developed as part of the **CSC311 — Introduction to Machine Learning** challenge project at the University of Toronto Mississauga.
