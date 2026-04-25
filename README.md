# 3 Paintings Response Classifier

A machine learning classifier that predicts viewer responses to the paintings of: The Persistence of Memory - by Salvador Dalí, The Starry Night - by Vincent Van Gogh, and one of the Water Lilies paintings - by Claude Monet, as part of the CSC311 (Introduction to Machine Learning) challenge project at the University of Toronto. The system uses a Logistic Regression model with a custom 322-feature representation and ships with a **scikit-learn-free inference pipeline** for lightweight, dependency-minimal deployment.

---

## Overview

Given a dataset of paintings paired with viewer metadata and response attributes, the model predicts how a viewer will respond to a given painting. The project emphasizes thoughtful feature engineering over model complexity — a well-tuned Logistic Regression achieves **~89.35% validation accuracy** and a **stratified 5-fold cross-validation accuracy of 0.8855 ± 0.0070**, outperforming more complex approaches that overfit to the training distribution.

---

## Feature Engineering (322 Features)

The feature representation combines three encoding strategies into a single dense vector per sample:

| Feature Type | Description | Details |
|---|---|---|
| **Standardized Numerics** | Continuous and ordinal fields (emotional intensity, prominent colours, objects noticed, log-transformed price, likert-scale feelings) | Zero-mean, unit-variance scaling applied to numeric columns to ensure equal contribution across features with different magnitudes. |
| **Multi-Hot Encodings** | Categorical fields (season association, room placement, viewing companion) | Categories are encoded as binary indicator vectors, supporting multi-label fields where a single sample can belong to multiple categories. |
| **Bag-of-Words (BoW)** | Free-text response fields (description, food association) | Tokenized using sklearn's `CountVectorizer` with unigrams and a max feature cap of ~200. A consistent tokenizer is used across training and inference to prevent vocabulary mismatches. |

All three representations are concatenated horizontally into a single 322-dimensional feature vector per sample.

### Ablation Study

A systematic feature ablation study was conducted to evaluate the impact of each feature group. Notably, removing the **soundtrack BoW feature improved accuracy by +0.30%**, indicating it introduced noise rather than useful signal. The most impactful features were **season association** (−4.14% when removed) and **likert emotions** (−2.37%), confirming the value of categorical and ordinal features for this task. The soundtrack feature was dropped from the final model accordingly.

---

## Model

- **Algorithm:** Logistic Regression (one-vs-rest for multiclass)
- **Regularization:** L2 with `C=0.1` (inverse regularization strength), selected via hyperparameter sweep over `C ∈ [0.01, 10.0]`, then narrowed to `[0.01, 0.2]`.
- **Validation Accuracy:** ~89.35%
- **Cross-Validation Accuracy:** 0.8855 ± 0.0070 (stratified 5-fold)
- **Training:** Fit using scikit-learn during development, then model weights exported for standalone inference.

---

## Model Exploration

Before selecting Logistic Regression, four model families were evaluated and compared using stratified 5-fold cross-validation:

| Model | Best Hyperparameters | Cross-Validation Accuracy |
|---|---|---|
| **Logistic Regression** | `C=0.1`, L2 regularization | **0.8796 ± 0.0090** |
| **Random Forest** | 200 trees, max depth 10 | 0.8772 ± 0.0116 |
| **MLP (Neural Network)** | 1 hidden layer (64 units), α=0.01 | 0.8766 ± 0.0111 |
| **Naive Bayes** | Gaussian NB and Complement NB variants explored | Lower than the above accuracies |

### Why Logistic Regression Was Chosen

All three top models performed within a narrow band (~0.87–0.88), but Logistic Regression was chosen for several reasons: it achieved the highest cross-validation accuracy, exhibited the lowest variance across folds (± 0.0090), and is the simplest to implement and deploy. Its linear decision boundary paired well with the carefully engineered feature set, and its weight vectors are directly interpretable — revealing which features (e.g., season, emotion likert scores) most strongly drive predictions.

The Random Forest showed signs of overfitting at higher depths (training accuracy approaching 98% while validation plateaued around 89%), and the MLP offered no meaningful accuracy gain over the linear model despite significantly higher complexity. Naive Bayes (both Gaussian and Complement variants) underperformed, likely due to its conditional independence assumption not holding well across the correlated feature groups.

### BoW Hyperparameter Tuning

After selecting Logistic Regression, further tuning was applied to the Bag-of-Words representation. Count-based unigrams consistently outperformed TF-IDF and bigram alternatives. The optimal max feature count was found to be approximately 200 features, beyond which additional vocabulary introduced diminishing returns and noise.

---

## Sklearn-Free Inference Pipeline

To meet the challenge's deployment constraints (minimal dependencies), the trained model parameters are exported to a `.npz` file and inference is performed using only the NumPy library:

```
Training Phase (sklearn allowed)          Inference Phase (sklearn-free)
┌──────────────────────────┐              ┌──────────────────────────┐
│  Load & preprocess data  │              │  Load model_params.npz   │
│  Fit LogisticRegression  │  ──export─ ▶ │  (weights, bias, vocab,  │
│  Validate (~89.35% acc)  │              │   scaler params, etc.)   │
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

### Inference Logic (Using NumPy)
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

> **Note:** The actual file structure may vary — the above reflects the logical separation of concerns.

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
- **Cross-validation accuracy of 0.8855 ± 0.0070** with a single Logistic Regression model (`C=0.1`, L2 regularization).
- **Systematic model selection** — evaluated Logistic Regression, Random Forest, MLP, and Naive Bayes; chose the simplest model that achieved the best generalization.
- **Ablation-driven feature selection** — identified and removed noisy features (soundtrack BoW) that improved accuracy by +0.30%.
- **Fully portable inference** — the prediction pipeline depends only on NumPy, with all preprocessing parameters serialized into a single `.npz` artifact.
- Diagnosed and resolved subtle **pandas Copy-on-Write** bugs and **tokenizer consistency** issues that silently degraded model performance.

---

## Technologies

- **Language:** Python
- **Training:** scikit-learn, pandas, NumPy
- **Inference:** NumPy only (sklearn-free)
- **Concepts:** Logistic Regression, feature engineering, bag-of-words, multi-hot encoding, standardization, model serialization, deployment-constrained inference, hyperparameter tuning, cross-validation, ablation studies

---

## Acknowledgements

Developed as part of the **CSC311 — Introduction to Machine Learning** challenge project at the University of Toronto Mississauga.
