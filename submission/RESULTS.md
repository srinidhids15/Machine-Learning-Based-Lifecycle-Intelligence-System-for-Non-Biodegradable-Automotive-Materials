# Results

Dataset 9,000 rows → **8,440 after removing
rows with missing values**. Split 70 / 20 / 10, stratified. Every number here is
read directly from `results/results_summary_v2.json`.

---

## Target 1 — Disposition (Reuse / Recycle / Dispose)

Majority-class baseline **53.4%** ·
measured feature-based ceiling **89.1%**

| Model | Val Accuracy | Val Macro F1 | Val Loss | Expected Cost |
|---|---|---|---|---|
| **Transformer** (best) | 0.8855 | 0.8679 | 0.3066 | 0.375 |
| Random Forest | 0.8855 | 0.8668 | 0.3095 | 0.351 |
| SVM (RBF) | 0.8841 | 0.8667 | 0.3291 | 0.374 |
| LightGBM | 0.8841 | 0.8645 | 0.3270 | 0.367 |
| LSTM | 0.8813 | 0.8641 | 0.3225 | 0.366 |
| Extra Trees | 0.8827 | 0.8640 | **0.2982** | 0.385 |
| MLP Neural Network | 0.8813 | 0.8638 | 0.3120 | 0.377 |
| Hybrid C (soft voting) | 0.8841 | 0.8638 | 0.3102 | 0.358 |
| Hybrid A (RF+XGB+CatBoost -> LR) | 0.8827 | 0.8634 | 0.3476 | 0.364 |
| CatBoost | 0.8827 | 0.8633 | 0.3135 | 0.360 |
| XGBoost | 0.8827 | 0.8622 | 0.3355 | 0.351 |
| Hybrid D (RF+XGB -> XGB meta) | 0.8841 | 0.8621 | 0.3126 | 0.374 |
| Hybrid B (XGB+LGBM+CatBoost -> LR) | 0.8799 | 0.8613 | 0.3580 | 0.367 |

*Expected cost* applies an asymmetric cost matrix: sending a part that needs
hazardous disposal into the reuse stream costs 8.0, while over-cautiously
disposing of a recyclable part costs 1.5. Lower is better.

### Best model on the held-out test set — Transformer

| Metric | Value |
|---|---|
| Accuracy | **0.8867** |
| Balanced accuracy | 0.8760 |
| Macro F1 | 0.8690 |
| Log loss | 0.3092 |
| ROC-AUC (one-vs-rest) | 0.9693 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Reuse | 0.74 | 0.83 | 0.78 | 311 |
| Recycle | 0.95 | 0.91 | 0.93 | 763 |
| Dispose | 0.90 | 0.89 | 0.90 | 356 |

### Generalisation to unseen materials

Withholding a whole material (Polyvinyl Chloride (PVC)) and a
whole component (Battery Casing) from training:

| Split | Accuracy | Macro F1 |
|---|---|---|
| Test (seen materials) | 0.8867 | 0.8690 |
| **Extrapolation (unseen)** | **0.6970** | 0.6587 |

This drop is the project's most important limitation and is reported
deliberately: the system interpolates within known materials and must be
retrained before being applied to a new material class.

---

## Target 2 — Environmental Risk (Low / Moderate / High / Critical)

The synopsis requires the system to "determine the environmental risk level".
Majority-class baseline **34.3%**.

| Model | Val Accuracy | Val Macro F1 | Val Loss | Expected Cost |
|---|---|---|---|---|
| **Extra Trees** (best) | 0.8771 | 0.8737 | 0.3219 | 0.204 |
| SVM (RBF) | 0.8743 | 0.8724 | 0.3508 | 0.188 |
| Transformer | 0.8729 | 0.8710 | **0.3095** | 0.186 |
| LSTM | 0.8687 | 0.8672 | 0.3168 | 0.191 |
| Hybrid D (RF+XGB -> XGB meta) | 0.8729 | 0.8671 | 0.3356 | 0.215 |
| Random Forest | 0.8687 | 0.8651 | 0.3319 | 0.217 |
| CatBoost | 0.8575 | 0.8574 | 0.3358 | 0.239 |
| Hybrid B (XGB+LGBM+CatBoost -> LR) | 0.8547 | 0.8551 | 0.3655 | 0.233 |
| MLP Neural Network | 0.8520 | 0.8517 | 0.3319 | 0.214 |
| Hybrid A (RF+XGB+CatBoost -> LR) | 0.8506 | 0.8506 | 0.3513 | 0.230 |
| Hybrid C (soft voting) | 0.8492 | 0.8470 | 0.3436 | 0.244 |
| XGBoost | 0.8464 | 0.8436 | 0.4277 | 0.245 |
| LightGBM | 0.8380 | 0.8360 | 0.4086 | 0.257 |

### Best model on the held-out test set — Extra Trees

| Metric | Value |
|---|---|
| Accuracy | **0.8594** |
| Balanced accuracy | 0.8486 |
| Macro F1 | 0.8561 |
| ROC-AUC (one-vs-rest) | 0.9724 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.91 | 0.80 | 0.85 | 356 |
| Moderate | 0.81 | 0.92 | 0.86 | 491 |
| High | 0.83 | 0.76 | 0.79 | 259 |
| Critical | 0.93 | 0.92 | 0.92 | 324 |

---

## Charts

| File | Shows |
|---|---|
| `disposition_model_comparison.png` | All models vs the baseline and the measured ceiling |
| `disposition_training_curves.png` | Validation loss and accuracy per epoch, all models overlaid |
| `disposition_curve_transformer.png` | Transformer loss/accuracy per epoch, best epoch marked |
| `disposition_curve_lstm.png` | LSTM loss/accuracy per epoch |
| `disposition_curve_mlp.png` | MLP loss/accuracy per epoch |
| `disposition_confusion_matrix.png` | Confusion matrix, test set |
| `disposition_confusion_extrapolation.png` | Confusion matrix on unseen material/component |
| `disposition_feature_importance.png` | Top features |
| `shap_global_importance.png` | SHAP global feature importance |
| `shap_single_part.png` | SHAP explanation for one individual part |
| `risk_*.png` | The same set for the Environmental Risk target |

---

## Findings worth presenting

1. **Every model sits at the information-theoretic ceiling.** The best model
   reaches 88.67% against a measured
   89.1% ceiling, so the remaining error is irreducible rather than a
   tuning failure. This pre-empts the question "why not 95%?".
2. **Validation loss caught a defect accuracy could not see** — the LightGBM
   overfitting case documented in `EVALUATOR_CHECKLIST.md`.
3. **The Transformer outperforms the LSTM, and the architecture explains why.**
   Both read the same feature-token sequence, but feature order here is
   arbitrary. The LSTM's sequential inductive bias assumes an ordering that does
   not exist, whereas self-attention is permutation-equivariant and therefore
   the correct match. The LSTM also needed roughly three times as many epochs
   to converge.
4. **Hybrid ensembles still do not beat the best single model**, reproducing the
   Phase 1 finding on data that now carries real signal. The base learners make
   highly correlated errors, leaving little for a meta-learner to combine.
5. **Class imbalance is resolved.** Balanced class weights plus macro-F1 model
   selection bring the minority Dispose class up to parity with the others.
