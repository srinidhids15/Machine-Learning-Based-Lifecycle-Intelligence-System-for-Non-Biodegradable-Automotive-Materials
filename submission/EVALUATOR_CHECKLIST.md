# Evaluator Checklist — every requirement and where it is met

Each row names the file and the artefact that implements the requirement, so
each can be verified directly rather than taken on trust.

| # | Requirement | Status | Where it is implemented |
|---|---|---|---|
| 1 | Hyperparameter tuning | Done | `code/train.py` → `model_registry()`. GridSearchCV, 3-fold, on all 6 classical models; separate learning-rate search for the 3 neural models |
| 2 | Learning rate | Done | `code/train.py` → `train_deep_models()`. The **entire grid** is saved, not just the winner — see `results/results_summary_v2.json` → `lr_search` |
| 3 | Remove missing values | Done | `code/train.py` → `apply_missing_policy()`. Rows with missing values are **removed**: 9,000 → 8,440 (6.2%). All three policies are measured and compared |
| 4 | Problem statement (single line) | Done | `README.md`, top of the document |
| 5 | Epochs | Done | `code/deep_models.py` → explicit epoch loop. Boosting rounds for XGBoost/LightGBM/CatBoost; incremental trees for Random Forest/Extra Trees |
| 6 | Split 70 / 20 / 10 | Done | `code/train.py` → `make_splits()`. Train **5007 (70%)**, test **1430 (20%)**, validation **716 (10%)**, stratified |
| 7 | Validation loss & accuracy | Done | Recorded **every epoch** for 8 models. Early stopping monitors validation **loss** and restores the best-epoch weights |
| 8 | Multiple + hybrid algorithms | Done | **13 models per target**: 9 base + 4 hybrids (3 stacking, 1 soft-voting) |
| 9 | LSTM | Done | `code/deep_models.py` → `LSTMNet`, bidirectional LSTM over feature-tokens |
| 10 | Transformer | Done | `code/deep_models.py` → `TransformerNet`, encoder with `[CLS]` token and self-attention |
| 11 | Validation loss curves | Done | `results/disposition_training_curves.png`, `results/risk_training_curves.png`, plus per-model `*_curve_*.png` with the best epoch marked |
| 12 | High accuracy, minimal loss | Done | Best test accuracy **88.67%**; see the LightGBM finding below |
| 13 | Activation function | Done | Explicit for every network — never left at a library default |

---

## 2. Learning-rate search (Disposition target)

| Model | Grid searched (validation loss) | Chosen | Best epoch |
|---|---|---|---|
| MLP Neural Network | 0.0003 (loss 0.3356), 0.001 (loss 0.3330), 0.003 (loss 0.3331) | **0.001** | 6 |
| LSTM | 0.001 (loss 0.3344), 0.003 (loss 0.3305) | **0.003** | 34 |
| Transformer | 0.001 (loss 0.3272), 0.003 (loss 0.3266) | **0.003** | 10 |

## 13. Activation functions

| Model | Hidden / internal | Output |
|---|---|---|
| MLP | **ReLU** (with BatchNorm and Dropout) | Softmax |
| LSTM | **Sigmoid** on the input/forget/output gates and **Tanh** on the cell state — intrinsic to the LSTM cell — plus **Tanh** on the classification head | Softmax |
| Transformer | **GELU** in the feed-forward block, with LayerNorm and pre-norm residuals | Softmax |

Softmax is folded into `CrossEntropyLoss` during training for numerical
stability (log-sum-exp) and applied explicitly at prediction time.

Full strings as recorded by the pipeline:

- **MLP Neural Network** — ReLU (hidden), Softmax (output)
- **LSTM** — Sigmoid + Tanh (cell gates, intrinsic), Tanh (head), Softmax (output)
- **Transformer** — GELU (feed-forward block), Softmax (output)

---

## 12. Why validation loss mattered more than accuracy

This is the clearest evidence for the instruction to watch validation loss.

An **unregularised LightGBM** reached competitive *accuracy* while its
validation loss was more than double every other model:

| | Accuracy | Macro F1 | Validation loss |
|---|---|---|---|
| LightGBM, unregularised (`num_leaves=63`, 400 trees) | 0.8659 | 0.8416 | **0.7386** |
| LightGBM, regularised (`num_leaves=15`, L2=5, `min_child_samples=40`) | 0.8841 | 0.8645 | **0.3270** |

The model was overfitting into overconfident probabilities. Accuracy could not
see it; the training curve showed validation loss climbing steadily after about
20% of training while every other model stayed flat. Regularising **cut the loss
by more than half and raised accuracy at the same time**.

This also explains a design choice worth noting: model selection scores **macro
F1** rather than accuracy, because accuracy alone rewards a model that simply
predicts the majority class well and would have sacrificed the minority
Dispose class.
