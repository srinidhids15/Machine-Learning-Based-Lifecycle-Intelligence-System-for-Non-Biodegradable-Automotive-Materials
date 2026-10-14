# Machine Learning Based Lifecycle Intelligence System for Non-Biodegradable Automotive Materials

**Team 53** — Prajwal B T (NNM23IS128), Srinidhi D S (NNM23IS185), Swaroop Sathish Y (NNM23IS191), V K Ronith (NNM23IS203)
**Guide:** Pratheeksha Hegde N
**Department:** Information Science and Engineering, NMAM Institute of Technology, Nitte

## Problem Statement

> End-of-life vehicles accumulate faster than current waste-management practice can assess them, and there is no intelligent tool that decides whether a non-biodegradable automotive material should be **reused, recycled, or disposed of** — or how much **environmental risk** it carries if mishandled.

## Project Overview

Modern automobiles increasingly use non-biodegradable materials — composite fibers, reinforced plastics, and polymer-based materials — to improve performance. At end-of-life, these materials become an environmental hazard if not identified and handled correctly. This project builds a machine learning based lifecycle intelligence system that classifies automotive materials into **Reuse**, **Recycle**, or **Dispose** categories *and* predicts their **environmental risk level**, to support sustainable waste-management decisions before mechanical processing.

## Objectives

1. Analyze lifecycle data of non-biodegradable automotive materials; perform preprocessing and feature extraction.
2. Implement supervised ML algorithms to classify materials into recyclable, reusable, or disposal categories.
3. Evaluate and compare model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. Identify the best-performing model and build a lifecycle intelligence system supporting sustainable decision-making.

## Why the dataset was rebuilt

The Phase 1 dataset (`data/automotive_material_lifecycle_dataset.csv`) could not support the project's objectives. Measured independently of the training pipeline:

| Check | Result |
|---|---|
| Random Forest, 5-fold CV accuracy | 58.4% |
| Majority-class baseline | 45.0% |
| Highest mutual information (Toxicity Index) | 0.14 |
| Mutual information of `Vehicle Type`, `Manufacturing Year` | **0.0000** |

Only 13 points above baseline from 19 features means the labels were assigned close to independently of the features — which is why all five Phase 1 hybrids converged on the same ~0.556 validation accuracy. That is a noise floor, not a tuning problem. The data was also physically inconsistent: Copper Alloy part mass had mean 15.1 kg with a standard deviation of 100.1 kg, and carbon footprint did not track embodied energy.

The synopsis permits **"simulation-based lifecycle data"**. The project therefore now generates its data from a physics-grounded simulator rather than from arbitrary random draws. See `data/dataset_card.md` for full provenance, and keep describing the dataset as simulated in all reports.

## For the evaluator

Everything produced in Phase 2 is collected in **[`submission/`](submission/)** —
code, dataset, all charts, fitted models, and four short documents:

| File | Purpose |
|---|---|
| [`submission/README.md`](submission/README.md) | Overview and headline results |
| [`submission/EVALUATOR_CHECKLIST.md`](submission/EVALUATOR_CHECKLIST.md) | **Start here** — each requirement mapped to the file and result implementing it |
| [`submission/RESULTS.md`](submission/RESULTS.md) | Full per-model tables for both targets |
| [`submission/METHOD.md`](submission/METHOD.md) | Data provenance, design decisions, honest limitations |

Phase 1 outputs stay in `results/` and are deliberately not copied there, so the
submission folder shows one coherent set of current results. Rebuild it any time
with `python3 -m src.lifecycle.make_submission` — its documents are generated
from `results_summary_v2.json`, so the numbers can never drift from the ones the
pipeline actually produced.

## Repository Structure

```
├── src/lifecycle/
│   ├── material_properties.py     # Published property table (Ashby, GREET 2023, EU ELV)
│   ├── simulator.py               # Physics + degradation + MCDA label generator
│   ├── make_dataset.py            # Builds the dataset and its dataset card
│   ├── deep_models.py             # PyTorch MLP / LSTM / Transformer + epoch loop
│   ├── train.py                   # Preprocessing, 13 models, dual-target evaluation
│   └── make_submission.py         # Assembles the submission/ folder
├── submission/                    # <-- EVALUATOR REVIEW FOLDER (self-contained)
├── data/
│   ├── automotive_material_lifecycle_v2.csv    # 9,000 rows, 30 cols (current)
│   ├── automotive_material_lifecycle_dataset.csv  # Phase 1 dataset (kept for comparison)
│   ├── dataset_card.md                         # Provenance, method, caveats
│   └── _simulator_diagnostics.csv              # Latent state — DIAGNOSTICS ONLY, never train on this
├── models/                        # Fitted disposition and risk models (joblib)
├── results/                       # Metrics JSON, comparison charts, confusion matrices, SHAP
├── notebooks/                     # Phase 1 notebook
└── docs/, presentation/           # Synopsis, literature review, Phase 1 deck
```

## How the data is generated

1. **Physical anchors.** Density, embodied energy, cradle-to-gate GWP, recycling rate, recyclate value, tensile strength and hazard class are taken per material from published sources (Ashby *Materials and the Environment* 3rd ed.; Argonne GREET 2023; EU Directive 2000/53/EC; US EPA WARM v16). Part mass derives from the component's primary-material mass scaled by density ratio, so mass, embodied energy and carbon footprint are physically consistent.
2. **Degradation model.** Each part accumulates thermal, photo-oxidative and corrosive dose over its service life, weighted by its component's duty cycle and the vehicle type, under a first-order saturation law. This is why service life and odometer are genuinely predictive.
3. **Partially observed decision rule.** Labels are drawn from a multi-criteria utility model over latent lifecycle state, with Gumbel noise. Class composition is calibrated to the EU ELV stream (~20% reuse / ~50% recycle / ~30% dispose).

### The honest caveat

Labels come from a rule, so the models learn a **surrogate of that rule**. Reported accuracy measures surrogate fidelity, not field disposition accuracy. Three design choices keep the task genuine:

- **Unobserved heterogeneity.** Each part carries utility shocks standing for hidden internal damage, local recyclate market conditions, and operator/plant capability. None are exported, so no model can recover them. The rest of the latent state largely *is* recoverable from the features — correctly, since separability really is determined by a part's component, joining method and coating.
- **Two measured ceilings.** An oracle knowing the shocks scores **92.5%** (loose upper bound). Marginalising them out gives **89.1%** — the ceiling for any model restricted to the exported features, and the number results should be read against.
- **Extrapolation testing.** A whole material (PVC) and a whole component (Battery Casing) are withheld from training entirely.

## Evaluator requirements — where each is implemented

| # | Requirement | Where |
|---|---|---|
| 1 | Hyperparameter tuning | `GridSearchCV` (3-fold) on all 6 classical models; explicit learning-rate search for all 3 neural models |
| 2 | Learning rate | Searched per neural architecture; the **full grid** is saved to `results_summary_v2.json` under `lr_search`, not just the winner |
| 3 | Remove missing columns | `apply_missing_policy()` — **removes** rather than imputes. Default `drop_rows` (6.2% of rows, keeps all 30 features); `drop_columns` and `impute` also implemented, and `report_missing_strategies()` prints the cost of each |
| 4 | Problem statement (single line) | Top of this README |
| 5 | Epochs | Real epoch loop in `deep_models.py`; boosting rounds for XGB/LGBM/CatBoost; incremental trees for RF/ET |
| 6 | Split 70 / 20 / 10 | `TRAIN_FRACTION=0.70`, `TEST_FRACTION=0.20`, `VAL_FRACTION=0.10`, stratified |
| 7 | Validation loss & accuracy focus | Recorded **every epoch** for every model that has an iteration axis; early stopping monitors validation loss and restores best-epoch weights |
| 8 | Multiple + hybrid algorithms | 9 base models + 4 hybrids = **13 models**, per target |
| 9 | LSTM | `LSTMNet` — bidirectional LSTM over feature-tokens |
| 10 | Transformer | `TransformerNet` — encoder with `[CLS]` token and self-attention |
| 11 | Validation loss curves | `*_training_curves.png` (all models overlaid) + `*_curve_<model>.png` (per model, best epoch marked) |
| 12 | High accuracy, minimal loss | Both reported for every model; see the LightGBM note below |
| 13 | Activation functions | Explicit, never left at library defaults — see table below |

### Activation functions used

| Model | Hidden / internal | Output |
|---|---|---|
| MLP | **ReLU** (+ BatchNorm, Dropout) | Softmax |
| LSTM | **Sigmoid** on input/forget/output gates, **Tanh** on cell state (intrinsic to the LSTM cell); **Tanh** on the classification head | Softmax |
| Transformer | **GELU** in the feed-forward block; LayerNorm, pre-norm residuals | Softmax |

Softmax is folded into `CrossEntropyLoss` during training for numerical stability (log-sum-exp) and applied explicitly at prediction time.

## Results

Dataset: 9,000 simulated parts → **8,440 after removing rows with missing values** (6.2%), 36 columns after feature engineering.
Split **70% train (5,007) / 20% test (1,430) / 10% validation (716)**, plus a 1,287-row extrapolation set held out entirely.

### Target 1 — Disposition (Reuse / Recycle / Dispose)

Majority baseline **53.4%** · feature-based ceiling **89.1%**

| Model | Val Accuracy | Val Macro F1 | **Val Loss** |
|---|---|---|---|
| **Transformer** | **0.8855** | **0.8679** | 0.3066 |
| Random Forest | 0.8855 | 0.8668 | 0.3095 |
| SVM (RBF) | 0.8841 | 0.8667 | 0.3291 |
| LightGBM | 0.8841 | 0.8645 | 0.3270 |
| LSTM | 0.8813 | 0.8641 | 0.3225 |
| Extra Trees | 0.8827 | 0.8640 | **0.2982** |
| MLP Neural Network | 0.8813 | 0.8638 | 0.3120 |
| Hybrid C (soft voting) | 0.8841 | 0.8638 | 0.3102 |
| Hybrid A (RF+XGB+CatBoost → LR) | 0.8827 | 0.8634 | 0.3476 |
| CatBoost | 0.8827 | 0.8633 | 0.3135 |
| XGBoost | 0.8827 | 0.8622 | 0.3355 |
| Hybrid D (RF+XGB → XGB meta) | 0.8841 | 0.8621 | 0.3126 |
| Hybrid B (XGB+LGBM+CatBoost → LR) | 0.8799 | 0.8613 | 0.3580 |

Best model (**Transformer**, lr 0.003, best epoch 10) on the held-out **test** set:
**accuracy 0.8867**, macro F1 0.8690, **loss 0.3092**.
Lowest validation loss overall: **Extra Trees (0.2982)**.

### Target 2 — Environmental Risk (Low / Moderate / High / Critical)

The synopsis's second required output, which Phase 1 did not implement. Majority baseline **34.3%**.

| Model | Val Accuracy | Val Macro F1 | **Val Loss** |
|---|---|---|---|
| **Extra Trees** | **0.8771** | **0.8737** | 0.3219 |
| SVM (RBF) | 0.8743 | 0.8724 | 0.3508 |
| Transformer | 0.8729 | 0.8710 | **0.3095** |
| LSTM | 0.8687 | 0.8672 | 0.3168 |
| Hybrid D (RF+XGB → XGB meta) | 0.8729 | 0.8671 | 0.3356 |
| Random Forest | 0.8687 | 0.8651 | 0.3319 |
| CatBoost | 0.8575 | 0.8574 | 0.3358 |
| MLP Neural Network | 0.8520 | 0.8517 | 0.3319 |
| XGBoost | 0.8464 | 0.8436 | 0.4277 |
| LightGBM | 0.8380 | 0.8360 | 0.4086 |

Best model (**Extra Trees**) on test: **accuracy 0.8594**, macro F1 0.8561.

### Findings worth presenting

- **Every model sits at the information-theoretic ceiling.** The Transformer's 88.6% validation accuracy against a measured **89.1% ceiling** means the remaining error is irreducible given the observable features — not a tuning failure. This is a far stronger claim than a bare accuracy number, and it pre-empts "why not 95%?"
- **Accuracy and loss disagreed, and loss caught a real bug.** An unregularised LightGBM reached competitive *accuracy* (0.866) while its validation loss was **0.739** — more than double every other model. Its training curve climbed steadily after ~20% of training: classic overfitting producing overconfident probabilities, invisible to accuracy alone. Adding `num_leaves` capping, L2 (`reg_lambda=5`) and `min_child_samples=40` cut the loss to **0.327** *and* raised accuracy to 0.884. This is the clearest argument for the evaluator's instruction to watch validation loss, not just accuracy.
- **The Transformer beats the LSTM, and the architecture explains why.** Both read the same feature-token sequence, but feature ORDER here is arbitrary. The LSTM's sequential inductive bias assumes an ordering that does not exist, while self-attention is permutation-equivariant and therefore the correct match. The LSTM also needed ~3× more epochs to converge (best epoch 34 vs 10).
- **Hybrids still do not beat the best single model** — reproducing the Phase 1 finding, now on data with real signal. The base learners make highly correlated errors, so there is little left for a meta-learner to combine.
- **The models do not extrapolate to unseen materials.** On PVC parts and Battery Casings never seen in training, disposition accuracy falls from 88.7% to **69.7%**, and risk accuracy from 85.9% to **50.1%**. This is the most important limitation in the project: the system interpolates within known materials and must be retrained before being applied to a new material class.
- **Class imbalance is resolved.** Dispose now reaches 0.90 precision / 0.89 recall, and balanced accuracy tracks raw accuracy closely.

## Pipeline improvements over Phase 1

| Phase 1 | Now |
|---|---|
| Dropped 201 of 1,800 rows with missing values (11%) | Missing rows **removed** at a 6.2% cost, with all three policies measured and compared |
| Label-encoded ordered categories to arbitrary integers | **Ordinal encoding** preserving Poor < Fair < Good < Excellent |
| No class weighting; Dispose under-predicted | **Balanced class weights** across all models |
| Selected on accuracy | Selected on **macro F1**, so the minority class counts |
| Accuracy only | Accuracy, balanced accuracy, macro/weighted F1, **per-epoch validation loss**, ROC-AUC, and an **asymmetric cost matrix** |
| Single target | **Two targets**: disposition and environmental risk |
| Random split 70/20/10 | Same **70/20/10** split, **plus an extrapolation holdout** of an unseen material and component |
| No explainability | **SHAP** global importance and per-part waterfall |
| 7 models, no deep learning | **13 models** including **LSTM** and **Transformer**, trained with explicit epoch loops |

The cost matrix encodes that releasing a part needing hazardous disposal into the reuse stream (cost 8.0) is far worse than over-cautiously disposing of a recyclable part (cost 1.5).

## Running it

```bash
pip3 install pandas numpy scikit-learn xgboost lightgbm catboost torch matplotlib seaborn shap joblib

python3 -m src.lifecycle.make_dataset   # regenerate data (deterministic, seed 42)
python3 -m src.lifecycle.train          # train, evaluate, write results/
python3 -m src.lifecycle.make_submission  # rebuild submission/ for review
```

## Status

Phase 1 complete (synopsis, literature review, presentation). Data foundation rebuilt and the full dual-target pipeline retrained against measured ceilings. All 13 evaluator requirements implemented and verified (see the table above).

**Next:** the Streamlit lifecycle intelligence interface that closes Objective 4, whole-vehicle teardown mode, and confidence-based abstention.
