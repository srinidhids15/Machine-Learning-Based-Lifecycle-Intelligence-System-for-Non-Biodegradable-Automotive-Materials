# Machine Learning Based Lifecycle Intelligence System for Non-Biodegradable Automotive Materials

**Team 53** — Prajwal B T (NNM23IS128),  Srinidhi D S (NNM23IS185), Swaroop Sathish Y (NNM23IS191), V K Ronith (NNM23IS203),
**Guide:** Pratheeksha Hegde N
**Department:** Information Science and Engineering, NMAM Institute of Technology, Nitte

## Project Overview

Modern automobiles increasingly use non-biodegradable materials — composite fibers, reinforced plastics, and polymer-based materials — to improve performance. At end-of-life, these materials become an environmental hazard if not identified and handled correctly. This project builds a machine learning-based lifecycle intelligence system that classifies automotive materials into **recyclable**, **reusable**, or **disposal** categories and predicts their environmental risk level, to support smarter, more sustainable waste-management decisions before mechanical processing (crushing/shredding).

## Objectives

1. Analyze lifecycle data of non-biodegradable automotive materials; perform preprocessing and feature extraction.
2. Implement supervised ML algorithms to classify materials into recyclable, reusable, or disposal categories.
3. Evaluate and compare model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. Identify the best-performing model and build a lifecycle intelligence system supporting sustainable decision-making.

## Repository Structure

```
├── data/
│   └── automotive_material_lifecycle_dataset.csv   # 1,800-row dataset (materials, components, lifecycle features)
├── notebooks/
│   └── ML_Lifecycle_Intelligence.ipynb              # Preprocessing + multi-algorithm training pipeline
├── results/
│   ├── results_summary.json                        # Validation metrics per algorithm
│   ├── model_comparison.png                         # Validation accuracy across all 12 models
│   ├── confusion_matrix.png                         # Confusion matrix for best model (test set)
│   ├── rf_feature_importance.png                    # Random Forest feature importances
│   └── *_training_curves.png                        # Per-epoch loss/accuracy curves (RF, ET, XGBoost, LightGBM, CatBoost, MLP)
├── docs/
│   ├── literature_review_final.docx                 # Literature review, including research gap analysis
│   └── Presentation_Speaking_Notes.md                # Slide-by-slide speaking notes for Phase 1 presentation
└── presentation/
    └── ISE_Phase1_Presentation_Team53.pptx           # Phase 1 presentation deck
```

## Dataset

`automotive_material_lifecycle_dataset.csv` — 1,800 rows, 20 columns, covering 10 material types (Polyurethane, HDPE, Structural Steel, Copper Alloy, Aluminum Alloy, Glass, ABS, Natural Rubber, Polypropylene, CFRP) across 10 vehicle components and 3 vehicle types (ICE/Hybrid/Electric). Target classes: **Recycle**, **Reuse**, **Dispose**. Includes lifecycle/environmental features (Carbon Footprint, Toxicity Index, Recycling Efficiency, Hazardous Level, Energy Consumption, etc.).

## Methodology / Pipeline

`notebooks/ML_Lifecycle_Intelligence.ipynb` runs the full pipeline:

1. **Preprocessing** — drops rows with missing values (~11% of rows affected across 6 numeric columns), encodes categorical features.
2. **Split** — 70% train / 20% test / 10% validation (stratified).
3. **Base models trained** — Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, SVM, and an MLP Neural Network. Boosting models (XGBoost/LightGBM/CatBoost) and the MLP track true per-epoch/per-round validation loss & accuracy; Random Forest/Extra Trees get an equivalent simulated curve via `warm_start`.
4. **Hybrid (ensemble) models** — 5 combinations: 3 stacking variants (RF+XGBoost→LR, RF+XGBoost+CatBoost→LR, XGBoost+LightGBM+CatBoost→LR), 1 soft-voting ensemble (RF+XGBoost+CatBoost), and 1 non-linear-meta-learner stack (RF+XGBoost→XGBoost).
5. **Hyperparameter tuning** — GridSearchCV (3-fold CV) on Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost.
6. **Evaluation** — validation loss & accuracy tracked per epoch/round for every applicable model; final comparison across all 12 models on accuracy, precision, recall, F1; confusion matrix on the held-out test set for the best model.

### Running it

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn jupyter
jupyter notebook notebooks/ML_Lifecycle_Intelligence.ipynb
```

Run all cells top to bottom; outputs (plots + `results_summary.json`) are written to `results/`.

### Current Results (Phase 1, extended model set)

| Model | Validation Accuracy | Validation Loss |
|---|---|---|
| Random Forest | 0.544 | 0.888 |
| Extra Trees | 0.563 | 0.873 |
| XGBoost | 0.544 | 0.903 |
| LightGBM | 0.544 | 0.887 |
| **CatBoost** | **0.581** | **0.880** |
| SVM | 0.431 | 1.038 |
| MLP Neural Network | 0.550 | 1.034 |
| Hybrid 1 (RF+XGBoost→LR) | 0.556 | 0.867 |
| Hybrid 2 (RF+XGBoost+CatBoost→LR) | 0.556 | 0.864 |
| Hybrid 3 (XGBoost+LightGBM+CatBoost→LR) | 0.556 | 0.854 |
| Hybrid 4 (RF+XGBoost+CatBoost Soft-Voting) | 0.550 | 0.881 |
| Hybrid 5 (RF+XGBoost→XGBoost meta) | 0.513 | 0.885 |

Best model overall is **CatBoost** (single model, not a hybrid), reaching **60.6% test accuracy** — currently beating every stacking/voting hybrid tried. This is a genuinely useful finding worth stating in your presentation: the extra base learners didn't help the hybrids surpass CatBoost alone, most likely because CatBoost's own base error rate is already close to what the meta-learners have to work with. Known issue to address next: the model under-predicts the minority "Dispose" class — flagged as a next step (class-weighting / resampling).

## Status

Phase 1 complete: synopsis, literature review, and presentation finalized. Baseline preprocessing and multi-algorithm training pipeline implemented on the available dataset per Phase 1 evaluator feedback. Next: address class imbalance and overfitting, then finalize the model for Demo 1.
