# Machine Learning Based Lifecycle Intelligence System for Non-Biodegradable Automotive Materials

**Team 53** — Prajwal B T (NNM23IS128), V K Ronith (NNM23IS203), Srinidhi D S (NNM23IS185), Swaroop Sathish Y (NNM23IS191)
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
├── src/
│   └── train_pipeline.py                           # Preprocessing + multi-algorithm training pipeline
├── results/
│   ├── results_summary.json                        # Validation metrics per algorithm
│   ├── xgboost_training_curves.png                  # Validation loss & accuracy per boosting round
│   ├── model_comparison.png                         # Validation accuracy across algorithms
│   └── confusion_matrix.png                         # Confusion matrix for best model (test set)
├── docs/
│   ├── literature_review_final.docx                 # Literature review, including research gap analysis
│   └── Presentation_Speaking_Notes.md                # Slide-by-slide speaking notes for Phase 1 presentation
└── presentation/
    └── ISE_Phase1_Presentation_Team53.pptx           # Phase 1 presentation deck
```

## Dataset

`automotive_material_lifecycle_dataset.csv` — 1,800 rows, 20 columns, covering 10 material types (Polyurethane, HDPE, Structural Steel, Copper Alloy, Aluminum Alloy, Glass, ABS, Natural Rubber, Polypropylene, CFRP) across 10 vehicle components and 3 vehicle types (ICE/Hybrid/Electric). Target classes: **Recycle**, **Reuse**, **Dispose**. Includes lifecycle/environmental features (Carbon Footprint, Toxicity Index, Recycling Efficiency, Hazardous Level, Energy Consumption, etc.).

## Methodology / Pipeline

`src/train_pipeline.py` runs the full pipeline:

1. **Preprocessing** — drops rows with missing values (~11% of rows affected across 6 numeric columns), encodes categorical features.
2. **Split** — 70% train / 20% test / 10% validation (stratified).
3. **Models trained** — Random Forest, XGBoost (with learning rate + boosting-round/epoch tracking), SVM, and a **hybrid stacking ensemble** (Random Forest + XGBoost base learners → Logistic Regression meta-learner).
4. **Hyperparameter tuning** — GridSearchCV on Random Forest and XGBoost.
5. **Evaluation** — validation loss & accuracy tracked per epoch/boosting round; final comparison across all algorithms on accuracy, precision, recall, F1; confusion matrix on the held-out test set for the best model.

### Running it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
cd src
python train_pipeline.py
```

### Current Results (baseline, Phase 1)

| Model | Validation Accuracy | Validation Loss |
|---|---|---|
| Random Forest | 0.506 | 0.895 |
| XGBoost | 0.544 | 0.994 |
| SVM | 0.431 | 1.038 |
| **Hybrid (RF + XGBoost Stacked)** | **0.563** | **0.893** |

Best model (Hybrid) reaches **57.8% test accuracy**. Known issue to address next: the model under-predicts the minority "Dispose" class, and XGBoost shows signs of overfitting past ~20 boosting rounds — both flagged as next steps (class-weighting / resampling, early stopping).

## Status

Phase 1 complete: synopsis, literature review, and presentation finalized. Baseline preprocessing and multi-algorithm training pipeline implemented on the available dataset per Phase 1 evaluator feedback. Next: address class imbalance and overfitting, then finalize the model for Demo 1.
