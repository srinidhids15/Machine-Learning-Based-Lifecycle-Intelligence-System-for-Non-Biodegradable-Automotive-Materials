"""
Team 53 — Machine Learning Based Lifecycle Intelligence System
Preprocessing + Model Training Pipeline (Demo 1 progress)

Addresses Phase 1 evaluator feedback:
  - Remove rows with missing values
  - 70% train / 20% test / 10% validation split
  - Multiple algorithms, including a hybrid (stacked) model
  - Learning rate + epoch/boosting-round tracking (XGBoost)
  - Hyperparameter tuning (GridSearchCV)
  - Validation loss & validation accuracy tracked and plotted
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("automotive_material_lifecycle_dataset.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. REMOVE ROWS WITH MISSING VALUES  (evaluator feedback: "remove missing")
#    Only ~2% of rows are affected, so dropping rows (not whole columns)
#    keeps every feature while losing minimal data.
# ---------------------------------------------------------------------------
before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows with missing values ({before} -> {after})")

# ---------------------------------------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------------------------
categorical_cols = [
    "Material Name", "Vehicle Type", "Component",
    "Hazardous Level", "Material Condition", "Corrosion Resistance"
]
target_col = "Target"

encoders = {}
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

target_encoder = LabelEncoder()
y_all = target_encoder.fit_transform(df_enc[target_col])
class_names = list(target_encoder.classes_)
print("Target classes:", class_names)

feature_cols = [c for c in df_enc.columns if c != target_col]
X_all = df_enc[feature_cols].values

# ---------------------------------------------------------------------------
# 4. DATASET SPLIT — 70% train / 20% test / 10% validation
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all
)
# temp (30%) -> split into test (20% of total) and validation (10% of total)
# 20/30 = 0.6667 of temp goes to test, 10/30 = 0.3333 goes to validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=(10/30), random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X_all):.1%})  "
      f"Test: {len(X_test)} ({len(X_test)/len(X_all):.1%})  "
      f"Validation: {len(X_val)} ({len(X_val)/len(X_all):.1%})")

results = {}

# ---------------------------------------------------------------------------
# 5A. MODEL 1 — Random Forest (with hyperparameter tuning)
# ---------------------------------------------------------------------------
print("\n=== Training Random Forest (with GridSearchCV tuning) ===")
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, None],
    "min_samples_split": [2, 5],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_param_grid, cv=3, scoring="accuracy", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

val_pred_rf = best_rf.predict(X_val)
val_proba_rf = best_rf.predict_proba(X_val)
results["Random Forest"] = {
    "best_params": rf_grid.best_params_,
    "val_accuracy": accuracy_score(y_val, val_pred_rf),
    "val_loss": log_loss(y_val, val_proba_rf),
    "val_precision": precision_score(y_val, val_pred_rf, average="macro"),
    "val_recall": recall_score(y_val, val_pred_rf, average="macro"),
    "val_f1": f1_score(y_val, val_pred_rf, average="macro"),
}

# ---------------------------------------------------------------------------
# 5B. MODEL 2 — XGBoost (tracks learning rate + boosting rounds/"epochs",
#     and validation loss/accuracy per round -- directly addresses feedback
#     on epoch count, learning rate, and validation loss/accuracy tracking)
# ---------------------------------------------------------------------------
print("\n=== Training XGBoost (tracking epochs / learning rate / val loss) ===")
xgb_param_grid = {
    "n_estimators": [150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [4, 6],
}
xgb_grid = GridSearchCV(
    XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_STATE, use_label_encoder=False
    ),
    xgb_param_grid, cv=3, scoring="accuracy", n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
best_params = xgb_grid.best_params_
print("Best XGBoost params:", best_params)

# Retrain the best configuration with an explicit eval_set so we can capture
# validation loss & validation accuracy at every boosting round ("epoch").
best_xgb = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    objective="multi:softprob",
    eval_metric=["mlogloss", "merror"],
    random_state=RANDOM_STATE,
)
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)
evals_result = best_xgb.evals_result()

val_pred_xgb = best_xgb.predict(X_val)
val_proba_xgb = best_xgb.predict_proba(X_val)
results["XGBoost"] = {
    "best_params": best_params,
    "val_accuracy": accuracy_score(y_val, val_pred_xgb),
    "val_loss": log_loss(y_val, val_proba_xgb),
    "val_precision": precision_score(y_val, val_pred_xgb, average="macro"),
    "val_recall": recall_score(y_val, val_pred_xgb, average="macro"),
    "val_f1": f1_score(y_val, val_pred_xgb, average="macro"),
}

# ---------------------------------------------------------------------------
# 5C. MODEL 3 — SVM (baseline comparison algorithm)
# ---------------------------------------------------------------------------
print("\n=== Training SVM ===")
svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)
val_pred_svm = svm.predict(X_val)
val_proba_svm = svm.predict_proba(X_val)
results["SVM"] = {
    "best_params": {"kernel": "rbf"},
    "val_accuracy": accuracy_score(y_val, val_pred_svm),
    "val_loss": log_loss(y_val, val_proba_svm),
    "val_precision": precision_score(y_val, val_pred_svm, average="macro"),
    "val_recall": recall_score(y_val, val_pred_svm, average="macro"),
    "val_f1": f1_score(y_val, val_pred_svm, average="macro"),
}

# ---------------------------------------------------------------------------
# 5D. MODEL 4 — HYBRID: Stacking ensemble (RF + XGBoost as base learners,
#     Logistic Regression as meta-learner). This directly addresses the
#     "try hybrid algorithms" feedback point.
# ---------------------------------------------------------------------------
print("\n=== Training Hybrid Stacking Ensemble (RF + XGBoost -> meta-learner) ===")
from sklearn.linear_model import LogisticRegression

hybrid = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(**rf_grid.best_params_, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(**best_params, objective="multi:softprob",
                               eval_metric="mlogloss", random_state=RANDOM_STATE)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
)
hybrid.fit(X_train, y_train)
val_pred_hybrid = hybrid.predict(X_val)
val_proba_hybrid = hybrid.predict_proba(X_val)
results["Hybrid (RF + XGBoost Stacked)"] = {
    "best_params": "RF + XGBoost base learners, Logistic Regression meta-learner",
    "val_accuracy": accuracy_score(y_val, val_pred_hybrid),
    "val_loss": log_loss(y_val, val_proba_hybrid),
    "val_precision": precision_score(y_val, val_pred_hybrid, average="macro"),
    "val_recall": recall_score(y_val, val_pred_hybrid, average="macro"),
    "val_f1": f1_score(y_val, val_pred_hybrid, average="macro"),
}

# ---------------------------------------------------------------------------
# 6. COMPARISON TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION SET RESULTS — MODEL COMPARISON")
print("=" * 70)
comp_df = pd.DataFrame(results).T[
    ["val_accuracy", "val_loss", "val_precision", "val_recall", "val_f1"]
]
comp_df.columns = ["Val Accuracy", "Val Loss", "Val Precision", "Val Recall", "Val F1"]
print(comp_df.round(4).to_string())

best_model_name = comp_df["Val Accuracy"].idxmax()
print(f"\nBest model on validation set: {best_model_name} "
      f"(Val Accuracy = {comp_df.loc[best_model_name, 'Val Accuracy']:.4f})")

# ---------------------------------------------------------------------------
# 7. FINAL TEST-SET EVALUATION OF BEST MODEL
# ---------------------------------------------------------------------------
model_lookup = {
    "Random Forest": best_rf,
    "XGBoost": best_xgb,
    "SVM": svm,
    "Hybrid (RF + XGBoost Stacked)": hybrid,
}
final_model = model_lookup[best_model_name]
test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"\nFinal held-out TEST accuracy for {best_model_name}: {test_acc:.4f}")
print("\nClassification report (test set):")
print(classification_report(y_test, test_pred, target_names=class_names))

# ---------------------------------------------------------------------------
# 8. PLOTS
# ---------------------------------------------------------------------------
# 8a. XGBoost validation loss & accuracy per boosting round ("epoch")
epochs = range(1, len(evals_result["validation_0"]["mlogloss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(epochs, evals_result["validation_0"]["mlogloss"], label="Train Loss")
axes[0].plot(epochs, evals_result["validation_1"]["mlogloss"], label="Validation Loss")
axes[0].set_xlabel("Boosting Round (Epoch)")
axes[0].set_ylabel("Log Loss")
axes[0].set_title(f"XGBoost Loss Curve (learning_rate={best_params['learning_rate']})")
axes[0].legend()
axes[0].grid(alpha=0.3)

train_acc_curve = [1 - e for e in evals_result["validation_0"]["merror"]]
val_acc_curve = [1 - e for e in evals_result["validation_1"]["merror"]]
axes[1].plot(epochs, train_acc_curve, label="Train Accuracy")
axes[1].plot(epochs, val_acc_curve, label="Validation Accuracy")
axes[1].set_xlabel("Boosting Round (Epoch)")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("XGBoost Accuracy Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("xgboost_training_curves.png", dpi=150)
plt.close()

# 8b. Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
comp_df["Val Accuracy"].plot(kind="bar", ax=ax, color="#4472C4")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Validation Accuracy by Algorithm")
ax.set_ylim(0, 1)
for i, v in enumerate(comp_df["Val Accuracy"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 8c. Confusion matrix for best model
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name} (Test Set)")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. SAVE RESULTS SUMMARY
# ---------------------------------------------------------------------------
summary = {
    "dataset_rows_after_cleaning": after,
    "rows_dropped_missing": before - after,
    "split": {"train": len(X_train), "test": len(X_test), "validation": len(X_val)},
    "results": {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                     for kk, vv in v.items()} for k, v in results.items()},
    "best_model": best_model_name,
    "best_model_test_accuracy": float(test_acc),
}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nSaved: xgboost_training_curves.png, model_comparison.png, "
      "confusion_matrix.png, results_summary.json")
