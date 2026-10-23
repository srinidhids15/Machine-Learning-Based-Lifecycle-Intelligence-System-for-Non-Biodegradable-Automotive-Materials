"""
Training and evaluation pipeline for the Lifecycle Intelligence System.

Covers synopsis Objectives 1-3:
  1. preprocessing and feature extraction (imputation, ordinal/nominal encoding,
     scaling, engineered circularity features)
  2. supervised classification into Reuse / Recycle / Dispose, PLUS the second
     head the synopsis requires - Environmental Risk level
  3. comparative evaluation on Accuracy, Precision, Recall, F1 and confusion
     matrix, against both the majority baseline and the simulator's measured
     Bayes ceiling

What this pipeline does differently from the Phase 1 version
-----------------------------------------------------------
* IMPUTES missing values instead of dropping the rows. The old pipeline dropped
  201 of 1800 rows (11%), discarding data and biasing the sample toward
  completely-recorded parts.
* ENCODES ORDERED CATEGORIES AS ORDINAL. 'Poor' < 'Fair' < 'Good' < 'Excellent'
  is real information that label-encoding to arbitrary integers destroys.
* WEIGHTS THE CLASSES, so the minority 'Dispose' class is not sacrificed. This
  was the known defect flagged at the end of Phase 1.
* SCORES COST-SENSITIVELY. Releasing a hazardous part into the recycling stream
  is far more damaging than the reverse error, so a cost matrix is reported
  alongside plain accuracy.
* TESTS EXTRAPOLATION. A held-out material and a held-out component, never seen
  during training, measure whether the model generalises across the material
  space or merely memorises the ones it was shown.

Run from the repository root:
    python3 -m src.lifecycle.train
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.base

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from . import simulator as sim
from .deep_models import TorchTabularClassifier

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "automotive_material_lifecycle_v2.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

TARGET_MAIN = "Disposition"
TARGET_RISK = "Environmental Risk"

# Materials and components withheld entirely from training, to test whether the
# model generalises to material/component combinations it has never seen.
HOLDOUT_MATERIAL = "Polyvinyl Chloride (PVC)"
HOLDOUT_COMPONENT = "Battery Casing"

RANDOM_STATE = 42

# Dataset split, per evaluator instruction: 70% training, 20% testing,
# 10% validation. Validation drives model selection and the early-stopping
# criterion; the test split is touched only once, for the final model.
TRAIN_FRACTION = 0.70
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10

# Missing-value policy, per evaluator instruction to REMOVE missing data rather
# than impute it.
#   "drop_rows"    - discard any row containing a missing value, keeping all
#                    30 features. This is the default: it is what the Phase 1
#                    notebook did under the earlier evaluator note, and it
#                    preserves every predictor.
#   "drop_columns" - discard any COLUMN containing a missing value. Reads the
#                    instruction literally, but costs 6 features including
#                    Measured Tensile Strength, one of the strongest condition
#                    signals available.
#   "impute"       - median/most-frequent imputation. Statistically preferable
#                    but NOT what was asked for; kept only for comparison.
# `report_missing_strategies` quantifies the cost of each so the choice can be
# justified rather than asserted.
MISSING_STRATEGY = "drop_rows"

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# Ordered categories, encoded as ordinal so the model can exploit their order.
ORDINAL_FEATURES = {
    "Material Condition": ["Poor", "Fair", "Good", "Excellent"],
    "Hazardous Level": ["Low", "Medium", "High"],
    "Fluid Contamination": ["No", "Yes"],
    "Coating Present": ["No", "Yes"],
    "Legacy Additive (pre-ELV)": ["No", "Yes"],
    "Safety Critical": ["No", "Yes"],
    "Reprocessable": ["No", "Yes"],
}

# Unordered categories, one-hot encoded.
NOMINAL_FEATURES = ["Material Name", "Material Family", "Component",
                    "Vehicle Type", "Joining Method"]

# ---------------------------------------------------------------------------
# Cost matrix for disposition errors
# ---------------------------------------------------------------------------
# Rows = true class, columns = predicted class, in order [Reuse, Recycle, Dispose].
# The asymmetry is the point. Sending a part that genuinely needs hazardous
# disposal into the reuse or recycling stream contaminates downstream material
# and creates a real environmental release, so those cells carry the highest
# cost. Over-cautiously disposing of a recyclable part wastes recoverable value
# and embodied carbon, which is a genuine but much smaller loss.
COST_MATRIX = np.array([
    [0.0, 1.0, 2.0],   # true Reuse
    [1.5, 0.0, 2.5],   # true Recycle
    [8.0, 6.0, 0.0],   # true Dispose
])

# Cost matrix for the Environmental Risk head, over
# [Low, Moderate, High, Critical]. The asymmetry here is the same in spirit:
# UNDER-estimating risk means a hazardous part is handled with insufficient
# containment, so those cells (below the diagonal) are penalised far more
# heavily than over-estimating it, which only wastes handling capacity. Cost
# also grows with how far the prediction misses by, since calling a Critical
# part Low is far worse than calling it High.
RISK_COST_MATRIX = np.array([
    [0.0, 0.5, 1.0, 1.5],   # true Low
    [2.0, 0.0, 0.5, 1.0],   # true Moderate
    [5.0, 2.5, 0.0, 0.5],   # true High
    [9.0, 6.0, 3.0, 0.0],   # true Critical
])

COST_MATRICES = {TARGET_MAIN: COST_MATRIX, TARGET_RISK: RISK_COST_MATRIX}


def engineer_features(df):
    """
    Derive circularity and economic ratios that the raw columns only imply.

    These are deliberately computed from observable quantities only - nothing
    here touches the simulator's latent state - and each has a physical reading
    that can be defended in a viva.
    """
    out = df.copy()
    eps = 1e-6

    # Recovery value per unit of dismantling effort: the core economic test a
    # dismantler actually applies when deciding whether to pull a part.
    out["Value per Dismantling Cost"] = (
        out["Recyclate Value ($)"] / (out["Dismantling Cost ($)"] + eps))

    # Carbon payback: CO2 avoided per dollar spent recovering the part.
    out["CO2 Avoided per Cost"] = (
        out["CO2 Avoided if Recycled (kg)"] / (out["Dismantling Cost ($)"] + eps))

    # Strength retention. The model is given virgin and measured strength
    # separately, but the RATIO is the quantity that carries the condition
    # signal, so stating it explicitly saves the model from having to learn
    # division from finite data.
    out["Strength Retention Ratio"] = (
        out["Measured Tensile Strength (MPa)"]
        / (out["Virgin Tensile Strength (MPa)"] + eps))

    # Carbon intensity per kg, independent of how large the part happens to be.
    out["Carbon Intensity (kg CO2eq/kg)"] = (
        out["Carbon Footprint (kg CO2eq)"] / (out["Part Mass (kg)"] + eps))

    # Ageing rate: how much distance this part accumulated per year in service.
    out["Annual Duty (km/yr)"] = (
        out["Odometer (km)"] / (out["Service Life (years)"] + eps))

    # Fraction of embodied energy recoverable through the recycling route.
    out["Recoverable Energy (MJ)"] = (
        out["Embodied Energy (MJ)"] * out["Recycling Efficiency (%)"] / 100.0)

    return out


def build_preprocessor(X):
    """
    Assemble the preprocessing ColumnTransformer.

    Numeric: median imputation then standardisation. Median rather than mean
    because several features are strongly right-skewed (part mass, odometer,
    recyclate value); standardisation because SVM and the MLP need it, and it
    is harmless to the tree models.
    """
    ordinal_cols = [c for c in ORDINAL_FEATURES if c in X.columns]
    nominal_cols = [c for c in NOMINAL_FEATURES if c in X.columns]
    numeric_cols = [c for c in X.columns
                    if c not in ordinal_cols and c not in nominal_cols]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=[ORDINAL_FEATURES[c] for c in ordinal_cols],
            handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ], remainder="drop")


def feature_names(preprocessor, X):
    """Recover post-transform feature names, for importance and SHAP plots."""
    names = []
    for name, _, cols in preprocessor.transformers_:
        if name == "nom":
            enc = preprocessor.named_transformers_["nom"].named_steps["encode"]
            names.extend(enc.get_feature_names_out(cols).tolist())
        else:
            names.extend(list(cols))
    return names


def expected_cost(y_true, y_pred, labels, cost_matrix):
    """
    Mean per-decision cost under the given cost matrix. Lower is better.

    Reported alongside accuracy because the two can disagree: a model can look
    accurate while making exactly the errors that matter most environmentally.
    """
    idx = {c: i for i, c in enumerate(labels)}
    costs = [cost_matrix[idx[t], idx[p]] for t, p in zip(y_true, y_pred)]
    return float(np.mean(costs))


def evaluate(model, X, y, labels, cost_matrix, needs_proba=True):
    """Score a fitted model on one split."""
    # CatBoost returns predictions shaped (n, 1) rather than (n,), so flatten
    # to a 1-D array of labels before any metric sees it.
    y_pred = np.asarray(model.predict(X)).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "expected_cost": expected_cost(y, y_pred, labels, cost_matrix),
    }
    if needs_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        try:
            metrics["log_loss"] = float(log_loss(y, proba, labels=list(model.classes_)))
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y, proba, multi_class="ovr", average="macro",
                labels=list(model.classes_)))
        except ValueError:
            pass
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class XGBWrapper(ClassifierMixin, BaseEstimator):
    """
    Thin adapter around XGBClassifier.

    XGBoost requires integer class targets, while the rest of this pipeline
    works with the human-readable string labels used throughout the project.
    This wrapper label-encodes on the way in and decodes on the way out, and
    applies balanced sample weights internally so that class balancing happens
    identically to the other models without GridSearchCV having to route a
    sample_weight fit parameter.
    """

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.1,
                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                 random_state=RANDOM_STATE, balanced=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.balanced = balanced

    # VotingClassifier and StackingClassifier call sklearn's is_classifier()
    # on each base estimator; this makes the wrapper pass that check on both
    # the modern tag-based path and the legacy attribute path.
    _estimator_type = "classifier"

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        self._le = LabelEncoder().fit(y)
        y_enc = self._le.transform(y)
        self.classes_ = self._le.classes_
        if sample_weight is None and self.balanced:
            sample_weight = compute_sample_weight("balanced", y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            random_state=self.random_state, n_jobs=-1,
            eval_metric="mlogloss", tree_method="hist",
            objective="multi:softprob")
        self.model_.fit(X, y_enc, sample_weight=sample_weight)
        # Surface the underlying importances so the shared plotting and SHAP
        # helpers treat this exactly like any other tree model.
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def model_registry(n_classes):
    """
    Candidate models with modest hyperparameter grids.

    Every model that supports it is given balanced class weights, so the
    minority Dispose class is not traded away for headline accuracy - the
    specific defect left open at the end of Phase 1.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    reg = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                   class_weight="balanced"),
            {"n_estimators": [300], "max_depth": [None, 14],
             "min_samples_leaf": [1, 3]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced"),
            {"n_estimators": [300], "max_depth": [None, 16],
             "min_samples_leaf": [1, 3]},
        ),
        "XGBoost": (
            XGBWrapper(),
            {"n_estimators": [400], "max_depth": [5, 8],
             "learning_rate": [0.06, 0.12]},
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                           class_weight="balanced"),
            # Regularisation is searched here, not left at defaults. An earlier
            # unregularised grid let LightGBM settle on num_leaves=63 with 400
            # trees, which reached competitive ACCURACY while producing badly
            # overconfident probabilities - validation log loss near 0.74 against
            # roughly 0.31 for every other model. Because model selection scores
            # macro F1, nothing penalised that miscalibration. Capping leaf
            # count and adding L2 plus a minimum leaf population keeps the
            # probabilities usable, which matters because validation LOSS is a
            # reported metric and the system's confidence estimates feed the
            # cost-sensitive decision layer.
            {"n_estimators": [300], "num_leaves": [15, 31],
             "learning_rate": [0.05, 0.1], "min_child_samples": [40],
             "reg_lambda": [5.0]},
        ),
        "CatBoost": (
            # train_dir keeps CatBoost's per-run logs out of the project root,
            # which it otherwise litters with a catboost_info/ directory.
            CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0,
                               auto_class_weights="Balanced",
                               train_dir=os.path.join(ROOT, "results", ".catboost_logs"),
                               allow_writing_files=False),
            {"iterations": [500], "depth": [6, 8], "learning_rate": [0.08]},
        ),
        "SVM (RBF)": (
            SVC(probability=True, random_state=RANDOM_STATE,
                class_weight="balanced"),
            {"C": [3.0, 10.0], "gamma": ["scale"]},
        ),
    }
    return reg


# Deep models are kept in a separate registry because they are tuned and
# trained differently: they use an explicit epoch loop, they early stop on
# validation loss against the REAL validation split, and their learning rate
# is searched directly rather than through GridSearchCV's inner CV. Each entry
# declares its activation function explicitly.
def deep_registry(epochs=80):
    """
    Neural architectures, with the learning-rate grid searched for each.

    LEARNING RATE is the single most important hyperparameter for all three,
    so it is searched explicitly and the full search is recorded in the results
    JSON rather than only the winner.
    """
    return {
        "MLP Neural Network": dict(
            arch="mlp", hidden=128, activation="relu", dropout=0.2,
            epochs=epochs, patience=15,
            lr_grid=[3e-4, 1e-3, 3e-3],
            note="3-layer feed-forward, ReLU activation, BatchNorm + Dropout",
        ),
        "LSTM": dict(
            arch="lstm", d_model=32, hidden=64, num_layers=2,
            activation="tanh", dropout=0.2, epochs=epochs, patience=15,
            lr_grid=[1e-3, 3e-3],
            note=("Bidirectional LSTM over feature-tokens; Sigmoid/Tanh gates "
                  "intrinsic to the cell, Tanh activation on the head"),
        ),
        "Transformer": dict(
            arch="transformer", d_model=32, num_layers=2, dropout=0.1,
            activation="gelu", epochs=epochs, patience=15,
            lr_grid=[1e-3, 3e-3],
            note=("Transformer encoder with [CLS] token and self-attention "
                  "across feature-tokens; GELU activation in the feed-forward "
                  "block"),
        ),
    }


def build_hybrids(tuned):
    """
    Ensemble combinations over the already-tuned base models.

    Phase 1 found that stacking did not beat the best single model. That result
    is worth re-testing on data that actually carries signal, because stacking
    has little to combine when every base learner is near the noise floor.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    def clone_of(name):
        import sklearn.base
        return sklearn.base.clone(tuned[name])

    hybrids = {}

    hybrids["Hybrid A (RF+XGB+CatBoost -> LR)"] = StackingClassifier(
        estimators=[("rf", clone_of("Random Forest")),
                    ("xgb", clone_of("XGBoost")),
                    ("cat", clone_of("CatBoost"))],
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced"),
        cv=3, n_jobs=-1, passthrough=False)

    hybrids["Hybrid B (XGB+LGBM+CatBoost -> LR)"] = StackingClassifier(
        estimators=[("xgb", clone_of("XGBoost")),
                    ("lgbm", clone_of("LightGBM")),
                    ("cat", clone_of("CatBoost"))],
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced"),
        cv=3, n_jobs=-1)

    hybrids["Hybrid C (soft voting)"] = VotingClassifier(
        estimators=[("rf", clone_of("Random Forest")),
                    ("xgb", clone_of("XGBoost")),
                    ("cat", clone_of("CatBoost")),
                    ("lgbm", clone_of("LightGBM"))],
        voting="soft", n_jobs=-1)

    # A non-linear combiner: lets the meta-learner model interactions between
    # base-learner disagreements rather than only a weighted vote.
    hybrids["Hybrid D (RF+XGB -> XGB meta)"] = StackingClassifier(
        estimators=[("rf", clone_of("Random Forest")),
                    ("xgb", clone_of("XGBoost"))],
        final_estimator=XGBWrapper(
            n_estimators=200, max_depth=3, learning_rate=0.1, balanced=False),
        cv=3, n_jobs=-1)

    return hybrids


def apply_missing_policy(df, strategy=None, verbose=True):
    """
    Remove missing data according to MISSING_STRATEGY.

    The evaluator asked for missing values to be REMOVED rather than imputed,
    so removal is what happens here. The counts are printed so the cost of the
    policy is visible rather than silent.
    """
    strategy = strategy or MISSING_STRATEGY
    n_before, c_before = df.shape
    total_missing = int(df.isna().sum().sum())
    affected_cols = df.columns[df.isna().any()].tolist()

    if strategy == "drop_rows":
        out = df.dropna().reset_index(drop=True)
    elif strategy == "drop_columns":
        out = df.drop(columns=affected_cols)
    elif strategy == "impute":
        out = df.copy()
    else:
        raise ValueError(f"unknown missing strategy: {strategy}")

    if verbose:
        print(f"Missing-value policy: {strategy}")
        print(f"  {total_missing} missing cells across "
              f"{len(affected_cols)} columns")
        print(f"  rows {n_before} -> {out.shape[0]} "
              f"({n_before - out.shape[0]} removed, "
              f"{100 * (n_before - out.shape[0]) / n_before:.1f}%)")
        print(f"  columns {c_before} -> {out.shape[1]} "
              f"({c_before - out.shape[1]} removed)")
    return out


def report_missing_strategies(df):
    """Quantify what each missing-value policy costs, for the report."""
    rows = []
    for strat in ("drop_rows", "drop_columns", "impute"):
        out = apply_missing_policy(df, strat, verbose=False)
        rows.append({"strategy": strat, "rows": int(out.shape[0]),
                     "columns": int(out.shape[1]),
                     "rows_lost_pct": round(100 * (len(df) - len(out)) / len(df), 2)})
    return rows


def make_splits(df, target):
    """
    Build four evaluation splits.

    Proportions follow the evaluator instruction exactly: 70% train, 20% test,
    10% validation, stratified on the target so every class keeps its share in
    each split.

    The random split measures interpolation. The separate extrapolation split
    withholds a whole material and a whole component from training, which asks
    the harder question: does the model transfer to parts it has never seen?
    """
    extrap_mask = ((df["Material Name"] == HOLDOUT_MATERIAL)
                   | (df["Component"] == HOLDOUT_COMPONENT))
    main = df[~extrap_mask].reset_index(drop=True)
    extrap = df[extrap_mask].reset_index(drop=True)

    y = main[target]
    X = main.drop(columns=[TARGET_MAIN, TARGET_RISK])

    # First peel off the 30% that is not training data...
    holdout_fraction = TEST_FRACTION + VAL_FRACTION
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y,
        random_state=RANDOM_STATE)
    # ...then divide it 20:10, so validation is one third of the holdout.
    val_share_of_holdout = VAL_FRACTION / holdout_fraction
    X_te, X_val, y_te, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_share_of_holdout, stratify=y_tmp,
        random_state=RANDOM_STATE)

    X_ex = extrap.drop(columns=[TARGET_MAIN, TARGET_RISK])
    y_ex = extrap[target]

    return dict(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
                X_test=X_te, y_test=y_te, X_extrap=X_ex, y_extrap=y_ex)


# ---------------------------------------------------------------------------
# Per-epoch training curves
# ---------------------------------------------------------------------------
# The evaluator asked for epoch-wise training with validation loss and
# validation accuracy tracked throughout. Different model families express
# "epoch" differently, so each is handled on its own terms rather than faked:
#
#   neural networks  - a true epoch: one full pass over the training set
#   XGBoost/LightGBM - one boosting round, via the library's own eval_set
#                      callback, which records validation loss per round
#   CatBoost         - one boosting iteration, via its eval-set history
#   Random Forest /  - trees added incrementally with warm_start, re-scoring
#   Extra Trees        validation after each increment
#   SVM              - NOT iterative. SVC solves a convex dual problem in one
#                      shot, so it has no epoch axis. Reported as such rather
#                      than invented.
def compute_training_curve(name, estimator, pre, X_tr, y_tr, X_val, y_val,
                           labels, max_points=60):
    """
    Return {'epoch', 'val_loss', 'val_acc', ...} for one model, or None if the
    model has no meaningful iteration axis.
    """
    A = pre.transform(X_tr)
    B = pre.transform(X_val)
    classes = sorted(labels)

    # --- neural networks: history recorded during fit --------------------
    if isinstance(estimator, TorchTabularClassifier):
        h = getattr(estimator, "history_", None)
        if not h:
            return None
        return {"epoch": h["epoch"], "train_loss": h["train_loss"],
                "val_loss": h["val_loss"], "train_acc": h["train_acc"],
                "val_acc": h["val_acc"], "lr": h["lr"], "unit": "epoch"}

    # --- random forest / extra trees: grow the ensemble incrementally -----
    if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)):
        import sklearn.base
        est = sklearn.base.clone(estimator)
        total = est.get_params().get("n_estimators", 300)
        step = max(total // 30, 1)
        est.set_params(warm_start=True, n_estimators=step)
        epochs, vl, va, tl, ta = [], [], [], [], []
        for k in range(step, total + 1, step):
            est.set_params(n_estimators=k)
            est.fit(A, y_tr)
            vp, tp = est.predict_proba(B), est.predict_proba(A)
            epochs.append(k)
            vl.append(float(log_loss(y_val, vp, labels=list(est.classes_))))
            tl.append(float(log_loss(y_tr, tp, labels=list(est.classes_))))
            va.append(float(accuracy_score(y_val, est.predict(B))))
            ta.append(float(accuracy_score(y_tr, est.predict(A))))
        return {"epoch": epochs, "train_loss": tl, "val_loss": vl,
                "train_acc": ta, "val_acc": va, "unit": "trees"}

    # --- XGBoost: boosting rounds ----------------------------------------
    if isinstance(estimator, XGBWrapper):
        from xgboost import XGBClassifier
        le = LabelEncoder().fit(y_tr)
        model = XGBClassifier(
            n_estimators=estimator.n_estimators, max_depth=estimator.max_depth,
            learning_rate=estimator.learning_rate, subsample=estimator.subsample,
            colsample_bytree=estimator.colsample_bytree,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            eval_metric=["mlogloss", "merror"], objective="multi:softprob")
        model.fit(A, le.transform(y_tr),
                  sample_weight=compute_sample_weight("balanced", y_tr),
                  eval_set=[(A, le.transform(y_tr)), (B, le.transform(y_val))],
                  verbose=False)
        r = model.evals_result()
        return {"epoch": list(range(1, len(r["validation_1"]["mlogloss"]) + 1)),
                "train_loss": r["validation_0"]["mlogloss"],
                "val_loss": r["validation_1"]["mlogloss"],
                "train_acc": [1 - e for e in r["validation_0"]["merror"]],
                "val_acc": [1 - e for e in r["validation_1"]["merror"]],
                "unit": "boosting round"}

    # --- LightGBM: boosting rounds ---------------------------------------
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        if isinstance(estimator, LGBMClassifier):
            import sklearn.base
            model = sklearn.base.clone(estimator)
            rec = {}
            model.fit(A, y_tr, eval_set=[(A, y_tr), (B, y_val)],
                      eval_names=["train", "valid"], eval_metric="multi_logloss",
                      callbacks=[log_evaluation(0),
                                 __import__("lightgbm").record_evaluation(rec)])
            vlos = rec["valid"]["multi_logloss"]
            tlos = rec["train"]["multi_logloss"]
            return {"epoch": list(range(1, len(vlos) + 1)),
                    "train_loss": tlos, "val_loss": vlos,
                    "train_acc": None, "val_acc": None,
                    "unit": "boosting round"}
    except ImportError:
        pass

    # --- CatBoost: iterations --------------------------------------------
    try:
        from catboost import CatBoostClassifier
        if isinstance(estimator, CatBoostClassifier):
            import sklearn.base
            model = sklearn.base.clone(estimator)
            model.set_params(verbose=0)
            model.fit(A, y_tr, eval_set=(B, y_val))
            r = model.get_evals_result()
            key = "validation" if "validation" in r else list(r)[-1]
            metric = [m for m in r[key] if "Logloss" in m or "MultiClass" in m]
            if not metric:
                return None
            vlos = r[key][metric[0]]
            tlos = r.get("learn", {}).get(metric[0])
            return {"epoch": list(range(1, len(vlos) + 1)),
                    "train_loss": tlos, "val_loss": vlos,
                    "train_acc": None, "val_acc": None, "unit": "iteration"}
    except ImportError:
        pass

    # --- SVM and anything else: no iteration axis -------------------------
    return None


def plot_training_curves(curves, path, title):
    """
    Plot validation loss and validation accuracy per epoch for every model
    that has an iteration axis. Validation loss is shown on the left, since
    that is the quantity being minimised and the early-stopping criterion.
    """
    curves = {k: v for k, v in curves.items() if v}
    if not curves:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    cmap = plt.get_cmap("tab10")
    for i, (name, c) in enumerate(sorted(curves.items())):
        colour = cmap(i % 10)
        # Normalise the x-axis to 0-100% of training so models with very
        # different iteration counts (60 epochs vs 400 rounds) stay comparable.
        x = np.linspace(0, 100, len(c["epoch"]))
        ax1.plot(x, c["val_loss"], label=name, color=colour, lw=1.7)
        if c.get("val_acc"):
            ax2.plot(x, c["val_acc"], label=name, color=colour, lw=1.7)
    ax1.set_xlabel("Training progress (%)"); ax1.set_ylabel("Validation loss")
    ax1.set_title("Validation loss per epoch"); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Training progress (%)"); ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Validation accuracy per epoch"); ax2.grid(alpha=0.3)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_single_curve(curve, name, path):
    """Loss and accuracy for one model, with the best epoch marked."""
    if not curve:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    e = curve["epoch"]
    if curve.get("train_loss"):
        ax1.plot(e, curve["train_loss"], label="train loss", color="#8ab4d8")
    ax1.plot(e, curve["val_loss"], label="validation loss", color="#b3261e", lw=1.8)
    best = int(np.argmin(curve["val_loss"]))
    ax1.axvline(e[best], ls="--", color="#2e7d32", lw=1.3,
                label=f"best epoch {e[best]} (loss {curve['val_loss'][best]:.4f})")
    ax1.set_xlabel(curve.get("unit", "epoch")); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_title(f"{name} - loss")

    if curve.get("val_acc"):
        if curve.get("train_acc"):
            ax2.plot(e, curve["train_acc"], label="train accuracy", color="#8ab4d8")
        ax2.plot(e, curve["val_acc"], label="validation accuracy",
                 color="#3b6ea5", lw=1.8)
        ax2.axvline(e[best], ls="--", color="#2e7d32", lw=1.3)
        ax2.set_xlabel(curve.get("unit", "epoch")); ax2.set_ylabel("Accuracy")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_title(f"{name} - accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_model_comparison(results, ceiling, baseline, path):
    names = list(results.keys())
    acc = [results[n]["validation"]["accuracy"] for n in names]
    f1 = [results[n]["validation"]["f1_macro"] for n in names]
    order = np.argsort(acc)
    names = [names[i] for i in order]
    acc = [acc[i] for i in order]
    f1 = [f1[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 0.52 * len(names) + 2.6))
    ax.barh(y - 0.2, acc, height=0.4, label="Accuracy", color="#3b6ea5")
    ax.barh(y + 0.2, f1, height=0.4, label="Macro F1", color="#8ab4d8")
    ax.axvline(ceiling, color="#2e7d32", ls="--", lw=1.6,
               label=f"Feature-based ceiling ({ceiling:.1%})")
    ax.axvline(baseline, color="#b3261e", ls=":", lw=1.6,
               label=f"Majority baseline ({baseline:.1%})")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score (validation split)")
    ax.set_title("Model comparison - Disposition classification", fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_model_comparison_lines(summary, path):
    """
    Line graph of validation accuracy and macro F1 across all models, one panel
    per target. Reads the run summary, so it can be redrawn from
    results_summary_v2.json without retraining. Models are ordered by macro F1
    (the selection metric), so the best model sits at the right of each panel.
    """
    panels = [("disposition", "Disposition (Reuse / Recycle / Dispose)"),
              ("environmental_risk", "Environmental Risk (Low / Moderate / High / Critical)")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for ax, (key, title) in zip(axes, panels):
        s = summary[key]
        names = sorted(s["models"], key=lambda n: s["models"][n]["validation"]["f1_macro"])
        acc = [s["models"][n]["validation"]["accuracy"] for n in names]
        f1 = [s["models"][n]["validation"]["f1_macro"] for n in names]
        x = np.arange(len(names))
        ax.plot(x, acc, "-o", color="#2a78d6", lw=2, ms=8, label="Validation accuracy")
        ax.plot(x, f1, "-s", color="#eb6834", lw=2, ms=8, label="Validation macro F1")
        ax.annotate(f"{acc[-1]:.3f}", (x[-1], acc[-1]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10)
        ax.annotate(f"{f1[-1]:.3f}", (x[-1], f1[-1]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=10)
        lo, hi = min(acc + f1), max(acc + f1)
        ceiling = s.get("feature_based_ceiling")
        if ceiling:
            ax.axhline(ceiling, color="#444444", ls="--", lw=1.3,
                       label=f"Feature-based ceiling ({ceiling:.1%})")
            hi = max(hi, ceiling)
        ax.set_ylim(lo - 0.008, hi + 0.008)
        ax.set_xticks(x)
        # Hybrids are labelled "Hybrid A".."Hybrid D"; their members are listed
        # in the deck and README.
        ax.set_xticklabels([n.split(" (")[0] if n.startswith("Hybrid") else n
                            for n in names], rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{title}  -  majority baseline {s['majority_baseline']:.1%}",
                     fontsize=12)
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9.5)
    fig.suptitle("Model comparison on the validation split (ordered by macro F1)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, y_pred, labels, title, path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(disp, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar=False, ax=ax, vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(model, names, path, top_n=20, title=None):
    """Permutation-free importance, read straight off the fitted tree model."""
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    est = getattr(est, "model_", est)   # unwrap XGBWrapper
    if not hasattr(est, "feature_importances_"):
        return None
    imp = pd.Series(est.feature_importances_, index=names).sort_values()[-top_n:]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(imp) + 1.8))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(title or f"Top {len(imp)} features", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return imp.sort_values(ascending=False).to_dict()


def plot_shap(model, X_sample, names, out_dir, model_name="", focus_class="Dispose"):
    """
    SHAP explanations: a global importance view plus one per-part waterfall.

    For a decision-support system this is not decoration. A dismantler needs to
    know WHY a part was flagged for disposal before acting on it, and a single
    global importance ranking cannot answer that for an individual part.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping explainability plots")
        return None

    pre = model.named_steps["prep"]
    est = model.named_steps["clf"]
    est = getattr(est, "model_", est)   # unwrap XGBWrapper for TreeExplainer
    Xt = pre.transform(X_sample)
    Xt_df = pd.DataFrame(Xt, columns=names)

    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer(Xt_df, check_additivity=False)
    except Exception as exc:
        print(f"  SHAP unavailable for this model: {exc}")
        return None

    # classes_ is sorted alphabetically (Dispose, Recycle, Reuse), so class
    # indices are looked up by name rather than assumed.
    classes = [str(c) for c in getattr(est, "classes_", [])]
    suffix = f" - {model_name}" if model_name else ""

    plt.figure()
    shap.summary_plot(sv, Xt_df, plot_type="bar", max_display=18, show=False,
                      class_names=classes or None)
    plt.title(f"SHAP global feature importance{suffix}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_global_importance.png"), dpi=150)
    plt.close()

    # Per-part explanation on the Dispose class, since that is the decision a
    # user most needs justified. The part shown is the first test sample the
    # model actually predicts as Dispose. Feature values are standardised
    # (numeric) or ordinal codes, as the model sees them.
    try:
        cls = classes.index(focus_class) if focus_class in classes else 0
        pred = np.asarray(est.predict(Xt)).ravel()
        hits = np.flatnonzero(pred.astype(str) == classes[cls]) if classes else []
        idx = int(hits[0]) if len(hits) else 0
        single = sv[idx]
        if single.values.ndim > 1:
            single = single[..., cls]
        plt.figure()
        shap.plots.waterfall(single, max_display=14, show=False)
        plt.gcf().suptitle(f"Why one part was classed "
                           f"'{classes[cls] if classes else cls}'{suffix}",
                           fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_single_part.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"  per-part SHAP waterfall skipped: {exc}")

    return "results/shap_global_importance.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_deep_models(spec_registry, pre, splits, labels, verbose=True):
    """
    Train the neural architectures with an explicit epoch loop.

    For each architecture the LEARNING RATE is searched over its declared grid.
    Every candidate trains against the real validation split, early stopping on
    validation loss and restoring the best-epoch weights, and the winner is the
    learning rate reaching the lowest validation loss. The full search is
    returned so the report can show the learning-rate sensitivity rather than
    only the chosen value.
    """
    A = pre.transform(splits["X_train"])
    B = pre.transform(splits["X_val"])
    y_tr, y_val = splits["y_train"], splits["y_val"]

    trained, searches = {}, {}
    for name, spec in spec_registry.items():
        spec = dict(spec)
        lr_grid = spec.pop("lr_grid")
        note = spec.pop("note", "")
        if verbose:
            print(f"\n[{name}] {note}")
            print(f"  searching learning rates {lr_grid} "
                  f"over up to {spec['epochs']} epochs")

        best, best_loss, search = None, float("inf"), []
        for lr in lr_grid:
            model = TorchTabularClassifier(lr=lr, random_state=RANDOM_STATE,
                                           **spec)
            model.fit(A, y_tr, eval_set=(B, y_val))
            search.append({"lr": lr, "best_epoch": model.best_epoch_,
                           "val_loss": model.best_val_loss_,
                           "val_acc": model.best_val_acc_,
                           "epochs_run": len(model.history_["epoch"])})
            if verbose:
                print(f"    lr={lr:<8g} best_epoch={model.best_epoch_:3d}  "
                      f"val_loss={model.best_val_loss_:.4f}  "
                      f"val_acc={model.best_val_acc_:.4f}")
            if model.best_val_loss_ < best_loss:
                best_loss, best = model.best_val_loss_, model

        trained[name] = best
        searches[name] = {"grid": search, "chosen_lr": best.lr,
                          "activation": best.activation,
                          "best_epoch": best.best_epoch_,
                          "architecture": note}
        if verbose:
            print(f"  -> chose lr={best.lr}, best epoch {best.best_epoch_}, "
                  f"val_loss={best.best_val_loss_:.4f}")
    return trained, searches


def run_target(df, target, labels, out_dir, do_hybrids=True, do_shap=True,
               tag="disposition", deep_epochs=80):
    """Train, tune and compare every model for one target, and emit artefacts."""
    print(f"\n{'='*74}\nTARGET: {target}\n{'='*74}")
    splits = make_splits(df, target)
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    cost_matrix = COST_MATRICES[target]

    baseline = float(y_tr.value_counts(normalize=True).max())
    total = sum(len(splits[k]) for k in ("X_train", "X_val", "X_test"))
    print(f"split  train={len(X_tr)} ({len(X_tr)/total:.0%})  "
          f"test={len(splits['X_test'])} ({len(splits['X_test'])/total:.0%})  "
          f"validation={len(splits['X_val'])} ({len(splits['X_val'])/total:.0%})")
    print(f"       extrapolation set={len(splits['X_extrap'])} (held out entirely)")
    print(f"majority-class baseline: {baseline:.4f}")

    pre = build_preprocessor(X_tr)
    pre_fitted = sklearn.base.clone(pre).fit(X_tr, y_tr)

    results, tuned, curves = {}, {}, {}

    # --- classical models, tuned by grid search --------------------------
    for name, (estimator, grid) in model_registry(len(labels)).items():
        n_cfg = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n[{name}] tuning {n_cfg} configuration(s)...")
        pipe = Pipeline([("prep", pre), ("clf", estimator)])
        search = GridSearchCV(
            pipe, {f"clf__{k}": v for k, v in grid.items()},
            cv=3, scoring="f1_macro", n_jobs=1, refit=True)
        # Class balancing happens inside each estimator: class_weight for the
        # sklearn models, auto_class_weights for CatBoost, balanced sample
        # weights inside XGBWrapper.
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        tuned[name] = best.named_steps["clf"]

        val_m, _ = evaluate(best, splits["X_val"], splits["y_val"], labels,
                            cost_matrix)
        results[name] = {
            "best_params": {k.replace("clf__", ""): v
                            for k, v in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_),
            "validation": val_m,
            "model": best,
        }
        print(f"  val acc={val_m['accuracy']:.4f}  macroF1={val_m['f1_macro']:.4f}  "
              f"val loss={val_m.get('log_loss', float('nan')):.4f}  "
              f"cost={val_m['expected_cost']:.3f}")

        curves[name] = compute_training_curve(
            name, tuned[name], pre_fitted, X_tr, y_tr,
            splits["X_val"], splits["y_val"], labels)

    # --- deep models, trained with explicit epochs -----------------------
    deep_models, deep_searches = train_deep_models(
        deep_registry(epochs=deep_epochs), pre_fitted, splits, labels)

    for name, model in deep_models.items():
        pipe = Pipeline([("prep", pre_fitted), ("clf", model)])
        val_m, _ = evaluate(pipe, splits["X_val"], splits["y_val"], labels,
                            cost_matrix)
        results[name] = {
            "best_params": {"learning_rate": model.lr,
                            "activation": model.activation,
                            "epochs_trained": len(model.history_["epoch"]),
                            "best_epoch": model.best_epoch_,
                            "dropout": model.dropout},
            "cv_f1_macro": None,
            "lr_search": deep_searches[name],
            "validation": val_m,
            "model": pipe,
        }
        print(f"  [{name}] val acc={val_m['accuracy']:.4f}  "
              f"macroF1={val_m['f1_macro']:.4f}  "
              f"val loss={val_m.get('log_loss', float('nan')):.4f}")
        curves[name] = compute_training_curve(
            name, model, pre_fitted, X_tr, y_tr,
            splits["X_val"], splits["y_val"], labels)
        plot_single_curve(curves[name], name,
                          os.path.join(out_dir,
                                       f"{tag}_curve_{name.split()[0].lower()}.png"))

    # --- hybrid ensembles -------------------------------------------------
    if do_hybrids:
        for name, hybrid in build_hybrids(tuned).items():
            print(f"\n[{name}] fitting...")
            pipe = Pipeline([("prep", pre), ("clf", hybrid)])
            pipe.fit(X_tr, y_tr)
            val_m, _ = evaluate(pipe, splits["X_val"], splits["y_val"], labels,
                                cost_matrix)
            results[name] = {
                "best_params": "ensemble over tuned base learners",
                "cv_f1_macro": None,
                "validation": val_m,
                "model": pipe,
            }
            print(f"  val acc={val_m['accuracy']:.4f}  macroF1={val_m['f1_macro']:.4f}  "
                  f"val loss={val_m.get('log_loss', float('nan')):.4f}  "
                  f"cost={val_m['expected_cost']:.3f}")

    # --- Select on macro F1, not accuracy ---------------------------------
    # Accuracy rewards a model that simply predicts the majority class well.
    # Macro F1 forces the minority Dispose class to be handled, which is the
    # class that matters environmentally.
    best_name = max(results, key=lambda n: results[n]["validation"]["f1_macro"])
    best_model = results[best_name]["model"]
    print(f"\nBest model by validation macro F1: {best_name}")

    lowest_loss = min((n for n in results
                       if "log_loss" in results[n]["validation"]),
                      key=lambda n: results[n]["validation"]["log_loss"])
    print(f"Lowest validation loss: {lowest_loss} "
          f"({results[lowest_loss]['validation']['log_loss']:.4f})")

    # --- Final evaluation --------------------------------------------------
    test_m, test_pred = evaluate(best_model, splits["X_test"], splits["y_test"], labels, cost_matrix)
    extrap_m, extrap_pred = evaluate(best_model, splits["X_extrap"], splits["y_extrap"], labels, cost_matrix)
    print(f"  TEST          acc={test_m['accuracy']:.4f}  macroF1={test_m['f1_macro']:.4f}  "
          f"loss={test_m.get('log_loss', float('nan')):.4f}  cost={test_m['expected_cost']:.3f}")
    print(f"  EXTRAPOLATION acc={extrap_m['accuracy']:.4f}  macroF1={extrap_m['f1_macro']:.4f}  "
          f"cost={extrap_m['expected_cost']:.3f}   "
          f"(unseen: {HOLDOUT_MATERIAL} / {HOLDOUT_COMPONENT})")
    print("\n" + classification_report(splits["y_test"], test_pred,
                                       labels=labels, zero_division=0))

    # --- Artefacts ---------------------------------------------------------
    # The feature-based ceiling, not the full-latent oracle, is the fair
    # reference: it is what a model restricted to the exported features can
    # attain once the unobserved shocks are marginalised out.
    ceiling = (sim.estimate_feature_ceiling(n_samples=6000)
               if target == TARGET_MAIN else float("nan"))
    plot_model_comparison(results, ceiling if ceiling == ceiling else 1.0,
                          baseline, os.path.join(out_dir, f"{tag}_model_comparison.png"))
    plot_confusion(splits["y_test"], test_pred, labels,
                   f"{best_name} - test set ({target})",
                   os.path.join(out_dir, f"{tag}_confusion_matrix.png"))
    plot_confusion(splits["y_extrap"], extrap_pred, labels,
                   f"{best_name} - unseen material/component",
                   os.path.join(out_dir, f"{tag}_confusion_extrapolation.png"))
    plot_training_curves(curves, os.path.join(out_dir, f"{tag}_training_curves.png"),
                         f"Validation loss and accuracy per epoch - {target}")

    names = feature_names(pre_fitted, X_tr)

    # Importance and SHAP both need a tree model. If the overall winner is not
    # one (an SVM, LSTM or Transformer can win here), explain the best TREE
    # model instead and record which model the explanation describes, so the
    # plots are never silently attributed to the wrong estimator.
    tree_models = ["Random Forest", "Extra Trees", "CatBoost", "LightGBM", "XGBoost"]
    explain_name = best_name if best_name in tree_models else max(
        (n for n in tree_models if n in results),
        key=lambda n: results[n]["validation"]["f1_macro"])
    explain_model = results[explain_name]["model"]
    if explain_name != best_name:
        print(f"  explainability plots use {explain_name} "
              f"(best model {best_name} is not tree-based)")

    importance = plot_importance(explain_model, names,
                                 os.path.join(out_dir, f"{tag}_feature_importance.png"),
                                 title=f"Top 20 features - {explain_name} ({target})")

    if do_shap:
        plot_shap(explain_model, splits["X_test"].head(400), names, out_dir,
                  model_name=explain_name)

    summary = {
        "target": target,
        "labels": labels,
        "majority_baseline": baseline,
        "cost_matrix": cost_matrix.tolist(),
        "feature_based_ceiling": None if ceiling != ceiling else ceiling,
        "split_sizes": {k: int(len(v)) for k, v in splits.items() if k.startswith("X_")},
        "split_proportions": {"train": TRAIN_FRACTION, "test": TEST_FRACTION,
                              "validation": VAL_FRACTION},
        "extrapolation_holdout": {"material": HOLDOUT_MATERIAL,
                                  "component": HOLDOUT_COMPONENT},
        "models": {n: {k: v for k, v in r.items() if k != "model"}
                   for n, r in results.items()},
        "training_curves": {n: c for n, c in curves.items() if c},
        "best_model": best_name,
        "lowest_validation_loss_model": lowest_loss,
        "explained_model": explain_name,
        "best_model_test": test_m,
        "best_model_extrapolation": extrap_m,
        "top_features": (dict(list(importance.items())[:15]) if importance else None),
        "classification_report": classification_report(
            splits["y_test"], test_pred, labels=labels, zero_division=0,
            output_dict=True),
    }
    return summary, best_model, splits


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Dataset not found at {DATA_PATH}. "
                         "Run: python3 -m src.lifecycle.make_dataset")

    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}  shape={df_raw.shape}")

    # Evaluator instruction: remove missing data rather than impute it.
    missing_report = report_missing_strategies(df_raw)
    df = apply_missing_policy(df_raw)
    print("  cost of each policy, for comparison:")
    for r in missing_report:
        print(f"    {r['strategy']:<13} -> {r['rows']} rows x {r['columns']} cols "
              f"({r['rows_lost_pct']}% of rows lost)")

    df = engineer_features(df)
    print(f"After feature engineering: {df.shape[1]} columns")

    disp_summary, disp_model, _ = run_target(
        df, TARGET_MAIN, sim.DISPOSITIONS, RESULTS_DIR,
        do_hybrids=True, do_shap=True, tag="disposition")

    # Second head required by the synopsis: environmental risk level. It gets
    # the same full treatment as the primary target - every algorithm, the
    # hybrids, and epoch-wise curves - so the two are directly comparable.
    risk_summary, risk_model, _ = run_target(
        df, TARGET_RISK, sim.RISK_TIERS, RESULTS_DIR,
        do_hybrids=True, do_shap=False, tag="risk")

    out = {
        "dataset": os.path.relpath(DATA_PATH, ROOT),
        "dataset_rows_raw": int(len(df_raw)),
        "dataset_rows_used": int(len(df)),
        "missing_value_policy": MISSING_STRATEGY,
        "missing_value_policy_comparison": missing_report,
        "split": {"train": TRAIN_FRACTION, "test": TEST_FRACTION,
                  "validation": VAL_FRACTION},
        "activation_functions": {
            "MLP Neural Network": "ReLU (hidden), Softmax (output)",
            "LSTM": ("Sigmoid + Tanh (cell gates, intrinsic), Tanh (head), "
                     "Softmax (output)"),
            "Transformer": "GELU (feed-forward block), Softmax (output)",
        },
        "note": ("Dataset is simulation-based (see data/dataset_card.md). "
                 "Accuracy measures fidelity to a documented LCA decision "
                 "rule under partial observation, not field disposition "
                 "accuracy. Compare every score against "
                 "feature_based_ceiling, not against 1.0."),
        "cost_matrix": {"order": sim.DISPOSITIONS, "matrix": COST_MATRIX.tolist()},
        "disposition": disp_summary,
        "environmental_risk": risk_summary,
    }
    path = os.path.join(RESULTS_DIR, "results_summary_v2.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nWrote {path}")
    plot_model_comparison_lines(out, os.path.join(RESULTS_DIR,
                                                  "model_comparison_line.png"))

    try:
        import joblib
        models_dir = os.path.join(ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(disp_model, os.path.join(models_dir, "disposition_model.joblib"))
        joblib.dump(risk_model, os.path.join(models_dir, "risk_model.joblib"))
        print(f"Saved fitted models to {models_dir}/")
    except Exception as exc:
        print(f"Model persistence skipped: {exc}")

    return out


if __name__ == "__main__":
    main()
