"""
Hyperparameter-tuning and activation-function study.

Answers two review questions with measurements rather than assertions:

  1. Do the classical models need more tuning? The original GridSearchCV grids
     in train.py are small, and several winners sit on a grid EDGE (SVM C=3,
     XGBoost learning_rate=0.06 and max_depth=5 are all the smallest values
     tried). Each model is re-tuned on a wider grid that extends past those
     edges, and compared with its original grid under identical splits,
     preprocessing, scoring (3-fold CV, macro F1) and seeds.
  2. Is the chosen activation function the right one? Each neural network is
     trained with every applicable activation, each at every learning rate in
     its original grid, and the activation reaching the lowest validation loss
     is identified - the same selection rule train.py already uses for the
     learning rate.

Selection uses only cross-validation (classical) or the validation split
(neural). The test split is scored for reference but never used to choose.

Run from the repository root:
    python3 -m src.lifecycle.tuning_study
Writes results/tuning_study.json, results/tuning_comparison.png and
results/activation_comparison.png.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from . import simulator as sim
from . import train as T
from .deep_models import TorchTabularClassifier

# Wider grids. Each keeps the original winner inside the grid and extends past
# the edge(s) it was found on, plus the regularisation parameter the original
# grid left at its library default.
WIDE_GRIDS = {
    "Random Forest": {"n_estimators": [300, 600], "max_depth": [None, 10, 14, 20],
                      "min_samples_leaf": [1, 3, 5],
                      "max_features": ["sqrt", 0.3]},
    "Extra Trees": {"n_estimators": [300, 600], "max_depth": [None, 16, 24],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["sqrt", 0.5]},
    "XGBoost": {"n_estimators": [400], "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.03, 0.06], "subsample": [0.8, 0.9],
                "reg_lambda": [1.0, 5.0]},
    "LightGBM": {"n_estimators": [300, 600], "num_leaves": [7, 15, 31],
                 "learning_rate": [0.03, 0.05], "min_child_samples": [20, 40, 80],
                 "reg_lambda": [5.0, 10.0]},
    "CatBoost": {"iterations": [500], "depth": [4, 6, 8],
                 "learning_rate": [0.04, 0.08], "l2_leaf_reg": [3.0, 10.0]},
    "SVM (RBF)": {"C": [0.3, 1.0, 3.0, 10.0], "gamma": ["scale", 0.01, 0.03]},
}

# Activations applicable to each architecture. The LSTM's internal Sigmoid/Tanh
# gates are part of the cell definition, so only its head activation varies;
# PyTorch's Transformer feed-forward block supports only ReLU and GELU.
ACTIVATION_GRIDS = {
    "MLP Neural Network": ["relu", "gelu", "tanh", "leaky_relu"],
    "LSTM": ["tanh", "relu", "gelu"],
    "Transformer": ["gelu", "relu"],
}


def _grid_search(name, estimator, grid, pre, splits, labels, cost):
    pipe = Pipeline([("prep", pre), ("clf", estimator)])
    search = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()},
                          cv=3, scoring="f1_macro", n_jobs=-1, refit=True)
    t0 = time.time()
    search.fit(splits["X_train"], splits["y_train"])
    best = search.best_estimator_
    val, _ = T.evaluate(best, splits["X_val"], splits["y_val"], labels, cost)
    test, _ = T.evaluate(best, splits["X_test"], splits["y_test"], labels, cost)
    return {"n_configs": len(search.cv_results_["params"]),
            "best_params": {k.replace("clf__", ""): v
                            for k, v in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_),
            "validation": val, "test": test,
            "seconds": round(time.time() - t0, 1)}


def classical_study(splits, labels, cost):
    pre = T.build_preprocessor(splits["X_train"])
    out = {}
    for name, (estimator, grid) in T.model_registry(len(labels)).items():
        print(f"\n[{name}]")
        orig = _grid_search(name, sklearn.base.clone(estimator), grid, pre,
                            splits, labels, cost)
        print(f"  original {orig['n_configs']:3d} cfg  cv={orig['cv_f1_macro']:.4f}  "
              f"val acc={orig['validation']['accuracy']:.4f} "
              f"f1={orig['validation']['f1_macro']:.4f} "
              f"loss={orig['validation'].get('log_loss', float('nan')):.4f}")
        wide = _grid_search(name, sklearn.base.clone(estimator), WIDE_GRIDS[name],
                            pre, splits, labels, cost)
        print(f"  wide     {wide['n_configs']:3d} cfg  cv={wide['cv_f1_macro']:.4f}  "
              f"val acc={wide['validation']['accuracy']:.4f} "
              f"f1={wide['validation']['f1_macro']:.4f} "
              f"loss={wide['validation'].get('log_loss', float('nan')):.4f}  "
              f"{wide['best_params']}")
        out[name] = {"original": orig, "wide": wide}
    return out


def activation_study(splits, labels, cost):
    pre = sklearn.base.clone(T.build_preprocessor(splits["X_train"])).fit(
        splits["X_train"], splits["y_train"])
    A, B = pre.transform(splits["X_train"]), pre.transform(splits["X_val"])
    y_tr, y_val = splits["y_train"], splits["y_val"]

    out = {}
    for name, spec in T.deep_registry(epochs=80).items():
        spec = dict(spec)
        lr_grid = spec.pop("lr_grid")
        spec.pop("note", None)
        default_act = spec.pop("activation")
        print(f"\n[{name}] default activation={default_act}")
        rows = []
        for act in ACTIVATION_GRIDS[name]:
            best, best_loss, lr_rows = None, float("inf"), []
            for lr in lr_grid:
                m = TorchTabularClassifier(lr=lr, activation=act,
                                           random_state=T.RANDOM_STATE, **spec)
                m.fit(A, y_tr, eval_set=(B, y_val))
                lr_rows.append({"lr": lr, "val_loss": m.best_val_loss_,
                                "best_epoch": m.best_epoch_})
                if m.best_val_loss_ < best_loss:
                    best_loss, best = m.best_val_loss_, m
            pipe = Pipeline([("prep", pre), ("clf", best)])
            val, _ = T.evaluate(pipe, splits["X_val"], splits["y_val"], labels, cost)
            test, _ = T.evaluate(pipe, splits["X_test"], splits["y_test"], labels, cost)
            rows.append({"activation": act, "chosen_lr": best.lr,
                         "best_epoch": best.best_epoch_, "lr_grid": lr_rows,
                         "validation": val, "test": test})
            print(f"  {act:<11} lr={best.lr:<7g} epoch={best.best_epoch_:3d}  "
                  f"val acc={val['accuracy']:.4f} f1={val['f1_macro']:.4f} "
                  f"loss={val['log_loss']:.4f}")
        winner = min(rows, key=lambda r: r["validation"]["log_loss"])["activation"]
        out[name] = {"default": default_act, "selected": winner, "runs": rows}
        print(f"  -> lowest validation loss: {winner}")
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
TARGET_TITLES = {"disposition": "Disposition", "environmental_risk": "Environmental Risk"}


def plot_tuning(study, path):
    """Original vs wider-grid validation macro F1, one line per grid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, (tag, res) in zip(axes, study.items()):
        names = list(res["classical"])
        x = np.arange(len(names))
        o = [res["classical"][n]["original"]["validation"]["f1_macro"] for n in names]
        w = [res["classical"][n]["wide"]["validation"]["f1_macro"] for n in names]
        ax.plot(x, o, "-o", color="#8ab4d8", lw=2, ms=7, label="Original grid")
        ax.plot(x, w, "-s", color="#3b6ea5", lw=2, ms=7, label="Wider grid")
        for xi, a, b in zip(x, o, w):
            ax.annotate(f"{b - a:+.3f}", (xi, max(a, b)), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Validation macro F1", fontsize=11)
        lo, hi = min(o + w), max(o + w)
        ax.set_ylim(lo - 0.01, hi + 0.012)
        ax.set_title(TARGET_TITLES[tag], fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
    fig.suptitle("Hyperparameter tuning: original vs wider grid (validation split)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_activations(study, path):
    """Validation loss and macro F1 for each activation, per architecture."""
    archs = list(ACTIVATION_GRIDS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.8 / max(len(v) for v in ACTIVATION_GRIDS.values())
    colours = {"relu": "#3b6ea5", "gelu": "#2e7d32", "tanh": "#b3261e",
               "leaky_relu": "#e09f3e"}
    for ax, (tag, res) in zip(axes, study.items()):
        seen = set()
        for i, arch in enumerate(archs):
            runs = res["activation"][arch]["runs"]
            offs = (np.arange(len(runs)) - (len(runs) - 1) / 2) * width
            for off, r in zip(offs, runs):
                lab = r["activation"] if r["activation"] not in seen else None
                seen.add(r["activation"])
                ax.bar(i + off, r["validation"]["log_loss"], width * 0.92,
                       color=colours[r["activation"]], label=lab)
                if r["activation"] == res["activation"][arch]["default"]:
                    ax.text(i + off, r["validation"]["log_loss"], "used",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(archs))); ax.set_xticklabels(archs, fontsize=10)
        losses = [r["validation"]["log_loss"] for a in archs
                  for r in res["activation"][a]["runs"]]
        ax.set_ylim(min(losses) - 0.02, max(losses) + 0.02)
        ax.set_ylabel("Validation loss (lower is better)", fontsize=11)
        ax.set_title(TARGET_TITLES[tag], fontsize=12)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.suptitle("Activation function comparison - best learning rate per activation",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    df = T.engineer_features(T.apply_missing_policy(pd.read_csv(T.DATA_PATH)))
    study = {}
    for tag, target, labels in (("disposition", T.TARGET_MAIN, sim.DISPOSITIONS),
                                ("environmental_risk", T.TARGET_RISK, sim.RISK_TIERS)):
        print(f"\n{'=' * 74}\nTARGET: {target}\n{'=' * 74}")
        splits = T.make_splits(df, target)
        cost = T.COST_MATRICES[target]
        study[tag] = {"classical": classical_study(splits, labels, cost),
                      "activation": activation_study(splits, labels, cost)}

    with open(os.path.join(T.RESULTS_DIR, "tuning_study.json"), "w") as fh:
        json.dump({"wide_grids": WIDE_GRIDS, "activation_grids": ACTIVATION_GRIDS,
                   **study}, fh, indent=2, default=str)
    plot_tuning(study, os.path.join(T.RESULTS_DIR, "tuning_comparison.png"))
    plot_activations(study, os.path.join(T.RESULTS_DIR, "activation_comparison.png"))
    print("\nWrote results/tuning_study.json, tuning_comparison.png, "
          "activation_comparison.png")


if __name__ == "__main__":
    main()
