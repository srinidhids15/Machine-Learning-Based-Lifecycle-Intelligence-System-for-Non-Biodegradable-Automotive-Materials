# Phase 2 Submission — Team 53

**Machine Learning Based Lifecycle Intelligence System for Non-Biodegradable Automotive Materials**

Prajwal B T (NNM23IS128) · Srinidhi D S (NNM23IS185) · Swaroop Sathish Y (NNM23IS191) · V K Ronith (NNM23IS203)
Guide: Pratheeksha Hegde N · Dept. of Information Science & Engineering, NMAMIT Nitte

---

## Problem Statement

> End-of-life vehicles accumulate faster than current waste-management practice
> can assess them, and there is no intelligent tool that decides whether a
> non-biodegradable automotive material should be **reused, recycled, or
> disposed of** — or how much **environmental risk** it carries if mishandled.

---

## Headline results

| Target | Baseline | Best model | Test accuracy | Val loss |
|---|---|---|---|---|
| Disposition (Reuse/Recycle/Dispose) | 53.4% | Transformer | **88.67%** | 0.3066 |
| Environmental Risk (4 tiers) | 34.3% | Extra Trees | **85.94%** | 0.3219 |

**13 models trained per target.** Dataset split
**70% train (5007) / 20% test (1430) /
10% validation (716)**.

### Read the accuracy against the ceiling, not against 100%

The data generator's irreducible error is measured, not guessed. A model that
sees only the exported features cannot exceed **89.1%** accuracy on
Disposition, because the labels also depend on factors no database records.
The best model reaches **88.67%**, which is
at that limit. The remaining error is irreducible, not a tuning failure.

---

## What is in this folder

| Folder | Contents |
|---|---|
| `code/` | The five source files: property table, simulator, dataset builder, deep models, training pipeline |
| `data/` | The dataset and its dataset card (provenance, method, caveats) |
| `results/` | All charts, confusion matrices, training curves, SHAP plots, and the full metrics JSON |
| `models/` | The two fitted models, ready to load with `joblib` |

| Document | Purpose |
|---|---|
| `EVALUATOR_CHECKLIST.md` | **Start here.** Each requirement mapped to the file and result that implements it |
| `RESULTS.md` | Full per-model tables, both targets, with per-class breakdowns |
| `METHOD.md` | How the data is generated, why, and the honest limitations |

---

## Honest statement about the data

The dataset is **simulated**, which the project synopsis permits
("simulation-based lifecycle data"). It is generated from published material
properties (Ashby; Argonne GREET 2023; EU ELV Directive 2000/53/EC; US EPA
WARM) with an explicit degradation model, and labelled by a documented
multi-criteria decision rule. The models therefore learn a **surrogate of that
rule**, and the reported accuracy measures surrogate fidelity rather than
field disposition accuracy. `METHOD.md` and `data/dataset_card.md` set out the
full reasoning, and the previous dataset's failure that made this necessary.
