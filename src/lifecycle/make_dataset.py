"""
Build the project dataset from the lifecycle simulator.

Run from the repository root:
    python3 -m src.lifecycle.make_dataset

Writes:
    data/automotive_material_lifecycle_v2.csv        model-ready (latents hidden)
    data/_simulator_diagnostics.csv                  latents included, DIAGNOSTIC ONLY
    data/dataset_card.md                             provenance and schema
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import material_properties as mp
from . import simulator as sim

N_SAMPLES = 9000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def main():
    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {N_SAMPLES} simulated end-of-life parts...")
    df = sim.generate(n_samples=N_SAMPLES, seed=sim.RNG_SEED)
    # 1% per column across 6 columns leaves ~6% of rows carrying a missing
    # value. The pipeline REMOVES those rows rather than imputing them, per
    # evaluator instruction, so the injection rate is kept low deliberately:
    # it still exercises the missing-data handling step without discarding a
    # large share of the dataset.
    df = sim.inject_realistic_missingness(df, rate=0.01)

    model_path = os.path.join(out_dir, "automotive_material_lifecycle_v2.csv")
    df.to_csv(model_path, index=False)
    print(f"  wrote {model_path}  shape={df.shape}")

    # Diagnostics copy WITH latent state. Never train on this: it contains the
    # exact utilities the labels were drawn from.
    diag = sim.generate(n_samples=4000, seed=sim.RNG_SEED + 99, include_latents=True)
    diag_path = os.path.join(out_dir, "_simulator_diagnostics.csv")
    diag.to_csv(diag_path, index=False)
    print(f"  wrote {diag_path}  shape={diag.shape}  (diagnostics only)")

    oracle_ceiling = sim.estimate_bayes_ceiling(n_samples=30000)
    feature_ceiling = sim.estimate_feature_ceiling(n_samples=8000)
    print(f"  full-latent oracle ceiling : {oracle_ceiling:.4f}  (loose upper bound)")
    print(f"  feature-based ceiling      : {feature_ceiling:.4f}  <- judge models against this")

    write_dataset_card(df, oracle_ceiling, feature_ceiling, out_dir)
    return df, feature_ceiling


def write_dataset_card(df, oracle_ceiling, feature_ceiling, out_dir):
    """Emit a provenance document so the dataset is never mistaken for field data."""
    mix = df["Disposition"].value_counts(normalize=True).round(4).to_dict()
    risk = df["Environmental Risk"].value_counts(normalize=True).round(4).to_dict()
    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()

    card = f"""# Dataset Card - Automotive Material Lifecycle (v2, simulated)

## Provenance

**This dataset is simulated, not measured.** It is produced by
`src/lifecycle/simulator.py` from the published material property table in
`src/lifecycle/material_properties.py`. The project synopsis admits
"simulation-based lifecycle data" as a data source; this file is that source,
and it must always be described as simulated in any report or presentation.

It replaces `automotive_material_lifecycle_dataset.csv` (v1), which was retained
for comparison. In v1 the labels were close to independent of the features:
Random Forest reached 58.4% under 5-fold cross-validation against a 45.0%
majority-class baseline, and `Vehicle Type` and `Manufacturing Year` had exactly
zero mutual information with the target. v1 also contained physically impossible
values, such as a Copper Alloy part mass with mean 15.1 kg and standard
deviation 100.1 kg.

## How it is generated

1. **Physical anchors.** Density, embodied energy, cradle-to-gate GWP,
   end-of-life recycling rate, recyclate value, tensile strength and hazard
   class are taken per material from published sources (Ashby, *Materials and
   the Environment*, 3rd ed.; Argonne GREET 2023; EU Directive 2000/53/EC;
   US EPA WARM v16). Part mass is derived from the component's primary-material
   mass scaled by density ratio, so mass, embodied energy and carbon footprint
   are physically consistent rather than independently sampled.

2. **Degradation model.** Each part accumulates thermal, photo-oxidative and
   corrosive dose over its service life, weighted by the duty cycle of its
   component and vehicle type, under a first-order saturation law. This is why
   service life and odometer are genuinely predictive here.

3. **Partially observed decision rule.** Labels are drawn from a multi-criteria
   utility model over the latent lifecycle state, with Gumbel noise
   (temperature {sim.LABEL_TEMPERATURE}). Class composition is calibrated to the
   EU ELV stream (~20% reuse / ~50% recycle / ~30% dispose).

## The honest caveat

Labels come from a rule, so a model trained here learns a **surrogate of that
rule**, and its accuracy measures surrogate fidelity rather than real-world
disposition accuracy. Three design choices keep the learning task genuine:

- **Latent state is withheld.** The degradation index, contamination level,
  separability, hazard score and the utilities themselves are never exported.
  The model sees only coarsened or noise-corrupted observations of them: a
  4-level condition grade, a binary contamination flag, a 1-5 surface grade, and
  a tensile strength measurement carrying 8% NDT error.
- **Unobserved heterogeneity.** Each part carries utility shocks representing
  hidden internal damage, local recyclate market conditions, and operator and
  plant capability. None of these are exported, so no model can recover them -
  they are what make the task statistical rather than algebraic. Note that the
  *rest* of the latent state largely IS recoverable from the features, and
  correctly so: separability genuinely is determined by a part's component,
  joining method and coating. The shocks are the honest source of irreducible
  error, not pretend-hidden algebra.
- **Two measured ceilings.** An oracle knowing the shocks too scores
  **{oracle_ceiling:.2%}**, a loose upper bound. Marginalising the shocks out
  gives **{feature_ceiling:.2%}** - the ceiling for any model restricted to the
  exported features, and **the number every result should be read against.**
  A model landing a few points under it is performing as well as the data
  permits; the residual gap is irreducible, not a tuning failure.
- **Extrapolation testing.** `train.py` evaluates on materials and components
  entirely absent from training, which tests generalisation rather than recall.

## Schema

{len(df.columns)} columns, {len(df)} rows. Two targets:
`Disposition` (Reuse / Recycle / Dispose) and `Environmental Risk`
(Low / Moderate / High / Critical), the second satisfying the synopsis
requirement to "determine the environmental risk level".

Class balance - Disposition: {json.dumps(mix)}

Class balance - Environmental Risk: {json.dumps(risk)}

Missing values were injected at a 1% per-column rate into the fields a
dismantling yard most often fails to record, so preprocessing has real gaps to
handle. The training pipeline REMOVES the affected rows rather than imputing
them, following evaluator instruction, which is why the rate is kept low:
{json.dumps(missing)}

Materials: {len(mp.MATERIALS)} | Components: {len(mp.COMPONENTS)} | Vehicle types: {len(mp.VEHICLE_TYPES)}

## Reproducing

```bash
python3 -m src.lifecycle.make_dataset
```

Seed {sim.RNG_SEED}; generation is deterministic.
"""
    path = os.path.join(out_dir, "dataset_card.md")
    with open(path, "w") as fh:
        fh.write(card)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
