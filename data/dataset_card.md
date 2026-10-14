# Dataset Card - Automotive Material Lifecycle (v2, simulated)

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
   (temperature 5.5). Class composition is calibrated to the
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
  **92.49%**, a loose upper bound. Marginalising the shocks out
  gives **89.07%** - the ceiling for any model restricted to the
  exported features, and **the number every result should be read against.**
  A model landing a few points under it is performing as well as the data
  permits; the residual gap is irreducible, not a tuning failure.
- **Extrapolation testing.** `train.py` evaluates on materials and components
  entirely absent from training, which tests generalisation rather than recall.

## Schema

30 columns, 9000 rows. Two targets:
`Disposition` (Reuse / Recycle / Dispose) and `Environmental Risk`
(Low / Moderate / High / Critical), the second satisfying the synopsis
requirement to "determine the environmental risk level".

Class balance - Disposition: {"Recycle": 0.5041, "Dispose": 0.295, "Reuse": 0.2009}

Class balance - Environmental Risk: {"Moderate": 0.324, "Critical": 0.284, "Low": 0.2232, "High": 0.1688}

Missing values were injected at a 1% per-column rate into the fields a
dismantling yard most often fails to record, so preprocessing has real gaps to
handle. The training pipeline REMOVES the affected rows rather than imputing
them, following evaluator instruction, which is why the rate is kept low:
{"Odometer (km)": 79, "Measured Tensile Strength (MPa)": 98, "Recycling Efficiency (%)": 109, "Recycled Content (%)": 90, "Surface Degradation Grade": 100, "Recyclate Value ($)": 95}

Materials: 12 | Components: 12 | Vehicle types: 3

## Reproducing

```bash
python3 -m src.lifecycle.make_dataset
```

Seed 42; generation is deterministic.
