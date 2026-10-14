# Method and Limitations

## Why the dataset was rebuilt

The Phase 1 dataset could not support the project's objectives. Measured
independently of the training pipeline:

| Check | Result |
|---|---|
| Random Forest, 5-fold cross-validated accuracy | 58.4% |
| Majority-class baseline | 45.0% |
| Highest mutual information (Toxicity Index) | 0.14 |
| Mutual information of `Vehicle Type`, `Manufacturing Year` | **0.0000** |

Only 13 points above baseline from 19 features means the labels were assigned
close to independently of the features. That is why all five Phase 1 hybrid
models converged on the same ~0.556 validation accuracy: it was a noise floor,
not a tuning problem. The data was also physically inconsistent — Copper Alloy
part mass had a mean of 15.1 kg with a standard deviation of 100.1 kg.

The synopsis permits **"simulation-based lifecycle data"**, so the project now
generates its data from a physics-grounded simulator.

## How the data is generated

1. **Physical anchors.** Density, embodied energy, cradle-to-gate GWP,
   end-of-life recycling rate, recyclate value, tensile strength and hazard
   class come per material from published sources (Ashby, *Materials and the
   Environment* 3rd ed.; Argonne GREET 2023; EU Directive 2000/53/EC; US EPA
   WARM v16). Part mass derives from the component's primary-material mass
   scaled by density ratio, so mass, embodied energy and carbon footprint stay
   physically consistent instead of being sampled independently.

2. **Degradation model.** Each part accumulates thermal, photo-oxidative and
   corrosive dose across its service life, weighted by its component's duty
   cycle and the vehicle type, under a first-order saturation law. This is why
   service life and odometer are genuinely predictive here, as they are in real
   end-of-life data.

3. **Partially observed decision rule.** Labels are drawn from a multi-criteria
   utility model over the latent lifecycle state, with Gumbel noise. Class
   composition is calibrated to the EU ELV stream (~20% reuse / ~50% recycle /
   ~30% dispose).

## The honest limitation

Labels come from a rule, so the models learn a **surrogate of that rule**.
Reported accuracy measures surrogate fidelity, not field disposition accuracy.
Three design choices keep the learning task genuine rather than circular:

- **Unobserved heterogeneity.** Each part carries utility shocks representing
  hidden internal damage, local recyclate market conditions, and operator and
  plant capability. None are exported as features, so no model can recover
  them. Note that the *rest* of the latent state largely is recoverable from
  the features, and correctly so: separability genuinely is determined by a
  part's component, joining method and coating. The shocks are the honest
  source of irreducible error.

- **Two measured ceilings.** An oracle that also knows the shocks scores 92.5%,
  a loose upper bound. Marginalising the shocks out gives
  **89.1%** — the ceiling for any model restricted
  to the exported features, and the number every result should be read against.

- **Extrapolation testing.** A whole material and a whole component are withheld
  from training entirely, which tests generalisation rather than recall.

## Pipeline design decisions

| Decision | Reason |
|---|---|
| Missing values **removed**, not imputed | Evaluator instruction. All three policies are measured so the cost is visible: `drop_rows` loses 6.2% of rows but keeps all 30 features |
| Ordinal encoding for ordered categories | `Poor < Fair < Good < Excellent` is real information that arbitrary label-encoding destroys |
| Balanced class weights everywhere | The minority Dispose class was under-predicted in Phase 1 |
| Model selection on **macro F1**, not accuracy | Accuracy rewards predicting the majority class well, which would sacrifice the Dispose class |
| Asymmetric cost matrix | Releasing a hazardous part into the reuse stream (cost 8.0) is far worse than over-cautiously disposing of a recyclable one (cost 1.5) |
| Early stopping on validation **loss** | Loss detects overconfidence that accuracy cannot, as the LightGBM case demonstrates |

## Reproducing everything

```bash
pip3 install pandas numpy scikit-learn xgboost lightgbm catboost torch \
             matplotlib seaborn shap joblib

python3 -m src.lifecycle.make_dataset     # regenerate the dataset (seed 42)
python3 -m src.lifecycle.train            # train, evaluate, write results/
python3 -m src.lifecycle.make_submission  # rebuild this folder
```

Generation is deterministic at seed 42.
