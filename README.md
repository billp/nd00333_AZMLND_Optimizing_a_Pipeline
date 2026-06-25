# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Engineer Nanodegree. We build and
optimize an Azure ML pipeline two ways and compare them:

1. A custom **Scikit-learn Logistic Regression** whose hyperparameters are tuned
   with **HyperDrive**.
2. An **AutoML** run on the same dataset.

> **Note:** Replace every `XX.XX%` / `<...>` placeholder below with the actual
> values from your run before submitting.

## Summary
**The problem.** The dataset is the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
It contains data from direct marketing (phone call) campaigns of a Portuguese
bank — client demographics, contact details, and campaign attributes. We are
solving a **binary classification** problem: predict whether a client will
**subscribe to a term deposit** (column `y`, yes/no).

**The result.** The best performing model was **\<HyperDrive Logistic Regression | AutoML \<algorithm\>\>**,
which achieved an accuracy of **XX.XX%**. (AutoML typically wins by a small
margin — fill in your real comparison below.)

## Scikit-learn Pipeline

**Architecture.**
- `train.py` loads the Bank Marketing CSV as an Azure `TabularDataset`,
  cleans/encodes it (`clean_data`), and splits it 80/20 into train/test.
- It trains a `LogisticRegression` parameterized by two hyperparameters:
  - `--C` — inverse of regularization strength (lower = stronger regularization).
  - `--max_iter` — maximum solver iterations.
- It logs `Accuracy` and saves the model to `outputs/model.joblib`.
- HyperDrive launches many child runs of `train.py`, each with a different
  `(C, max_iter)` combination sampled by the parameter sampler, and keeps the
  run with the highest `Accuracy`.

**Data.** ~32,950 rows. After encoding (one-hot for `job`, `contact`,
`education`; binary for `marital`, `default`, `housing`, `loan`, `poutcome`;
ordinal maps for `month`, `day_of_week`) we obtain a numeric feature matrix and
a 0/1 label.

**Benefit of the parameter sampler.**
I used **`RandomParameterSampling`** over `C` (continuous, `uniform(0.01, 10)`)
and `max_iter` (discrete, `choice(50, 100, 150, 200)`). Random sampling explores
the search space broadly and cheaply: it finds strong configurations far faster
than an exhaustive **grid search** (which scales combinatorially) while, unlike
grid search, not wasting budget evaluating every point. It is a great default
when you do not yet know which region of the space is promising.

**Benefit of the early stopping policy.**
I used a **`BanditPolicy(evaluation_interval=2, slack_factor=0.1)`**. After each
2 evaluation intervals, any run whose primary metric is more than 10% worse
(`slack_factor=0.1`) than the best run so far is terminated. This frees compute
from clearly inferior configurations and reallocates it, cutting total runtime
and cost without hurting the quality of the best model found.

**Best HyperDrive result.**
- Best accuracy: **XX.XX%**
- Best hyperparameters: `C = <value>`, `max_iter = <value>`

## AutoML

AutoML was given the same cleaned data (label `y` concatenated back in),
`task='classification'`, `primary_metric='accuracy'`, `n_cross_validations=5`,
and a 30-minute timeout. It automatically performed featurization, tried many
algorithms with different hyperparameters, and used an ensemble at the end.

- Best model generated: **\<e.g. VotingEnsemble / StackEnsemble / LightGBM\>**
- Best accuracy: **XX.XX%**
- Notable generated hyperparameters: **\<e.g. learning_rate, n_estimators,
  min_samples_leaf, ensemble weights\>**

AutoML also surfaces useful extras the manual pipeline does not: data-guardrail
checks (e.g. **class imbalance** detection on this dataset), feature importance,
and a ranked leaderboard of candidate models.

## Pipeline comparison

| Pipeline | Best model | Accuracy | Notes |
|---|---|---|---|
| Scikit-learn + HyperDrive | Logistic Regression | **XX.XX%** | We pick the algorithm; tune only `C`, `max_iter`. |
| AutoML | \<algorithm\> | **XX.XX%** | Tries many algorithms + ensembling automatically. |

**Difference in accuracy:** **\<small, e.g. ~0.X percentage points\>**.

**Difference in architecture and why results differ.** The HyperDrive pipeline
is constrained to a *single* model family (logistic regression) — a linear model
— so its ceiling is whatever the best-regularized linear boundary can achieve.
AutoML searches across many model families (tree ensembles, boosting, ensembles
of models) and applies richer featurization, so it can capture non-linear
structure the logistic regression cannot, usually giving a small accuracy edge.
The trade-off is transparency and cost: the HyperDrive logistic regression is
faster and far more interpretable.

## Future work
- **Address class imbalance.** AutoML flags the target as imbalanced (far more
  "no" than "yes"). Accuracy is misleading here; switch the primary metric to
  **AUC_weighted** or **F1**, and try resampling (SMOTE) or class weights.
- **Smarter sampling.** Try `BayesianParameterSampling` for HyperDrive, or widen
  the `C`/`max_iter` ranges and add more hyperparameters.
- **Longer / broader AutoML.** Increase the timeout and enable more models /
  deep featurization.
- **Better evaluation.** Use cross-validation in `train.py` too, and report
  precision/recall/AUC, not just accuracy.

## Proof of cluster clean up
The final notebook cell deletes the compute cluster:
```python
cpu_cluster.delete()
```
\<Include a screenshot of the cluster being deleted, as required by the rubric.\>

## Files
| File | Purpose |
|---|---|
| `train.py` | Loads/cleans data, trains the Scikit-learn Logistic Regression, logs Accuracy, saves the model. |
| `udacity-project.ipynb` | Orchestrates HyperDrive tuning and the AutoML run, registers best models, deletes the cluster. |
| `conda_dependencies.yml` | Environment used to run `train.py` on the compute cluster. |
| `README.md` | This writeup. |
