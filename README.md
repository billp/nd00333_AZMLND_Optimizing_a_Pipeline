# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Engineer Nanodegree. We build and
optimize an Azure ML pipeline two ways and compare them:

1. A custom **Scikit-learn Logistic Regression** whose hyperparameters are tuned
   with **HyperDrive**.
2. An **AutoML** run on the same dataset.

## Summary
**The problem.** The dataset is the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
It contains data from direct marketing (phone call) campaigns of a Portuguese
bank — client demographics, contact details, and campaign attributes (32,950
rows). We are solving a **binary classification** problem: predict whether a
client will **subscribe to a term deposit** (column `y`, yes/no).

**The result.** The best performing model came from **AutoML** — a
**StandardScalerWrapper + XGBoostClassifier** pipeline with an accuracy of
**0.9151 (91.51%)**. It beat the best HyperDrive-tuned Logistic Regression,
which reached **0.9088 (90.88%)**.

## Scikit-learn Pipeline

**Architecture.**
- `train.py` loads the Bank Marketing CSV as an Azure `TabularDataset`, cleans
  and encodes it (`clean_data`), and splits it 80/20 into train/test.
- It trains a `LogisticRegression` parameterized by two hyperparameters:
  - `--C` — inverse of regularization strength (lower = stronger regularization).
  - `--max_iter` — maximum solver iterations.
- It logs `Accuracy` and saves the model to `outputs/model.joblib`.
- HyperDrive launches many child runs of `train.py`, each with a different
  `(C, max_iter)` combination drawn by the parameter sampler, and keeps the run
  with the highest `Accuracy`.

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
I used a **`BanditPolicy(evaluation_interval=2, slack_factor=0.1)`**. After every
2 evaluation intervals, any run whose primary metric is more than 10%
(`slack_factor=0.1`) worse than the best run so far is terminated. This frees
compute from clearly inferior configurations and reallocates it, cutting total
runtime and cost without hurting the quality of the best model found.

**Best HyperDrive result.**
- Best accuracy: **0.9088**
- Best hyperparameters: **`C = 5.101`**, **`max_iter = 150`**
- Best run id: `HD_bc0e3e06-8e84-455d-af4c-1009ee831ec9_11`

## AutoML

AutoML was given the same cleaned data (registered as a TabularDataset),
`task='classification'`, `primary_metric='accuracy'`, `n_cross_validations=5`,
and a 30-minute timeout. It automatically performed featurization, swept many
algorithms with different hyperparameters and scalers, and finished with
ensembles.

**Models AutoML explored** (sample of the leaderboard):

| Iter | Pipeline | Accuracy |
|---|---|---|
| 0 | MaxAbsScaler · LightGBM | 0.9142 |
| 3 | SparseNormalizer · XGBoostClassifier | 0.9141 |
| 7 | MaxAbsScaler · LogisticRegression | 0.9084 |
| 11 | **StandardScalerWrapper · XGBoostClassifier** | **0.9151** |
| — | VotingEnsemble / StackEnsemble (also generated) | ~0.915 |

- **Best model (registered): `StandardScalerWrapper + XGBoostClassifier`** —
  gradient-boosted trees (`booster=gbtree`, `max_depth=6`, `eta=0.3`,
  `reg_alpha=0.3125`, `reg_lambda≈2.40`, `n_estimators=10`) on standard-scaled
  features. This was the model returned by `get_output()` and registered as
  `automl_best_model`.
- **Best accuracy: 0.9151.**
- AutoML also automatically generated **VotingEnsemble** and **StackEnsemble**
  candidates during the search (soft-voting / stacking over the top pipelines),
  illustrating how it ensembles models beyond what the single Scikit-learn
  pipeline can do.

AutoML also ran **data guardrails**, which flagged **class imbalance**: the
positive class (`y = yes`) has only **3,692 of 32,950** samples (~11%). It
confirmed there were no missing values and no high-cardinality features.

## Pipeline comparison

| Pipeline | Best model | Accuracy |
|---|---|---|
| Scikit-learn + HyperDrive | LogisticRegression (`C=5.101`, `max_iter=150`) | **0.9088** |
| AutoML | **StandardScalerWrapper + XGBoostClassifier** | **0.9151** |

**Difference in accuracy:** AutoML was higher by **~0.63 percentage points**
(0.9151 vs 0.9088).

**Difference in architecture and why results differ.** The HyperDrive pipeline
is constrained to a *single* model family — logistic regression, a linear model —
so its ceiling is whatever the best-regularized linear decision boundary can
achieve. AutoML searches across many model families (gradient-boosted trees like
XGBoost and LightGBM, random forests, etc.), applies richer automated
featurization, and finally **ensembles** the best of them. That lets it capture
non-linear structure the logistic regression cannot, giving a small but
consistent accuracy edge. The trade-off is interpretability and cost: the
HyperDrive logistic regression is faster, simpler, and far easier to explain.

## Future work
- **Address class imbalance.** AutoML flagged the target as imbalanced (~11%
  positives). Accuracy is misleading here; switch the primary metric to
  **AUC_weighted** or **F1**, and try resampling (SMOTE) or class weights
  (`class_weight='balanced'` in the logistic regression).
- **Smarter / wider hyperparameter search.** Try `BayesianParameterSampling` for
  HyperDrive, widen the `C` and `max_iter` ranges, and tune more hyperparameters
  (solver, penalty).
- **Longer / broader AutoML.** Increase `experiment_timeout_minutes` and enable
  more models / deep featurization to let the ensemble search go further.
- **Better evaluation.** Report precision/recall/AUC and a confusion matrix, not
  just accuracy, and use stratified cross-validation in `train.py` too.

## Proof of cluster clean up
The final notebook cell deletes the compute cluster:
```python
cpu_cluster.delete()
```
(See the accompanying screenshot of the cluster deletion.)

## Notes on reproducing this run
The original Azure sample blob for `bankmarketing_train.csv` is no longer
publicly accessible (HTTP 403), so an identical copy of the dataset is hosted in
this repo and `train.py` reads it from the raw GitHub URL (with a pandas
fallback). AutoML was run **locally** on the compute instance because the remote
`AutoML-Non-Prod` curated-environment image build fails with a `urllib3`
dependency conflict in this lab; running locally uses the already-installed
AutoML runtime and avoids the Docker image build.

## Files
| File | Purpose |
|---|---|
| `train.py` | Loads/cleans data, trains the Scikit-learn Logistic Regression, logs Accuracy, saves the model. |
| `udacity-project.ipynb` | Orchestrates HyperDrive tuning and the AutoML run, registers best models, deletes the cluster. |
| `bankmarketing_train.csv` | Self-hosted copy of the dataset (original Azure blob is offline). |
| `conda_dependencies.yml` | Environment used to run `train.py` on the compute cluster. |
| `README.md` | This writeup. |
