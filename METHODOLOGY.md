# Methodology

The task is binary prediction of subscription, using bank-full.csv from UCI.
This is a pre-call benchmark: the last call's duration is unavailable at
prediction time and is excluded everywhere. A campaign contact count is treated
as known when scheduling that contact. The code retains the original CSV order;
UCI describes bank-full as date ordered. Day and month alone cannot reconstruct
complete dates.

After removing identical feature-and-label rows, chronological partitions use
60% for training, 15% for model selection, 5% for sigmoid calibration, 5% for
threshold selection, and 15% for the final test. Quick mode samples up to 6,000
equally spaced rows while retaining order. It is a workflow check with a different
sample size, not the final course experiment.

Numeric columns use training medians and standardization. Categorical columns
use training-only one-hot encoding; unseen categories are ignored by the encoder.
The entire transformation is inside each fitted pipeline. Class-balanced
logistic regression and a random forest are compared with histogram gradient
boosting and a prior-probability baseline. Full mode evaluates three logistic
regularization strengths. Hyperparameters appear directly in project/models.py.

Selection maximizes validation average precision. The chosen pipeline is frozen
before a sigmoid calibrator is fitted on the calibration partition. The next
partition selects the threshold that maximizes positive-class F1. Exact score ties
favor the threshold nearest 0.5. The final test is used only after these choices.
Uncalibrated candidate scores and the calibrated deployed model are reported
separately; no winner is chosen from test scores.

Average precision measures ranking of the rare positive class. Precision, recall,
and F1 depend on the operating threshold. Brier score and log loss assess
probabilities. The plots show the selected model's confusion matrix,
precision-recall curve, and observed calibration. Permutation importance uses
validation data and reports a change in average precision; it is not causal
evidence and correlated features can share importance.

Repeated customers have no reliable identifier here, so these splits cannot
establish independence between customers. Chronological drift can change both
ranking and calibration. The example customers are constructed input examples,
not predictions about actual people. This benchmark supports method study;
new customer populations require their own validation.
