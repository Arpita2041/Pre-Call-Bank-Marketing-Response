"""Training-only preprocessing, classifiers, and operating thresholds."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             log_loss)
from .data import NUMERIC, CATEGORICAL

def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False))]), CATEGORICAL)])

def candidates(seed=42, mode="full"):
    choices = {"prevalence_baseline": DummyClassifier(strategy="prior")}
    for c in ([1.] if mode == "quick" else [.3, 1., 3.]):
        choices[f"logistic_C{c:g}"] = LogisticRegression(
            C=c, class_weight="balanced", max_iter=1500, random_state=seed)
    choices["random_forest"] = RandomForestClassifier(
        n_estimators=50 if mode == "quick" else 250,
        max_depth=16, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=seed, n_jobs=1)
    choices["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=50 if mode == "quick" else 180, max_leaf_nodes=15,
        learning_rate=.08, l2_regularization=1., early_stopping=False,
        random_state=seed)
    return {name: Pipeline([("preprocess", preprocessing()), ("classifier", model)])
            for name, model in choices.items()}

def choose_threshold(y, probability):
    thresholds = np.unique(np.r_[.5, np.quantile(probability, np.linspace(0, 1, 201))])
    scores = np.array([f1_score(y, probability >= t, zero_division=0) for t in thresholds])
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(thresholds[tied[np.argmin(np.abs(thresholds[tied] - .5))]])

def metrics(y, probability, threshold=.5):
    p = np.clip(np.asarray(probability), 1e-9, 1 - 1e-9)
    prediction = np.asarray(probability) >= threshold
    return {"accuracy": accuracy_score(y, prediction),
            "positive_f1": f1_score(y, prediction, zero_division=0),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "average_precision": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1])}
