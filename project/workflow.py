"""Train, select, calibrate, freeze the threshold, and evaluate."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, ConfusionMatrixDisplay,
                             PrecisionRecallDisplay)
from threadpoolctl import threadpool_limits
from .common import output_dirs, provenance, save_json, html_report, dataset_label
from .data import FEATURES, load_data, obtain_data, partition, validate_features
from .models import candidates, choose_threshold, metrics

@threadpool_limits.wrap(limits=1)
def train(mode="full", seed=42, data=None, output="outputs/full"):
    source = Path(data) if data else obtain_data()
    dataset = dataset_label(source, "UCI bank-full", supplied=data is not None)
    frame = load_data(source, mode)
    splits = partition(frame)
    out = output_dirs(output)
    audit = pd.concat([part[["source_row"]].assign(split=name)
                       for name, part in splits.items()], ignore_index=True)
    audit.to_csv(out / "audit/split_registry.csv", index=False)
    x = {name: validate_features(part) for name, part in splits.items()}
    y = {name: (part.y == "yes").astype(int).to_numpy() for name, part in splits.items()}
    fitted, validation = {}, []
    for name, model in candidates(seed, mode).items():
        model.fit(x["train"], y["train"])
        score = average_precision_score(y["validation"],
                                        model.predict_proba(x["validation"])[:, 1])
        validation.append({"model": name, "validation_average_precision": score})
        fitted[name] = model
    leaderboard = pd.DataFrame(validation).sort_values(
        "validation_average_precision", ascending=False, kind="stable")
    selected = str(leaderboard.iloc[0]["model"])
    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted[selected]), method="sigmoid")
    calibrator.fit(x["calibration"], y["calibration"])
    threshold = choose_threshold(y["threshold"],
                                 calibrator.predict_proba(x["threshold"])[:, 1])
    artifact = {"model": calibrator, "threshold": threshold, "selected": selected,
                "features": FEATURES, "seed": seed, "training_data_source": dataset}
    joblib.dump(artifact, out / "models/model.joblib")
    rows = []
    for name, model in fitted.items():
        rows.append({"model": name, "threshold": .5, "selected_from_validation": name == selected,
                     **metrics(y["test"], model.predict_proba(x["test"])[:, 1])})
    p = calibrator.predict_proba(x["test"])[:, 1]
    rows.append({"model": selected + "_calibrated", "threshold": threshold,
                 "selected_from_validation": True, **metrics(y["test"], p, threshold)})
    measured = pd.DataFrame(rows)
    measured["data_source"] = dataset
    measured.to_csv(out / "reports/metrics.csv", index=False)
    leaderboard.to_csv(out / "reports/validation.csv", index=False)
    pd.DataFrame({"source_row": splits["test"].source_row.to_numpy(),
                  "target": y["test"], "probability": p,
                  "prediction": (p >= threshold).astype(int)}).to_csv(
                      out / "reports/test_predictions.csv", index=False)
    importance = permutation_importance(fitted[selected], x["validation"], y["validation"],
                                        scoring="average_precision", n_repeats=3,
                                        random_state=seed, n_jobs=1)
    importance_frame = pd.DataFrame({"feature": FEATURES,
                                    "mean_AP_decrease": importance.importances_mean,
                                    "std_AP_decrease": importance.importances_std}).sort_values(
                                        "mean_AP_decrease", ascending=False)
    importance_frame.to_csv(out / "reports/permutation_importance.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y["test"], p >= threshold, ax=axes[0],
                                            display_labels=["no", "yes"], colorbar=False)
    PrecisionRecallDisplay.from_predictions(y["test"], p, ax=axes[1])
    fraction, mean = calibration_curve(y["test"], p, n_bins=8, strategy="quantile")
    axes[2].plot(mean, fraction, "o-", label="selected calibrated model")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(xlabel="Mean probability", ylabel="Observed positive fraction",
                title="Test-set calibration")
    if dataset.startswith("SIMULATED"):
        fig.suptitle("Simulated software demonstration")
    fig.tight_layout()
    fig.savefig(out / "reports/evaluation.png", dpi=150)
    plt.close(fig)
    record = provenance(mode, seed, source)
    record.update({"dataset": dataset,
                   "selected": selected, "threshold": threshold,
                   "selection": "maximum validation average precision",
                   "counts": {name: len(part) for name, part in splits.items()},
                   "excluded": ["duration"], "rows_after_deduplication_and_sampling": len(frame)})
    save_json(out / "audit/run.json", record)
    html_report(out / "reports/index.html", "Pre-call marketing response prediction",
                f"Data: {dataset}. Selected from validation: {selected}. Test rows were used only after "
                "calibration and threshold selection. Duration is excluded.",
                [("Held-out test metrics", measured), ("Validation selection", leaderboard),
                 ("Validation permutation importance", importance_frame)],
                [("evaluation.png", "Test confusion matrix, precision-recall curve, and calibration.")])
    print(measured.to_string(index=False))
    print(f"Report: {out / 'reports/index.html'}")
    return out

def predict(model_path, input_path):
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    x = validate_features(frame)
    p = artifact["model"].predict_proba(x)[:, 1]
    return pd.DataFrame({"subscription_probability": p,
                         "predicted_subscription": p >= artifact["threshold"],
                         "training_data_source": artifact["training_data_source"]})
