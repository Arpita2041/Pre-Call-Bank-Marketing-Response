import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from project.data import FEATURES, load_data, partition, validate_features
from project.models import choose_threshold, preprocessing
from project.workflow import train, predict

def fixture(n=400):
    """Simulated schema fixture, never a real benchmark dataset."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "age": rng.integers(20, 75, n), "balance": rng.normal(1800, 1200, n),
        "day": np.arange(n) % 28 + 1, "campaign": rng.integers(1, 5, n),
        "pdays": rng.choice([-1, 60, 120], n), "previous": rng.integers(0, 3, n),
        "job": rng.choice(["technician", "admin.", "retired"], n),
        "marital": rng.choice(["single", "married"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": "no", "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n), "contact": "cellular",
        "month": rng.choice(["may", "jun", "oct"], n), "poutcome": "unknown",
        "duration": rng.integers(0, 500, n),
        "y": np.where(np.arange(n) % 4 == 0, "yes", "no")})
    return frame

class BankTests(unittest.TestCase):
    def test_chronology_and_duration_exclusion(self):
        parts = partition(fixture())
        last = -1
        for part in parts.values():
            self.assertGreater(part.index.min(), last)
            last = part.index.max()
        self.assertNotIn("duration", FEATURES)
        f = preprocessing().fit(fixture()[FEATURES])
        self.assertFalse(any("duration" in name for name in f.get_feature_names_out()))

    def test_inference_schema_and_unknown_categories(self):
        frame = fixture()
        transform = preprocessing().fit(frame[FEATURES])
        one = validate_features(frame.iloc[:1])
        one["job"] = "previously_unseen_job"
        self.assertTrue(np.isfinite(transform.transform(one)).all())
        with self.assertRaises(ValueError):
            validate_features(one.drop(columns=["age"]))

    def test_threshold_uses_supplied_holdout(self):
        threshold = choose_threshold([0, 0, 1, 1], np.array([.1, .2, .4, .9]))
        self.assertTrue(np.array_equal(np.array([.1, .2, .4, .9]) >= threshold, [0, 0, 1, 1]))

    def test_complete_fixture_run_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary) / "fixture.csv", Path(temporary) / "run"
            fixture().to_csv(path, sep=";", index=False)
            train("quick", 42, path, output)
            customer = Path(temporary) / "customer.csv"
            fixture(3)[FEATURES].to_csv(customer, index=False)
            result = predict(output / "models/model.joblib", customer)
            self.assertEqual(len(result), 3)
            self.assertTrue(result.subscription_probability.between(0, 1).all())
            registry = pd.read_csv(output / "audit/split_registry.csv")
            self.assertEqual(registry.source_row.nunique(), len(registry))
            self.assertTrue((output / "reports/index.html").exists())

if __name__ == "__main__":
    unittest.main()
