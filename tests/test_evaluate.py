import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer

from src import evaluate
from src.evaluate import (
    TrainedValidationFold,
    ValidationFold,
    calculate_calibration_metrics,
    calculate_class_metrics,
    calculate_confusion_matrices,
    calculate_feature_importance,
    calculate_fold_metrics,
    create_expanding_window_splits,
    generate_evaluation_report,
    save_evaluation_tables,
)


class FakeProbabilityModel:
    """Small model used to test evaluation calculations."""

    classes_ = np.array(["A", "D", "H"])
    feature_importances_ = np.array([0.20, 0.70, 0.10])

    def predict(self, features):
        predictions = np.array(["H", "D", "A"])
        return predictions[: len(features)]

    def predict_proba(self, features):
        probabilities = np.array(
            [
                [0.10, 0.20, 0.70],
                [0.10, 0.80, 0.10],
                [0.70, 0.20, 0.10],
            ]
        )
        return probabilities[: len(features)]


def create_test_trained_fold():
    """Create one small fold with predictable evaluation results."""
    x_train = pd.DataFrame(
        {
            "feature_one": [1.0, 2.0, 3.0],
            "feature_two": [3.0, 2.0, 1.0],
            "feature_three": [2.0, 2.0, 2.0],
        }
    )
    x_test = pd.DataFrame(
        {
            "feature_one": [4.0, 5.0, 6.0],
            "feature_two": [1.0, 2.0, 3.0],
            "feature_three": [2.0, 2.0, 2.0],
        }
    )
    y_train = pd.Series(["H", "D", "A"])
    y_test = pd.Series(["H", "D", "A"])

    train_matches = pd.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-08", "2023-01-15"]
            ),
        }
    )
    test_matches = pd.DataFrame(
        {
            "season": ["2023-24"] * 3,
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15"]
            ),
        }
    )

    fold = ValidationFold(
        fold_number=1,
        training_seasons=("2022-23",),
        test_season="2023-24",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_matches=train_matches,
        test_matches=test_matches,
    )

    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")
    x_train_prepared = imputer.fit_transform(x_train)
    x_test_prepared = imputer.transform(x_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train_prepared, y_train)

    return TrainedValidationFold(
        fold=fold,
        imputer=imputer,
        x_train_prepared=x_train_prepared,
        x_test_prepared=x_test_prepared,
        model=FakeProbabilityModel(),
        baseline=baseline,
        baseline_accuracy=pytest.approx(1 / 3),
    )


def test_expanding_window_splits_use_only_earlier_seasons():
    seasons = [
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]
    matches = pd.DataFrame(
        {
            "season": seasons,
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2022-01-01",
                    "2023-01-01",
                    "2024-01-01",
                    "2025-01-01",
                    "2026-01-01",
                ]
            ),
        }
    )
    features = pd.DataFrame({"feature": range(6)})
    target = pd.Series(["H", "D", "A", "H", "D", "A"])

    folds = create_expanding_window_splits(
        features,
        target,
        matches,
    )

    assert len(folds) == 3
    assert [len(fold.x_train) for fold in folds] == [3, 4, 5]
    assert [fold.test_season for fold in folds] == [
        "2023-24",
        "2024-25",
        "2025-26",
    ]

    for fold in folds:
        assert fold.train_matches["date"].max() < (
            fold.test_matches["date"].min()
        )
        assert fold.test_season not in fold.training_seasons


def test_expanding_window_split_rejects_missing_seasons():
    matches = pd.DataFrame(
        {
            "season": ["2020-21", "2021-22"],
            "date": pd.to_datetime(["2021-01-01", "2022-01-01"]),
        }
    )
    features = pd.DataFrame({"feature": [1, 2]})
    target = pd.Series(["H", "A"])

    with pytest.raises(ValueError, match="missing seasons"):
        create_expanding_window_splits(features, target, matches)


def test_fold_metrics_measure_accuracy_and_log_loss():
    trained_fold = create_test_trained_fold()
    metrics = calculate_fold_metrics([trained_fold])

    assert metrics.loc[0, "model_accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "baseline_accuracy"] == pytest.approx(1 / 3)
    assert metrics.loc[0, "model_log_loss"] < (
        metrics.loc[0, "baseline_log_loss"]
    )


def test_class_and_confusion_metrics_include_all_results():
    trained_fold = create_test_trained_fold()
    class_metrics = calculate_class_metrics([trained_fold])
    confusion_metrics = calculate_confusion_matrices([trained_fold])

    assert set(class_metrics["result_class"]) == {"H", "D", "A"}
    assert class_metrics["recall"].tolist() == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert len(confusion_metrics) == 9
    assert confusion_metrics["match_count"].sum() == 3


def test_calibration_uses_model_class_order():
    trained_fold = create_test_trained_fold()
    brier_scores, calibration_metrics = calculate_calibration_metrics(
        [trained_fold]
    )

    home_brier = brier_scores.loc[
        brier_scores["result_class"].eq("H"),
        "brier_score",
    ].iloc[0]

    assert home_brier == pytest.approx(11 / 300)
    assert set(brier_scores["result_class"]) == {"H", "D", "A"}
    assert not calibration_metrics.empty


def test_feature_importance_is_ranked_for_every_feature():
    trained_fold = create_test_trained_fold()
    importance = calculate_feature_importance([trained_fold])

    assert len(importance) == 3
    assert importance.iloc[0]["feature"] == "feature_two"
    assert importance.iloc[0]["rank"] == 1
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_tables_and_report_are_saved(tmp_path, monkeypatch):
    trained_fold = create_test_trained_fold()
    fold_metrics = calculate_fold_metrics([trained_fold])
    class_metrics = calculate_class_metrics([trained_fold])
    confusion_metrics = calculate_confusion_matrices([trained_fold])
    brier_scores, calibration_metrics = calculate_calibration_metrics(
        [trained_fold]
    )
    feature_importance = calculate_feature_importance([trained_fold])

    tables_directory = tmp_path / "tables"
    report_path = tmp_path / "evaluation_report.md"
    monkeypatch.setattr(evaluate, "TABLES_DIR", tables_directory)
    monkeypatch.setattr(evaluate, "REPORT_PATH", report_path)

    importance_summary = save_evaluation_tables(
        fold_metrics,
        class_metrics,
        confusion_metrics,
        brier_scores,
        calibration_metrics,
        feature_importance,
    )
    saved_report = generate_evaluation_report(
        fold_metrics,
        class_metrics,
        brier_scores,
        importance_summary,
    )

    assert len(list(tables_directory.glob("*.csv"))) == 7
    assert saved_report.exists()

    report_text = saved_report.read_text(encoding="utf-8")
    assert "Time-Based Validation and Evaluation" in report_text
    assert "The production model" in report_text