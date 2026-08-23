import numpy as np
import pandas as pd
import pytest

from src.train import (
    MODEL_FEATURES,
    create_chronological_split,
    generate_probability_predictions,
    prepare_missing_values,
    train_production_model,
)


class FakeProbabilityModel:
    """Small fake model used to test probability mapping."""

    classes_ = np.array(["A", "D", "H"])

    def predict_proba(self, features):
        probabilities = np.array([[0.20, 0.25, 0.55]])
        return np.repeat(probabilities, len(features), axis=0)

    def predict(self, features):
        return np.repeat("H", len(features))


def test_model_features_exclude_leakage_columns():
    leakage_columns = {
        "home_goals",
        "away_goals",
        "result",
        "date",
        "home_team",
        "away_team",
    }

    assert leakage_columns.isdisjoint(MODEL_FEATURES)


def test_chronological_split_reserves_newest_season():
    matches = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-08",
                    "2026-01-01",
                    "2026-01-08",
                ]
            ),
            "season": [
                "2024-25",
                "2024-25",
                "2025-26",
                "2025-26",
            ],
        }
    )
    features = pd.DataFrame({"example_feature": [1, 2, 3, 4]})
    target = pd.Series(["H", "D", "A", "H"])

    (
        x_train,
        x_test,
        _y_train,
        _y_test,
        train_matches,
        test_matches,
    ) = create_chronological_split(features, target, matches)

    assert len(x_train) == 2
    assert len(x_test) == 2
    assert train_matches["season"].unique().tolist() == ["2024-25"]
    assert test_matches["season"].unique().tolist() == ["2025-26"]
    assert train_matches["date"].max() < test_matches["date"].min()


def test_imputer_learns_from_training_data_only():
    x_train = pd.DataFrame(
        {
            "rest_days": [4.0, 6.0, np.nan],
        }
    )
    x_test = pd.DataFrame(
        {
            "rest_days": [np.nan, 100.0],
        }
    )

    _imputer, x_train_prepared, x_test_prepared = (
        prepare_missing_values(x_train, x_test)
    )

    assert x_train_prepared.isna().sum().sum() == 0
    assert x_test_prepared.isna().sum().sum() == 0
    assert x_test_prepared.iloc[0]["rest_days"] == pytest.approx(5.0)


def test_probabilities_are_mapped_using_model_classes():
    model = FakeProbabilityModel()
    features = pd.DataFrame({"example_feature": [1]})
    matches = pd.DataFrame(
        {
            "match_id": [1],
            "date": pd.to_datetime(["2026-08-15"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "result": ["H"],
        }
    )

    predictions = generate_probability_predictions(
        model,
        features,
        matches,
    )

    assert predictions.loc[0, "home_win_probability"] == pytest.approx(
        0.55
    )
    assert predictions.loc[0, "draw_probability"] == pytest.approx(0.25)
    assert predictions.loc[0, "away_win_probability"] == pytest.approx(
        0.20
    )
    assert predictions.loc[0, "probability_total"] == pytest.approx(1.0)


def test_production_model_handles_missing_values_and_all_classes():
    features = pd.DataFrame(
        {
            "feature_one": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "feature_two": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    target = pd.Series(["H", "D", "A", "H", "D", "A"])

    model, imputer = train_production_model(features, target)
    prepared_features = imputer.transform(features)
    probabilities = model.predict_proba(prepared_features)

    assert list(model.classes_) == ["A", "D", "H"]
    assert not prepared_features.isna().any().any()
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)