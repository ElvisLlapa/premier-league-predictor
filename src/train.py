from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data/processed/premier_league_features_with_elo.csv")
MODEL_PATH = Path("models/random_forest_model.joblib")
PREDICTIONS_PATH = Path(
    "data/processed/random_forest_test_predictions.csv"
)
TARGET_COLUMN = "result"

MODEL_FEATURES = [
    "home_last_5_points",
    "home_last_5_wins",
    "home_last_5_draws",
    "home_last_5_losses",
    "home_last_5_goals_for",
    "home_last_5_goals_against",
    "home_last_5_goal_difference",
    "home_matches_available_last_5",
    "home_venue_last_5_points",
    "home_venue_last_5_wins",
    "home_venue_last_5_draws",
    "home_venue_last_5_losses",
    "home_venue_last_5_goals_for",
    "home_venue_last_5_goals_against",
    "home_venue_last_5_goal_difference",
    "home_venue_matches_available_last_5",
    "home_season_points_before_match",
    "home_season_matches_before_match",
    "home_rest_days",
    "away_last_5_points",
    "away_last_5_wins",
    "away_last_5_draws",
    "away_last_5_losses",
    "away_last_5_goals_for",
    "away_last_5_goals_against",
    "away_last_5_goal_difference",
    "away_matches_available_last_5",
    "away_venue_last_5_points",
    "away_venue_last_5_wins",
    "away_venue_last_5_draws",
    "away_venue_last_5_losses",
    "away_venue_last_5_goals_for",
    "away_venue_last_5_goals_against",
    "away_venue_last_5_goal_difference",
    "away_venue_matches_available_last_5",
    "away_season_points_before_match",
    "away_season_matches_before_match",
    "away_rest_days",
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_elo_expected",
    "away_elo_expected",
]


def load_model_data(
    data_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load matches and select only leakage-safe pre-match model features."""
    matches = pd.read_csv(data_path, parse_dates=["date"])
    matches = matches.sort_values(
        ["date", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(matches.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    features = matches[MODEL_FEATURES].copy()
    target = matches[TARGET_COLUMN].copy()

    return features, target, matches

def create_chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Train on older seasons and reserve the newest season for testing."""
    test_season = matches.iloc[-1]["season"]
    test_mask = matches["season"].eq(test_season)

    x_train = features.loc[~test_mask].copy()
    x_test = features.loc[test_mask].copy()
    y_train = target.loc[~test_mask].copy()
    y_test = target.loc[test_mask].copy()
    train_matches = matches.loc[~test_mask].copy()
    test_matches = matches.loc[test_mask].copy()

    if x_train.empty or x_test.empty:
        raise ValueError("Chronological split produced an empty dataset.")

    train_end_date = train_matches["date"].max()
    test_start_date = test_matches["date"].min()

    if train_end_date >= test_start_date:
        raise ValueError(
            "Training matches must occur before all test matches."
        )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        train_matches,
        test_matches,
    )

def prepare_missing_values(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[SimpleImputer, pd.DataFrame, pd.DataFrame]:
    """Fill missing values using medians learned from training data only."""
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")

    x_train_prepared = imputer.fit_transform(x_train)
    x_test_prepared = imputer.transform(x_test)

    x_train_prepared.index = x_train.index
    x_test_prepared.index = x_test.index

    return imputer, x_train_prepared, x_test_prepared

def build_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[DummyClassifier, float]:
    """Build and evaluate a most-common-result baseline."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    predictions = baseline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return baseline, accuracy

def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[RandomForestClassifier, float]:
    """Train and evaluate the first Random Forest match-result model."""
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy

def train_production_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[RandomForestClassifier, SimpleImputer]:
    """Train the final model on all completed seasons."""
    production_imputer = SimpleImputer(strategy="median")
    production_imputer.set_output(transform="pandas")
    prepared_features = production_imputer.fit_transform(features)

    production_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    production_model.fit(prepared_features, target)

    return production_model, production_imputer

def generate_probability_predictions(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Generate correctly mapped result probabilities for test matches."""
    class_positions = {
        result_class: position
        for position, result_class in enumerate(model.classes_)
    }
    required_classes = {"H", "D", "A"}
    missing_classes = required_classes - set(class_positions)

    if missing_classes:
        raise ValueError(
            f"Model is missing result classes: {sorted(missing_classes)}"
        )

    probabilities = model.predict_proba(x_test)
    predictions = test_matches[
        [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    predictions = predictions.rename(columns={"result": "actual_result"})
    predictions["predicted_result"] = model.predict(x_test)
    predictions["home_win_probability"] = probabilities[
        :, class_positions["H"]
    ]
    predictions["draw_probability"] = probabilities[
        :, class_positions["D"]
    ]
    predictions["away_win_probability"] = probabilities[
        :, class_positions["A"]
    ]
    predictions["probability_total"] = (
        predictions["home_win_probability"]
        + predictions["draw_probability"]
        + predictions["away_win_probability"]
    )

    return predictions

def save_training_artifacts(
    production_model: RandomForestClassifier,
    production_imputer: SimpleImputer,
    predictions: pd.DataFrame,
    matches: pd.DataFrame,
    baseline_accuracy: float,
    holdout_accuracy: float,
) -> None:
    """Save the production model and holdout-test predictions."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": production_model,
        "imputer": production_imputer,
        "features": MODEL_FEATURES,
        "class_labels": list(production_model.classes_),
        "baseline_accuracy": baseline_accuracy,
        "holdout_accuracy": holdout_accuracy,
        "training_match_count": len(matches),
        "training_seasons": list(
            matches["season"].drop_duplicates().astype(str)
        ),
        "training_end_date": str(matches["date"].max().date()),
        "prediction_season": "2026-27",
        "purpose": "production",
    }

    joblib.dump(model_bundle, MODEL_PATH)
    predictions.to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    features, target, matches = load_model_data()

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_matches,
        test_matches,
    ) = create_chronological_split(features, target, matches)

    (
        _evaluation_imputer,
        x_train_prepared,
        x_test_prepared,
    ) = prepare_missing_values(
        x_train,
        x_test,
    )


    print(f"Dataset: {DATA_PATH}")
    print(f"Model features: {features.shape[1]}")
    print(f"Training matches: {len(x_train):,}")
    print(f"Test matches: {len(x_test):,}")
    print(
        "Training seasons:",
        ", ".join(train_matches["season"].drop_duplicates().astype(str)),
    )
    print(
        "Test season:",
        ", ".join(test_matches["season"].drop_duplicates().astype(str)),
    )
    print(
        "Training dates:",
        f"{train_matches['date'].min().date()} to "
        f"{train_matches['date'].max().date()}",
    )
    print(
        "Test dates:",
        f"{test_matches['date'].min().date()} to "
        f"{test_matches['date'].max().date()}",
    )
    print("\nTraining result counts:")
    print(y_train.value_counts())
    print("\nTest result counts:")
    print(y_test.value_counts())
    baseline, baseline_accuracy = build_baseline(
        x_train_prepared,
        y_train,
        x_test_prepared,
        y_test,
    )

    print("\nMissing-value preparation:")
    print(f"Training missing before: {x_train.isna().sum().sum()}")
    print(f"Test missing before: {x_test.isna().sum().sum()}")
    print(
        "Training missing after:",
        x_train_prepared.isna().sum().sum(),
    )
    print(
        "Test missing after:",
        x_test_prepared.isna().sum().sum(),
    )

    print("\nBaseline model:")
    print(f"Strategy: {baseline.strategy}")
    baseline_result = baseline.predict(x_test_prepared.iloc[:1])[0]
    print(f"Predicted result: {baseline_result}")
    print(f"Accuracy: {baseline_accuracy:.3f}")

    model, model_accuracy = train_random_forest(
        x_train_prepared,
        y_train,
        x_test_prepared,
        y_test,
    )

    print("\nRandom Forest model:")
    print(f"Trees: {model.n_estimators}")
    print(f"Maximum depth: {model.max_depth}")
    print(f"Minimum samples per leaf: {model.min_samples_leaf}")
    print(f"Classes: {list(model.classes_)}")
    print(f"Accuracy: {model_accuracy:.3f}")
    print(
        "Improvement over baseline:",
        f"{model_accuracy - baseline_accuracy:+.3f}",
    )


    probability_predictions = generate_probability_predictions(
        model,
        x_test_prepared,
        test_matches,
    )

    probability_columns = [
        "date",
        "home_team",
        "away_team",
        "actual_result",
        "predicted_result",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    print("\nFirst five unseen-match predictions:")
    print(
        probability_predictions[probability_columns]
        .head()
        .to_string(index=False)
    )
    print(
        "\nProbability total range:",
        f"{probability_predictions['probability_total'].min():.6f}",
        "to",
        f"{probability_predictions['probability_total'].max():.6f}",
    )
    production_model, production_imputer = train_production_model(
        features,
        target,
    )

    save_training_artifacts(
        production_model,
        production_imputer,
        probability_predictions,
        matches,
        baseline_accuracy,
        model_accuracy,
    )

    print("\nProduction model:")
    print(f"Training matches: {len(features):,}")
    print(
        "Training seasons:",
        ", ".join(matches["season"].drop_duplicates().astype(str)),
    )
    print(f"Training end date: {matches['date'].max().date()}")
    print("Prediction season: 2026-27")
    print(f"Classes: {list(production_model.classes_)}")



    print("\nSaved artifacts:")
    print(f"Model bundle: {MODEL_PATH}")
    print(f"Test predictions: {PREDICTIONS_PATH}")
    print(
        "Model beat baseline:",
        model_accuracy > baseline_accuracy,
    )


if __name__ == "__main__":
    main()