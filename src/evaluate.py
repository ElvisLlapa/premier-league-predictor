from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from src.train import (
    build_baseline,
    load_model_data,
    prepare_missing_values,
    train_random_forest,
)

VALIDATION_FOLDS = [
    (
        ("2020-21", "2021-22", "2022-23"),
        "2023-24",
    ),
    (
        ("2020-21", "2021-22", "2022-23", "2023-24"),
        "2024-25",
    ),
    (
        (
            "2020-21",
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
        ),
        "2025-26",
    ),
]

REPORT_DIR = Path("reports/phase_7")
TABLES_DIR = REPORT_DIR / "tables"
CHARTS_DIR = REPORT_DIR / "charts"
REPORT_PATH = REPORT_DIR / "evaluation_report.md"


@dataclass
class ValidationFold:
    """Store the training and testing data for one validation fold."""

    fold_number: int
    training_seasons: tuple[str, ...]
    test_season: str
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_matches: pd.DataFrame
    test_matches: pd.DataFrame

@dataclass
class TrainedValidationFold:
    """Store one fold after its models have been trained."""

    fold: ValidationFold
    imputer: SimpleImputer
    x_train_prepared: pd.DataFrame
    x_test_prepared: pd.DataFrame
    model: RandomForestClassifier
    baseline: DummyClassifier
    baseline_accuracy: float


def create_expanding_window_splits(
    features: pd.DataFrame,
    target: pd.Series,
    matches: pd.DataFrame,
) -> list[ValidationFold]:
    """Create folds that train on the past and test on the next season."""
    available_seasons = set(matches["season"].astype(str))
    folds = []

    for fold_number, (training_seasons, test_season) in enumerate(
        VALIDATION_FOLDS,
        start=1,
    ):
        required_seasons = set(training_seasons) | {test_season}
        missing_seasons = sorted(required_seasons - available_seasons)

        if missing_seasons:
            raise ValueError(
                f"Fold {fold_number} is missing seasons: {missing_seasons}"
            )

        training_mask = matches["season"].isin(training_seasons)
        test_mask = matches["season"].eq(test_season)

        x_train = features.loc[training_mask].copy()
        x_test = features.loc[test_mask].copy()
        y_train = target.loc[training_mask].copy()
        y_test = target.loc[test_mask].copy()
        train_matches = matches.loc[training_mask].copy()
        test_matches = matches.loc[test_mask].copy()

        if x_train.empty or x_test.empty:
            raise ValueError(
                f"Fold {fold_number} produced an empty dataset."
            )

        training_end_date = train_matches["date"].max()
        test_start_date = test_matches["date"].min()

        if training_end_date >= test_start_date:
            raise ValueError(
                f"Fold {fold_number} training data must occur "
                "before its test data."
            )

        folds.append(
            ValidationFold(
                fold_number=fold_number,
                training_seasons=training_seasons,
                test_season=test_season,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                train_matches=train_matches,
                test_matches=test_matches,
            )
        )

    return folds

def train_validation_folds(
    folds: list[ValidationFold],
) -> list[TrainedValidationFold]:
    """Prepare and train fresh models and baselines inside every fold."""
    trained_folds = []

    for fold in folds:
        (
            imputer,
            x_train_prepared,
            x_test_prepared,
        ) = prepare_missing_values(
            fold.x_train,
            fold.x_test,
        )

        model, _accuracy = train_random_forest(
            x_train_prepared,
            fold.y_train,
            x_test_prepared,
            fold.y_test,
        )

        baseline, baseline_accuracy = build_baseline(
            x_train_prepared,
            fold.y_train,
            x_test_prepared,
            fold.y_test,
        )

        trained_folds.append(
            TrainedValidationFold(
                fold=fold,
                imputer=imputer,
                x_train_prepared=x_train_prepared,
                x_test_prepared=x_test_prepared,
                model=model,
                baseline=baseline,
                baseline_accuracy=baseline_accuracy,
            )
        )

    model_ids = {id(trained_fold.model) for trained_fold in trained_folds}
    baseline_ids = {
        id(trained_fold.baseline) for trained_fold in trained_folds
    }

    if len(model_ids) != len(trained_folds):
        raise ValueError("Every validation fold must use a fresh model.")

    if len(baseline_ids) != len(trained_folds):
        raise ValueError("Every validation fold must use a fresh baseline.")

    return trained_folds

def calculate_fold_metrics(
    trained_folds: list[TrainedValidationFold],
) -> pd.DataFrame:
    """Calculate accuracy and multiclass log loss for every fold."""
    metric_rows = []

    for trained_fold in trained_folds:
        fold = trained_fold.fold

        model_predictions = trained_fold.model.predict(
            trained_fold.x_test_prepared
        )
        model_probabilities = trained_fold.model.predict_proba(
            trained_fold.x_test_prepared
        )

        baseline_predictions = trained_fold.baseline.predict(
            trained_fold.x_test_prepared
        )
        baseline_probabilities = trained_fold.baseline.predict_proba(
            trained_fold.x_test_prepared
        )

        model_accuracy = accuracy_score(
            fold.y_test,
            model_predictions,
        )
        model_log_loss = log_loss(
            fold.y_test,
            model_probabilities,
            labels=list(trained_fold.model.classes_),
        )
        baseline_accuracy = accuracy_score(
            fold.y_test,
            baseline_predictions,
        )
        baseline_log_loss = log_loss(
            fold.y_test,
            baseline_probabilities,
            labels=list(trained_fold.baseline.classes_),
        )

        metric_rows.append(
            {
                "fold": fold.fold_number,
                "test_season": fold.test_season,
                "training_matches": len(fold.x_train),
                "test_matches": len(fold.x_test),
                "baseline_accuracy": baseline_accuracy,
                "model_accuracy": model_accuracy,
                "accuracy_improvement": (
                    model_accuracy - baseline_accuracy
                ),
                "baseline_log_loss": baseline_log_loss,
                "model_log_loss": model_log_loss,
                "log_loss_improvement": (
                    baseline_log_loss - model_log_loss
                ),
            }
        )

    return pd.DataFrame(metric_rows)

def calculate_class_metrics(
    trained_folds: list[TrainedValidationFold],
) -> pd.DataFrame:
    """Calculate precision, recall, and F1 for every result class."""
    class_labels = ["H", "D", "A"]
    metric_rows = []

    for trained_fold in trained_folds:
        fold = trained_fold.fold
        predictions = trained_fold.model.predict(
            trained_fold.x_test_prepared
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                fold.y_test,
                predictions,
                labels=class_labels,
                zero_division=0,
            )
        )

        for position, result_class in enumerate(class_labels):
            metric_rows.append(
                {
                    "fold": fold.fold_number,
                    "test_season": fold.test_season,
                    "result_class": result_class,
                    "precision": precision[position],
                    "recall": recall[position],
                    "f1_score": f1[position],
                    "support": int(support[position]),
                }
            )

    return pd.DataFrame(metric_rows)

def calculate_confusion_matrices(
    trained_folds: list[TrainedValidationFold],
) -> pd.DataFrame:
    """Create a confusion-matrix table for every validation fold."""
    class_labels = ["H", "D", "A"]
    matrix_rows = []

    for trained_fold in trained_folds:
        fold = trained_fold.fold
        predictions = trained_fold.model.predict(
            trained_fold.x_test_prepared
        )
        matrix = confusion_matrix(
            fold.y_test,
            predictions,
            labels=class_labels,
        )

        for actual_position, actual_result in enumerate(class_labels):
            for predicted_position, predicted_result in enumerate(
                class_labels
            ):
                matrix_rows.append(
                    {
                        "fold": fold.fold_number,
                        "test_season": fold.test_season,
                        "actual_result": actual_result,
                        "predicted_result": predicted_result,
                        "match_count": int(
                            matrix[actual_position, predicted_position]
                        ),
                    }
                )

    return pd.DataFrame(matrix_rows)

def calculate_calibration_metrics(
    trained_folds: list[TrainedValidationFold],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate one-vs-rest Brier scores and calibration bins."""
    result_labels = ["H", "D", "A"]
    probability_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    probability_bin_labels = [
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
    ]
    brier_rows = []
    calibration_rows = []

    for trained_fold in trained_folds:
        fold = trained_fold.fold
        probabilities = trained_fold.model.predict_proba(
            trained_fold.x_test_prepared
        )
        class_positions = {
            result_class: position
            for position, result_class in enumerate(
                trained_fold.model.classes_
            )
        }

        for result_class in result_labels:
            if result_class not in class_positions:
                raise ValueError(
                    f"Fold {fold.fold_number} is missing "
                    f"class {result_class}."
                )

            class_probabilities = probabilities[
                :, class_positions[result_class]
            ]
            actual_binary = fold.y_test.eq(result_class).astype(int)

            brier_score = brier_score_loss(
                actual_binary,
                class_probabilities,
            )

            brier_rows.append(
                {
                    "fold": fold.fold_number,
                    "test_season": fold.test_season,
                    "result_class": result_class,
                    "brier_score": brier_score,
                }
            )

            calibration_data = pd.DataFrame(
                {
                    "predicted_probability": class_probabilities,
                    "actual_result": actual_binary.to_numpy(),
                }
            )
            calibration_data["probability_bin"] = pd.cut(
                calibration_data["predicted_probability"],
                bins=probability_bins,
                labels=probability_bin_labels,
                include_lowest=True,
            )

            grouped_bins = calibration_data.groupby(
                "probability_bin",
                observed=True,
            )

            for probability_bin, bin_data in grouped_bins:
                calibration_rows.append(
                    {
                        "fold": fold.fold_number,
                        "test_season": fold.test_season,
                        "result_class": result_class,
                        "probability_bin": str(probability_bin),
                        "average_predicted_probability": (
                            bin_data["predicted_probability"].mean()
                        ),
                        "observed_frequency": (
                            bin_data["actual_result"].mean()
                        ),
                        "match_count": len(bin_data),
                    }
                )

    return (
        pd.DataFrame(brier_rows),
        pd.DataFrame(calibration_rows),
    )

def calculate_feature_importance(
    trained_folds: list[TrainedValidationFold],
) -> pd.DataFrame:
    """Collect and rank Random Forest feature importance for every fold."""
    importance_rows = []

    for trained_fold in trained_folds:
        fold = trained_fold.fold
        feature_names = list(trained_fold.x_train_prepared.columns)
        importance_values = trained_fold.model.feature_importances_

        if len(feature_names) != len(importance_values):
            raise ValueError(
                f"Fold {fold.fold_number} feature names and "
                "importance values do not match."
            )

        ranked_positions = importance_values.argsort()[::-1]

        for rank, position in enumerate(ranked_positions, start=1):
            importance_rows.append(
                {
                    "fold": fold.fold_number,
                    "test_season": fold.test_season,
                    "feature": feature_names[position],
                    "importance": importance_values[position],
                    "rank": rank,
                }
            )

    return pd.DataFrame(importance_rows)

def save_evaluation_tables(
    fold_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    brier_scores: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Save all Phase 7 evaluation tables as CSV files."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    feature_importance_summary = (
        feature_importance.groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            importance_std=("importance", "std"),
            average_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        )
        .reset_index()
        .sort_values(
            ["mean_importance", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    fold_metrics.to_csv(
        TABLES_DIR / "fold_metrics.csv",
        index=False,
    )
    class_metrics.to_csv(
        TABLES_DIR / "class_metrics.csv",
        index=False,
    )
    confusion_matrices.to_csv(
        TABLES_DIR / "confusion_matrices.csv",
        index=False,
    )
    brier_scores.to_csv(
        TABLES_DIR / "brier_scores.csv",
        index=False,
    )
    calibration_metrics.to_csv(
        TABLES_DIR / "calibration_metrics.csv",
        index=False,
    )
    feature_importance.to_csv(
        TABLES_DIR / "feature_importance_by_fold.csv",
        index=False,
    )
    feature_importance_summary.to_csv(
        TABLES_DIR / "feature_importance_summary.csv",
        index=False,
    )

    return feature_importance_summary

def create_accuracy_chart(
    fold_metrics: pd.DataFrame,
) -> Path:
    """Create a baseline-versus-model accuracy chart."""
    chart_path = CHARTS_DIR / "accuracy_comparison.png"
    positions = list(range(len(fold_metrics)))
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        [position - bar_width / 2 for position in positions],
        fold_metrics["baseline_accuracy"],
        width=bar_width,
        label="Baseline",
        color="#9ca3af",
    )
    axis.bar(
        [position + bar_width / 2 for position in positions],
        fold_metrics["model_accuracy"],
        width=bar_width,
        label="Random Forest",
        color="#2563eb",
    )

    axis.set_title("Accuracy by Unseen Test Season")
    axis.set_xlabel("Test season")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(positions)
    axis.set_xticklabels(fold_metrics["test_season"])
    axis.set_ylim(0, 0.7)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    return chart_path


def create_confusion_matrix_chart(
    confusion_matrices: pd.DataFrame,
) -> Path:
    """Create one confusion-matrix chart for every fold."""
    chart_path = CHARTS_DIR / "confusion_matrices.png"
    result_labels = ["H", "D", "A"]
    folds = sorted(confusion_matrices["fold"].unique())

    figure, axes = plt.subplots(
        1,
        len(folds),
        figsize=(14, 4.5),
    )

    for axis, fold_number in zip(axes, folds, strict=True):
        fold_data = confusion_matrices.loc[
            confusion_matrices["fold"].eq(fold_number)
        ]
        test_season = fold_data["test_season"].iloc[0]
        matrix = (
            fold_data.pivot(
                index="actual_result",
                columns="predicted_result",
                values="match_count",
            )
            .reindex(index=result_labels, columns=result_labels)
            .to_numpy()
        )

        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Fold {fold_number}: {test_season}")
        axis.set_xlabel("Predicted result")
        axis.set_ylabel("Actual result")
        axis.set_xticks(range(len(result_labels)))
        axis.set_yticks(range(len(result_labels)))
        axis.set_xticklabels(result_labels)
        axis.set_yticklabels(result_labels)

        for row in range(len(result_labels)):
            for column in range(len(result_labels)):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="black",
                )

    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.suptitle("Random Forest Confusion Matrices")
    figure.subplots_adjust(
        left=0.06,
        right=0.92,
        bottom=0.14,
        top=0.82,
        wspace=0.35,
    )
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    return chart_path


def create_calibration_chart(
    calibration_metrics: pd.DataFrame,
) -> Path:
    """Create combined one-vs-rest calibration curves."""
    chart_path = CHARTS_DIR / "probability_calibration.png"
    combined = calibration_metrics.copy()
    combined["predicted_probability_sum"] = (
        combined["average_predicted_probability"]
        * combined["match_count"]
    )
    combined["actual_positive_count"] = (
        combined["observed_frequency"] * combined["match_count"]
    )

    combined = (
        combined.groupby(
            ["result_class", "probability_bin"],
            as_index=False,
        )[
            [
                "predicted_probability_sum",
                "actual_positive_count",
                "match_count",
            ]
        ]
        .sum()
    )
    combined["average_predicted_probability"] = (
        combined["predicted_probability_sum"]
        / combined["match_count"]
    )
    combined["observed_frequency"] = (
        combined["actual_positive_count"]
        / combined["match_count"]
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    result_names = {
        "H": "Home win",
        "D": "Draw",
        "A": "Away win",
    }
    result_colors = {
        "H": "#2563eb",
        "D": "#f59e0b",
        "A": "#dc2626",
    }

    for result_class in ["H", "D", "A"]:
        class_data = combined.loc[
            combined["result_class"].eq(result_class)
        ].sort_values("average_predicted_probability")

        axis.plot(
            class_data["average_predicted_probability"],
            class_data["observed_frequency"],
            marker="o",
            linewidth=2,
            label=result_names[result_class],
            color=result_colors[result_class],
        )

    axis.set_title("Probability Calibration Across All Folds")
    axis.set_xlabel("Average predicted probability")
    axis.set_ylabel("Observed result frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    return chart_path


def create_feature_importance_chart(
    feature_importance_summary: pd.DataFrame,
) -> Path:
    """Create a chart showing the 15 most important features."""
    chart_path = CHARTS_DIR / "feature_importance.png"
    top_features = (
        feature_importance_summary.head(15)
        .sort_values("mean_importance")
        .copy()
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        top_features["feature"],
        top_features["mean_importance"],
        color="#2563eb",
    )
    axis.set_title("Top 15 Features Across Validation Folds")
    axis.set_xlabel("Average Random Forest importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    return chart_path


def save_evaluation_charts(
    fold_metrics: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    feature_importance_summary: pd.DataFrame,
) -> list[Path]:
    """Generate and save all Phase 7 evaluation charts."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        create_accuracy_chart(fold_metrics),
        create_confusion_matrix_chart(confusion_matrices),
        create_calibration_chart(calibration_metrics),
        create_feature_importance_chart(feature_importance_summary),
    ]

def generate_evaluation_report(
    fold_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    brier_scores: pd.DataFrame,
    feature_importance_summary: pd.DataFrame,
) -> Path:
    """Generate the written Phase 7 evaluation report."""
    average_baseline_accuracy = fold_metrics[
        "baseline_accuracy"
    ].mean()
    average_model_accuracy = fold_metrics["model_accuracy"].mean()
    average_accuracy_improvement = fold_metrics[
        "accuracy_improvement"
    ].mean()
    average_model_log_loss = fold_metrics["model_log_loss"].mean()

    average_class_metrics = (
        class_metrics.groupby("result_class", sort=False)[
            ["precision", "recall", "f1_score"]
        ]
        .mean()
        .reset_index()
    )
    average_brier_scores = (
        brier_scores.groupby("result_class", sort=False)["brier_score"]
        .mean()
        .reset_index()
    )

    report_lines = [
        "# Phase 7 — Time-Based Validation and Evaluation",
        "",
        "## Purpose",
        "",
        (
            "This report evaluates the Premier League Random Forest model "
            "using expanding-window chronological validation."
        ),
        (
            "Every fold trains only on earlier seasons and tests on the "
            "next unseen season."
        ),
        "",
                (
            "The production model trained for 2026–27 was not used to "
            "produce these evaluation results."
        ),

        "",
        "## Validation folds",
        "",

                (
            "| Fold | Training seasons | Test season | Training matches | "
            "Test matches |"
        ),

        "| --- | --- | --- | ---: | ---: |",
        (
            "| 1 | 2020-21 through 2022-23 | 2023-24 | "
            "1,140 | 380 |"
        ),
        (
            "| 2 | 2020-21 through 2023-24 | 2024-25 | "
            "1,520 | 380 |"
        ),
        (
            "| 3 | 2020-21 through 2024-25 | 2025-26 | "
            "1,900 | 380 |"
        ),
        "",
        "## Accuracy and probability quality",
        "",

                (
            "| Fold | Test season | Baseline accuracy | Model accuracy | "
            "Improvement | Log loss |"
        ),

        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in fold_metrics.itertuples(index=False):
        report_lines.append(
            f"| {row.fold} | {row.test_season} | "
            f"{row.baseline_accuracy:.1%} | "
            f"{row.model_accuracy:.1%} | "
            f"{row.accuracy_improvement:+.1%} | "
            f"{row.model_log_loss:.3f} |"
        )

    report_lines.extend(
        [
            "",
            "### Average results",
            "",
            (
                f"- Baseline accuracy: "
                f"**{average_baseline_accuracy:.1%}**"
            ),
            (
                f"- Random Forest accuracy: "
                f"**{average_model_accuracy:.1%}**"
            ),
            (
                f"- Accuracy improvement: "
                f"**{average_accuracy_improvement:+.1%}**"
            ),
            (
                f"- Random Forest log loss: "
                f"**{average_model_log_loss:.3f}**"
            ),
            "",
            "![Accuracy comparison](charts/accuracy_comparison.png)",
            "",
            "The Random Forest beat the baseline in all three folds.",
            (
                "However, accuracy decreased from 57.4% in 2023–24 to "
                "47.4% in 2025–26, showing that performance changes "
                "between seasons."
            ),
            "",
            "## Performance by result class",
            "",
            "| Result | Precision | Recall | F1 score |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    result_names = {
        "H": "Home win",
        "D": "Draw",
        "A": "Away win",
    }

    for row in average_class_metrics.itertuples(index=False):
        report_lines.append(
            f"| {result_names[row.result_class]} | "
            f"{row.precision:.3f} | {row.recall:.3f} | "
            f"{row.f1_score:.3f} |"
        )

    report_lines.extend(
        [
            "",
            "![Confusion matrices](charts/confusion_matrices.png)",
            "",
            (
                "Home wins were the strongest class. Away wins had "
                "moderate performance."
            ),
            (
                "Draws were the largest weakness: only 2 of 279 draws "
                "were predicted correctly across the three test seasons."
            ),
            (
                "Most actual draws were incorrectly classified as home "
                "or away wins."
            ),
            "",
            "## Probability calibration",
            "",
            "| Result | Average Brier score |",
            "| --- | ---: |",
        ]
    )

    for row in average_brier_scores.itertuples(index=False):
        report_lines.append(
            f"| {result_names[row.result_class]} | "
            f"{row.brier_score:.3f} |"
        )

    report_lines.extend(
        [
            "",
            "![Probability calibration](charts/probability_calibration.png)",
            "",
            (
                "Home-win and away-win probabilities generally followed "
                "the observed frequencies."
            ),
            (
                "Draw probabilities were usually low and rarely exceeded "
                "40%, which helps explain why the model almost never "
                "selected draws as its final prediction."
            ),
            "",
            "## Feature importance",
            "",
            "| Rank | Feature | Average importance | Average fold rank |",
            "| ---: | --- | ---: | ---: |",
        ]
    )

    for rank, row in enumerate(
        feature_importance_summary.head(10).itertuples(index=False),
        start=1,
    ):
        report_lines.append(
            f"| {rank} | `{row.feature}` | "
            f"{row.mean_importance:.4f} | {row.average_rank:.1f} |"
        )

    report_lines.extend(
        [
            "",
            "![Feature importance](charts/feature_importance.png)",
            "",
            (
                "Elo-based features were the most influential features "
                "in every fold."
            ),
            (
                "Feature importance shows what the Random Forest used "
                "most often, but it does not prove that a feature causes "
                "a match result."
            ),
            "",
            "## Main strengths",
            "",
            "- Beat the most-frequent-result baseline in every fold.",
            "- Performed best when identifying home wins.",
            "- Produced usable probabilities instead of only class labels.",
            "- Used stable Elo features across different seasons.",
            "",
            "## Main weaknesses",
            "",
            "- Almost never predicted draws correctly.",
            "- Accuracy and log loss became worse in newer seasons.",
            "- Some high-probability bins contained very few matches.",
            (
                "- The model does not yet include player availability, "
                "transfers, injuries, or promoted-team adjustments."
            ),
            "",
            "## Conclusion",
            "",
            (
                "The Random Forest is better than the simple baseline "
                "and provides useful starting probabilities."
            ),
            (
                "Before relying on it for the 2026–27 season, future "
                "work should focus on draw detection, probability "
                "calibration, and testing additional features."
            ),
            "",
            "## Saved evaluation data",
            "",
            "- [Fold metrics](tables/fold_metrics.csv)",
            "- [Class metrics](tables/class_metrics.csv)",
            "- [Confusion matrices](tables/confusion_matrices.csv)",
            "- [Brier scores](tables/brier_scores.csv)",
            "- [Calibration metrics](tables/calibration_metrics.csv)",
            (
                "- [Feature importance by fold]"
                "(tables/feature_importance_by_fold.csv)"
            ),
            (
                "- [Feature importance summary]"
                "(tables/feature_importance_summary.csv)"
            ),
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    return REPORT_PATH

def main() -> None:
    features, target, matches = load_model_data()
    folds = create_expanding_window_splits(features, target, matches)
    trained_folds = train_validation_folds(folds)

    fold_metrics = calculate_fold_metrics(trained_folds)
    class_metrics = calculate_class_metrics(trained_folds)
    confusion_matrices = calculate_confusion_matrices(trained_folds)
    brier_scores, calibration_metrics = (
        calculate_calibration_metrics(trained_folds)
    )
    feature_importance = calculate_feature_importance(trained_folds)
    feature_importance_summary = save_evaluation_tables(
        fold_metrics,
        class_metrics,
        confusion_matrices,
        brier_scores,
        calibration_metrics,
        feature_importance,
    )

    chart_paths = save_evaluation_charts(
        fold_metrics,
        confusion_matrices,
        calibration_metrics,
        feature_importance_summary,
    )

    report_path = generate_evaluation_report(
        fold_metrics,
        class_metrics,
        brier_scores,
        feature_importance_summary,
    )

    print(f"Validation folds created: {len(folds)}")
    print(f"Fresh models trained: {len(trained_folds)}")
    print(f"Fresh baselines trained: {len(trained_folds)}")

    display_columns = [
        "fold",
        "test_season",
        "baseline_accuracy",
        "model_accuracy",
        "accuracy_improvement",
        "model_log_loss",
    ]

    print("\nFold evaluation metrics:")
    print(
        fold_metrics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    average_class_metrics = (
        class_metrics.groupby("result_class", sort=False)[
            ["precision", "recall", "f1_score"]
        ]
        .mean()
        .reset_index()
    )

    print("\nAverage class performance:")
    print(
        average_class_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    average_brier_scores = (
        brier_scores.groupby("result_class", sort=False)["brier_score"]
        .mean()
        .reset_index()
    )

    print("\nAverage Brier scores:")
    print(
        average_brier_scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


    print("\nTop 15 features across all folds:")
    print(
        feature_importance_summary.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nEvaluation table sizes:")
    print(f"Fold metric rows: {len(fold_metrics)}")
    print(f"Class metric rows: {len(class_metrics)}")
    print(f"Confusion-matrix rows: {len(confusion_matrices)}")
    print(f"Brier-score rows: {len(brier_scores)}")
    print(f"Calibration rows: {len(calibration_metrics)}")
    print(f"Feature-importance rows: {len(feature_importance)}")
    print(f"\nSaved evaluation tables: {TABLES_DIR}")
    for table_path in sorted(TABLES_DIR.glob("*.csv")):
        print(f"- {table_path}")

    print(f"\nSaved evaluation charts: {CHARTS_DIR}")
    for chart_path in chart_paths:
        print(f"- {chart_path}")

    print(f"\nSaved evaluation report: {report_path}")


if __name__ == "__main__":
    main()