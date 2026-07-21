"""Model training orchestrator performing temporal data splitting, preprocessing, baseline benchmarking, ablation testing, and artifact serialization."""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from features.feature_pipeline import FeatureDataset
from models.data_splitter import TimeSplitter, WalkForwardSplitter
from models.preprocessing import FeaturePreprocessor
from models.model_registry import ModelRegistry
from models.evaluate import ModelEvaluator
from models.artifacts import ArtifactManager
from models.schemas import (
    EvaluationReport,
    ExperimentRecord,
    ModelArtifactMetadata,
    RegressionMetrics,
    FinancialMetrics,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrates machine learning training, walk-forward validation, ablation testing, and model selection."""

    def __init__(
        self,
        random_seed: int = 42,
        artifact_manager: Optional[ArtifactManager] = None,
    ):
        self.random_seed = random_seed
        self.model_registry = ModelRegistry(random_seed=random_seed)
        self.evaluator = ModelEvaluator()
        self.artifacts = artifact_manager or ArtifactManager()
        self.splitter = TimeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    def train_models(
        self,
        feature_dataset: FeatureDataset,
        target_column: str = "future_return_1d",
        model_names: Optional[List[str]] = None,
        include_sentiment: bool = True,
    ) -> Tuple[Dict[str, EvaluationReport], str, EvaluationReport]:
        """Trains baseline & ML models, evaluates validation metrics, selects best model, and evaluates test set."""
        start_time = time.time()
        symbol = feature_dataset.symbol
        df = feature_dataset.data_frame

        logger.info(f"ModelTrainer starting for {symbol} (target='{target_column}', include_sentiment={include_sentiment})")

        # 1. Dataset MLOps Validation
        if df is None or df.empty:
            raise ValueError(f"Training failed: Feature dataset for {symbol} is empty.")
        if target_column not in df.columns:
            raise ValueError(f"Training failed: Target column '{target_column}' not found in dataset columns.")

        # Determine feature columns
        all_feature_cols = feature_dataset.feature_columns
        if not include_sentiment and feature_dataset.sentiment_columns:
            feature_cols = [c for c in all_feature_cols if c not in feature_dataset.sentiment_columns]
            feature_set_type = "market_only"
        else:
            feature_cols = all_feature_cols
            feature_set_type = "market_plus_sentiment"

        # One-hot encode string/categorical columns (e.g. market_regime) or convert to numeric
        df_encoded = df.copy()
        str_cols = [c for c in feature_cols if c in df_encoded.columns and not pd.api.types.is_numeric_dtype(df_encoded[c])]
        if str_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=str_cols, drop_first=True)
            # Re-derive feature_cols from encoded columns
            raw_cols = set(feature_dataset.data_frame.columns)
            target_cols = set(feature_dataset.target_columns)
            feature_cols = [c for c in df_encoded.columns if c not in raw_cols and c not in target_cols]
            # Also include any numeric feature columns that remained
            for c in df_encoded.columns:
                if any(c.startswith(sc + "_") for sc in str_cols):
                    feature_cols.append(c)
            feature_cols = list(set(feature_cols))

        feature_cols = [c for c in feature_cols if c in df_encoded.columns and c != target_column and pd.api.types.is_numeric_dtype(df_encoded[c])]
        if not feature_cols:
            raise ValueError(f"Training failed: No valid numeric feature columns found for feature set type '{feature_set_type}'.")

        df = df_encoded
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = self.splitter.split(
            df=df, feature_cols=feature_cols, target_col=target_column
        )

        # 3. Preprocessing Fitting STRICTLY on Training Set
        preprocessor = FeaturePreprocessor(impute_strategy="median", scale=True)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        # 4. Determine Models to Train
        selected_models = model_names or self.model_registry.get_available_models()

        reports: Dict[str, EvaluationReport] = {}
        best_model_name = None
        best_val_rmse = float("inf")
        best_val_dir_acc = -1.0
        best_model_instance = None

        # 5. Train & Evaluate Each Registered Model
        for m_name in selected_models:
            try:
                estimator = self.model_registry.get_model(m_name)

                # Fit estimator on X_train_proc
                estimator.fit(X_train_proc, y_train.values)

                # Predict on Train and Validation sets
                train_preds = estimator.predict(X_train_proc)
                val_preds = estimator.predict(X_val_proc)

                # Calculate Train & Val Metrics
                tr_reg = self.evaluator.evaluate_regression(y_train.values, train_preds)
                tr_fin = self.evaluator.evaluate_financial(y_train.values, train_preds)
                val_reg = self.evaluator.evaluate_regression(y_val.values, val_preds)
                val_fin = self.evaluator.evaluate_financial(y_val.values, val_preds)

                report = EvaluationReport(
                    model_name=m_name,
                    target_name=target_column,
                    feature_set_type=feature_set_type,
                    train_metrics=tr_reg,
                    train_financial=tr_fin,
                    val_metrics=val_reg,
                    val_financial=val_fin,
                )
                reports[m_name] = report

                # Selection Priority: Lower Validation RMSE, Higher Validation Directional Accuracy
                if val_reg.rmse < best_val_rmse or (np.isclose(val_reg.rmse, best_val_rmse) and val_fin.directional_accuracy > best_val_dir_acc):
                    best_val_rmse = val_reg.rmse
                    best_val_dir_acc = val_fin.directional_accuracy
                    best_model_name = m_name
                    best_model_instance = estimator

            except Exception as e:
                logger.error(f"Failed to train model '{m_name}' for {symbol}: {str(e)}")

        if not best_model_name or best_model_instance is None:
            raise RuntimeError(f"Training failed: No models were successfully trained for {symbol}.")

        # 6. Evaluate Best Model on UNTOUCHED Test Set
        best_test_preds = best_model_instance.predict(X_test_proc)
        test_reg = self.evaluator.evaluate_regression(y_test.values, best_test_preds)
        test_fin = self.evaluator.evaluate_financial(y_test.values, best_test_preds)

        best_report = reports[best_model_name]
        best_report.test_metrics = test_reg
        best_report.test_financial = test_fin

        exec_duration = round(time.time() - start_time, 2)

        # 7. Log Experiment Record
        exp_record = ExperimentRecord(
            experiment_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            target=target_column,
            feature_set_type=feature_set_type,
            feature_version=feature_dataset.feature_version,
            sentiment_version=feature_dataset.sentiment_version,
            model_name=best_model_name,
            hyperparameters={"random_seed": self.random_seed},
            train_row_count=len(X_train_raw),
            val_row_count=len(X_val_raw),
            test_row_count=len(X_test_raw),
            metrics={
                "val_rmse": best_report.val_metrics.rmse,
                "val_mae": best_report.val_metrics.mae,
                "val_directional_accuracy": best_report.val_financial.directional_accuracy,
                "test_rmse": test_reg.rmse,
                "test_directional_accuracy": test_fin.directional_accuracy,
            },
            execution_duration_sec=exec_duration,
            random_seed=self.random_seed,
        )
        self.artifacts.log_experiment(exp_record)

        # 8. Save Best Model Artifact Bundle
        artifact_meta = ModelArtifactMetadata(
            symbol=symbol,
            model_name=best_model_name,
            target_name=target_column,
            feature_set_type=feature_set_type,
            feature_columns=feature_cols,
            feature_version=feature_dataset.feature_version,
            sentiment_version=feature_dataset.sentiment_version,
            val_rmse=best_report.val_metrics.rmse,
            val_directional_accuracy=best_report.val_financial.directional_accuracy,
        )
        self.artifacts.save_model_artifact(
            estimator=best_model_instance,
            preprocessor=preprocessor,
            metadata=artifact_meta,
        )

        logger.info(
            f"ModelTrainer selected best model '{best_model_name}' for {symbol}: "
            f"Val RMSE={best_report.val_metrics.rmse}, Val DirAcc={best_report.val_financial.directional_accuracy}, "
            f"Test RMSE={test_reg.rmse}, Test DirAcc={test_fin.directional_accuracy}"
        )

        return reports, best_model_name, best_report

    def run_ablation_experiment(
        self,
        feature_dataset: FeatureDataset,
        target_column: str = "future_return_1d",
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs paired ablation experiments comparing Market-Only vs Market-Plus-Sentiment features."""
        logger.info(f"Running Ablation Experiment for {feature_dataset.symbol} on target '{target_column}'")

        # Experiment A: Market Only
        reports_a, best_name_a, best_rep_a = self.train_models(
            feature_dataset=feature_dataset,
            target_column=target_column,
            model_names=model_names,
            include_sentiment=False,
        )

        # Experiment B: Market Plus Sentiment
        reports_b, best_name_b, best_rep_b = self.train_models(
            feature_dataset=feature_dataset,
            target_column=target_column,
            model_names=model_names,
            include_sentiment=True,
        )

        return {
            "symbol": feature_dataset.symbol,
            "target": target_column,
            "market_only": {
                "reports": reports_a,
                "best_model": best_name_a,
                "best_report": best_rep_a,
            },
            "market_plus_sentiment": {
                "reports": reports_b,
                "best_model": best_name_b,
                "best_report": best_rep_b,
            },
        }
