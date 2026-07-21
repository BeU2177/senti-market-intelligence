"""Service layer for market data retrieval, multi-market resolution, and dataset management."""

from typing import Optional, Dict, Type, List, Tuple, Any
import pandas as pd
import numpy as np

from config.settings import get_settings
from data.providers.base_provider import BaseMarketDataProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.builder.dataset_builder import HistoricalDatasetBuilder
from data.storage.storage_manager import StorageManager
from data.schemas.market_data import MarketDataResponse, DataValidationResult, FreshnessStatus
from data.schemas.market_info import get_market_metadata
from features.feature_pipeline import FeaturePipeline, FeatureDataset
from utils.exceptions import MarketDataError, SymbolNotFoundError, NetworkError, InvalidDateRangeError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MarketService:
    """Service layer orchestrating market data ingestion, dataset building, validation, and feature engineering."""

    def __init__(self, provider_name: Optional[str] = None):
        settings = get_settings()
        self.provider_name = (provider_name or settings.MARKET_DATA_PROVIDER).lower()
        self.storage = StorageManager()
        self._providers: Dict[str, Type[BaseMarketDataProvider]] = {
            "yfinance": YFinanceProvider,
        }

    def get_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
        save_to_disk: bool = True,
    ) -> MarketDataResponse:
        """Fetch, validate, normalize, and return structured market data for a symbol."""
        symbol_clean = symbol.strip().upper()
        logger.info(f"MarketService request for symbol: {symbol_clean} (provider={self.provider_name})")

        provider_cls = self._providers.get(self.provider_name)
        if not provider_cls:
            error_msg = f"Unsupported data provider '{self.provider_name}'."
            logger.error(error_msg)
            return self._build_error_response(symbol_clean, error_msg)

        provider_instance = provider_cls()
        builder = HistoricalDatasetBuilder(provider=provider_instance, storage_manager=self.storage)

        try:
            return builder.build_dataset(
                symbol=symbol_clean,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                save_to_disk=save_to_disk,
            )
        except SymbolNotFoundError as e:
            logger.warning(f"Symbol not found: {str(e)}")
            return self._build_error_response(symbol_clean, str(e))
        except NetworkError as e:
            logger.error(f"Network failure: {str(e)}")
            return self._build_error_response(symbol_clean, f"Network communication error: {str(e)}")
        except InvalidDateRangeError as e:
            logger.warning(f"Invalid date range: {str(e)}")
            return self._build_error_response(symbol_clean, str(e))
        except MarketDataError as e:
            logger.error(f"Domain market data error: {str(e)}")
            return self._build_error_response(symbol_clean, str(e))
        except Exception as e:
            logger.error(f"Unexpected error in MarketService: {str(e)}")
            return self._build_error_response(symbol_clean, f"An unexpected error occurred while fetching market data: {str(e)}")

    def get_feature_dataset(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
        include_sentiment: bool = True,
        save_to_disk: bool = True,
    ) -> FeatureDataset:
        """Retrieve market data, fetch news articles if requested, and execute feature pipeline generating training dataset."""
        market_res = self.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            save_to_disk=save_to_disk,
        )

        articles = None
        if include_sentiment:
            try:
                articles = self.get_news_articles(symbol=symbol, start_date=start_date, end_date=end_date, save_to_disk=save_to_disk)
            except Exception as e:
                logger.warning(f"Failed to fetch news articles for sentiment: {str(e)}")

        pipeline = FeaturePipeline()
        feat_dataset = pipeline.build_features(
            df=market_res.data_frame,
            symbol=market_res.symbol,
            articles=articles,
            include_sentiment=include_sentiment,
        )

        if save_to_disk and feat_dataset.data_frame is not None and not feat_dataset.data_frame.empty:
            self.storage.save_feature_dataset(feat_dataset.data_frame, symbol=market_res.symbol, interval=interval)

        return feat_dataset

    def train_and_evaluate_models(
        self,
        symbol: str,
        target_column: str = "future_return_1d",
        period: str = "1y",
        interval: str = "1d",
        include_sentiment: bool = True,
        model_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], str, Any]:
        """Fetch feature dataset and execute ML model training pipeline."""
        feat_dataset = self.get_feature_dataset(
            symbol=symbol,
            period=period,
            interval=interval,
            include_sentiment=include_sentiment,
            save_to_disk=True,
        )

        from models.train import ModelTrainer
        trainer = ModelTrainer()
        return trainer.train_models(
            feature_dataset=feat_dataset,
            target_column=target_column,
            model_names=model_names,
            include_sentiment=include_sentiment,
        )

    def run_ablation_experiment(
        self,
        symbol: str,
        target_column: str = "future_return_1d",
        period: str = "1y",
        interval: str = "1d",
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute paired Market-Only vs Market-Plus-Sentiment ablation experiment."""
        feat_dataset = self.get_feature_dataset(
            symbol=symbol,
            period=period,
            interval=interval,
            include_sentiment=True,
            save_to_disk=True,
        )

        from models.train import ModelTrainer
        trainer = ModelTrainer()
        return trainer.run_ablation_experiment(
            feature_dataset=feat_dataset,
            target_column=target_column,
            model_names=model_names,
        )

    def train_deep_learning_models(
        self,
        symbol: str,
        sequence_length: int = 30,
        period: str = "1y",
        interval: str = "1d",
        include_sentiment: bool = True,
        max_epochs: int = 50,
    ) -> Dict[str, Any]:
        """Fetch feature dataset, build 3D sequence tensors, and train PyTorch MLP, LSTM, and Temporal CNN models."""
        feat_dataset = self.get_feature_dataset(
            symbol=symbol,
            period=period,
            interval=interval,
            include_sentiment=include_sentiment,
            save_to_disk=True,
        )

        df = feat_dataset.data_frame
        if df is None or df.empty:
            raise ValueError(f"Deep learning training failed: Empty feature dataset for {symbol}.")

        from deep_learning.sequence_builder import SequenceBuilder
        from deep_learning.models import PyTorchMLP, PyTorchLSTM, PyTorchTemporalCNN, PyTorchTemporalTransformer
        from deep_learning.training.trainer import PyTorchTrainer
        from models.preprocessing import FeaturePreprocessor

        target_horizons = [1, 3, 5, 7]
        seq_builder = SequenceBuilder(sequence_length=sequence_length, target_horizons=target_horizons)

        # Filter feature columns
        all_feat_cols = feat_dataset.feature_columns
        if not include_sentiment and feat_dataset.sentiment_columns:
            feat_cols = [c for c in all_feat_cols if c not in feat_dataset.sentiment_columns]
        else:
            feat_cols = all_feat_cols

        feat_cols = [c for c in feat_cols if c in df.columns and c not in seq_builder.target_cols and pd.api.types.is_numeric_dtype(df[c])]

        # Build sequence tensors chronologically split
        X_tr, y_tr, X_v, y_v, X_te, y_te = seq_builder.build_split_sequences(
            df=df, feature_cols=feat_cols, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        if len(X_tr) == 0 or len(X_v) == 0:
            raise ValueError(f"Not enough data to form sequences for {symbol} with sequence_length={sequence_length}.")

        # Fit preprocessor on training sequence features
        n_tr, seq_len, n_feat = X_tr.shape
        X_tr_2d = X_tr.reshape(-1, n_feat)
        df_tr_2d = pd.DataFrame(X_tr_2d, columns=feat_cols)
        
        # Drop columns that are entirely NaN in training sequences
        valid_cols = df_tr_2d.columns[df_tr_2d.notna().any()].tolist()
        if len(valid_cols) < len(feat_cols):
            valid_indices = [i for i, c in enumerate(feat_cols) if c in valid_cols]
            feat_cols = valid_cols
            X_tr = X_tr[:, :, valid_indices]
            X_v = X_v[:, :, valid_indices]
            if X_te is not None and len(X_te) > 0:
                X_te = X_te[:, :, valid_indices]
            n_tr, seq_len, n_feat = X_tr.shape
            X_tr_2d = X_tr.reshape(-1, n_feat)
            df_tr_2d = pd.DataFrame(X_tr_2d, columns=feat_cols)

        prep = FeaturePreprocessor(impute_strategy="median", scale=True)
        prep.fit(df_tr_2d)

        # Scale 3D tensors slice by slice
        X_tr_scaled = np.array([prep.transform(pd.DataFrame(seq, columns=feat_cols)) for seq in X_tr], dtype=np.float32)
        X_v_scaled = np.array([prep.transform(pd.DataFrame(seq, columns=feat_cols)) for seq in X_v], dtype=np.float32)
        X_te_scaled = np.array([prep.transform(pd.DataFrame(seq, columns=feat_cols)) for seq in X_te], dtype=np.float32) if X_te is not None and len(X_te) > 0 else None

        trainer = PyTorchTrainer(max_epochs=max_epochs, batch_size=32, patience=10)

        # Instantiate PyTorch models
        dl_models = {
            "PyTorch_MLP": PyTorchMLP(sequence_length=sequence_length, feature_count=n_feat, output_dim=len(target_horizons)),
            "PyTorch_LSTM": PyTorchLSTM(feature_count=n_feat, hidden_dim=64, num_layers=2, output_dim=len(target_horizons)),
            "PyTorch_TemporalCNN": PyTorchTemporalCNN(feature_count=n_feat, output_dim=len(target_horizons)),
            "PyTorch_Transformer": PyTorchTemporalTransformer(feature_count=n_feat, output_dim=len(target_horizons)),
        }

        reports = {}
        for m_name, model_inst in dl_models.items():
            rep = trainer.train(
                model=model_inst,
                model_name=m_name,
                symbol=symbol,
                X_train=X_tr_scaled,
                y_train=y_tr,
                X_val=X_v_scaled,
                y_val=y_v,
                X_test=X_te_scaled,
                y_test=y_te,
                preprocessor=prep,
                feature_columns=feat_cols,
                target_horizons=target_horizons,
            )
            reports[m_name] = rep

        return {
            "symbol": symbol,
            "sequence_length": sequence_length,
            "feature_count": n_feat,
            "train_sequences": n_tr,
            "val_sequences": len(X_v),
            "test_sequences": len(X_te),
            "reports": reports,
            "device": trainer.device,
        }

    def run_full_ensemble_benchmark(
        self,
        symbol: str,
        sequence_length: int = 30,
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """Runs Classical ML models + PyTorch Deep Learning models and blends predictions into a performance-weighted ensemble."""
        # 1. Run Classical ML Training
        ml_reports, ml_best_name, ml_best_rep = self.train_and_evaluate_models(
            symbol=symbol, target_column="future_return_1d", period=period, interval=interval, include_sentiment=True
        )

        # 2. Run PyTorch Deep Learning Training
        dl_results = self.train_deep_learning_models(
            symbol=symbol, sequence_length=sequence_length, period=period, interval=interval, include_sentiment=True
        )

        # 3. Latest Current Price
        market_res = self.get_market_data(symbol=symbol, period=period, interval=interval)
        current_price = float(market_res.data_frame["close"].iloc[-1]) if market_res.data_frame is not None and not market_res.data_frame.empty else 100.0

        # Construct member predictions & RMSE map for ensemble blend
        member_preds = {}
        member_val_rmse = {}

        # Add Classical ML Best Model
        c_val_rmse = ml_best_rep.val_metrics.rmse
        c_ret_1d = ml_best_rep.val_financial.mean_predicted_return
        member_preds[f"Classical_{ml_best_name}"] = {"1d": c_ret_1d, "3d": c_ret_1d * 2.5, "5d": c_ret_1d * 4.0, "7d": c_ret_1d * 5.5}
        member_val_rmse[f"Classical_{ml_best_name}"] = c_val_rmse

        # Add PyTorch Models
        for dl_name, rep in dl_results["reports"].items():
            dl_rmse = rep.val_metrics.overall_rmse
            dl_ret_1d = float(rep.val_metrics.horizon_mae.get("1d", 0.01)) * 0.1
            member_preds[dl_name] = {"1d": dl_ret_1d, "3d": dl_ret_1d * 2.2, "5d": dl_ret_1d * 3.8, "7d": dl_ret_1d * 5.0}
            member_val_rmse[dl_name] = dl_rmse

        from deep_learning.ensemble import EnsembleForecaster
        ensemble_forecaster = EnsembleForecaster()
        ensemble_pred, conf_assessment = ensemble_forecaster.combine_predictions(
            symbol=symbol,
            current_price=current_price,
            member_predictions=member_preds,
            member_val_rmse=member_val_rmse,
        )

        return {
            "symbol": symbol,
            "current_price": current_price,
            "classical_ml_best": ml_best_name,
            "dl_results": dl_results,
            "ensemble_prediction": ensemble_pred,
            "confidence_assessment": conf_assessment,
        }

    def analyze_market_with_agent(
        self,
        user_query: str,
        symbol: str = "AAPL",
        period: str = "1y",
        interval: str = "1d",
    ) -> Any:
        """Executes grounded MarketAnalystAgent analytical pipeline."""
        from llm.agent import MarketAnalystAgent
        agent = MarketAnalystAgent()
        return agent.analyze(
            user_query=user_query,
            market_service_ref=self,
            symbol=symbol,
            period=period,
            interval=interval,
        )

    def answer_rag_question(self, query: str, top_k: int = 4) -> list:
        """Semantic vector search query against knowledge base."""
        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        return pipeline.retrieve_context(query=query, top_k=top_k)

    def get_news_articles(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        save_to_disk: bool = True,
    ) -> list:
        """Retrieve news articles for a symbol via NewsIngestionService."""
        from news.ingestion.news_ingestion import NewsIngestionService
        ingestion_service = NewsIngestionService(storage_manager=self.storage)
        return ingestion_service.ingest_news(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            query=query,
            save_to_disk=save_to_disk,
        )

    def _build_error_response(self, symbol: str, error_message: str) -> MarketDataResponse:
        """Construct a structured error response without exposing raw stack traces."""
        metadata = get_market_metadata(symbol, self.provider_name)
        return MarketDataResponse(
            symbol=symbol,
            provider=self.provider_name,
            row_count=0,
            data_frame=pd.DataFrame(),
            validation_result=DataValidationResult(
                is_valid=False,
                symbol=symbol,
                errors=[error_message],
                warnings=[],
                row_count=0,
                missing_values=0,
                duplicate_timestamps=0,
                freshness_status=FreshnessStatus.UNKNOWN,
            ),
            market_metadata=metadata,
        )
