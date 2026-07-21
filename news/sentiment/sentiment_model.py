"""FinBERT NLP Financial Sentiment Analysis Model."""

from functools import lru_cache
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from news.schemas.news_article import NewsArticle, ArticleSentiment
from utils.logging_config import get_logger

logger = get_logger(__name__)

FINBERT_MODEL_NAME = "ProsusAI/finbert"


class FinBERTSentimentModel:
    """Financial sentiment model running FinBERT sequence classification with CPU/GPU auto-selection."""

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._fallback_mode = False

        self._load_model()

    def _load_model(self) -> None:
        """Load pretrained FinBERT tokenizer and weights, falling back to rule-based lexicon if offline."""
        logger.info(f"Loading FinBERT model '{self.model_name}' on device '{self.device}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT from HuggingFace ({str(e)}). Activating lightweight lexicon fallback.")
            self._fallback_mode = True

    def predict_sentiment_batch(self, articles: List[NewsArticle], batch_size: int = 16) -> List[ArticleSentiment]:
        """Runs batch sentiment inference across a list of articles."""
        if not articles:
            return []

        if self._fallback_mode:
            return [self._predict_lexicon_fallback(art) for art in articles]

        sentiments: List[ArticleSentiment] = []
        texts = [f"{art.title}. {art.description or ''}".strip() for art in articles]

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_articles = articles[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

                # FinBERT label order: positive (0), negative (1), neutral (2)
                for idx, (prob_vec, art) in enumerate(zip(probs, batch_articles)):
                    p_pos = float(prob_vec[0])
                    p_neg = float(prob_vec[1])
                    p_neu = float(prob_vec[2])

                    labels = ["positive", "negative", "neutral"]
                    max_idx = int(prob_vec.argmax())
                    pred_label = labels[max_idx]
                    confidence = float(prob_vec[max_idx])

                    # Formula: sentiment_score = positive_prob - negative_prob in [-1, 1]
                    sentiment_score = round(p_pos - p_neg, 4)

                    sentiments.append(
                        ArticleSentiment(
                            article_id=art.article_id,
                            published_at=art.published_at,
                            symbol=art.symbol,
                            positive_probability=round(p_pos, 4),
                            negative_probability=round(p_neg, 4),
                            neutral_probability=round(p_neu, 4),
                            predicted_label=pred_label,
                            sentiment_score=sentiment_score,
                            confidence=round(confidence, 4),
                        )
                    )

            return sentiments

        except Exception as e:
            logger.error(f"Error during FinBERT batch inference: {str(e)}. Falling back to lexicon scoring.")
            return [self._predict_lexicon_fallback(art) for art in articles]

    def _predict_lexicon_fallback(self, article: NewsArticle) -> ArticleSentiment:
        """Lightweight rule-based financial lexicon fallback when HuggingFace is unreachable."""
        text = f"{article.title} {article.description or ''}".lower()

        pos_words = {"growth", "profit", "gain", "rise", "surge", "up", "bullish", "record", "revenue", "positive", "beat", "higher", "climb"}
        neg_words = {"loss", "decline", "fall", "drop", "down", "bearish", "risk", "debt", "slash", "plunge", "cut", "warning", "crash", "negative"}

        words = set(text.split())
        pos_count = len(words.intersection(pos_words))
        neg_count = len(words.intersection(neg_words))

        if pos_count > neg_count:
            p_pos, p_neg, p_neu = 0.70, 0.10, 0.20
            label = "positive"
        elif neg_count > pos_count:
            p_pos, p_neg, p_neu = 0.10, 0.70, 0.20
            label = "negative"
        else:
            p_pos, p_neg, p_neu = 0.15, 0.15, 0.70
            label = "neutral"

        return ArticleSentiment(
            article_id=article.article_id,
            published_at=article.published_at,
            symbol=article.symbol,
            positive_probability=p_pos,
            negative_probability=p_neg,
            neutral_probability=p_neu,
            predicted_label=label,
            sentiment_score=round(p_pos - p_neg, 4),
            confidence=max(p_pos, p_neg, p_neu),
        )


@lru_cache(maxsize=1)
def get_sentiment_model() -> FinBERTSentimentModel:
    """Returns a singleton resource-cached FinBERTSentimentModel instance."""
    return FinBERTSentimentModel()
