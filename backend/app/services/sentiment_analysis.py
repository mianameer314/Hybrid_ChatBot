"""
Advanced Sentiment Analysis using HuggingFace Transformers
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from typing import Dict, Any, List, Optional
import asyncio
from functools import lru_cache
import numpy as np

from app.core.config import settings
from app.core.cache import cache, make_embedding_key

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis using HuggingFace transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.SENTIMENT_MODEL
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.emotion_classifier = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the sentiment analysis models"""
        if self.is_initialized:
            return
        
        try:
            logger.info(f"Loading sentiment analysis model: {self.model_name}")
            
            # Load main sentiment classifier
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Load emotion classification model
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                logger.info("Emotion classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load emotion classifier: {e}")
                self.emotion_classifier = None
            
            self.is_initialized = True
            logger.info("Sentiment analysis models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analysis: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        Returns: {
            'label': 'positive|negative|neutral',
            'score': float,
            'confidence': float,
            'all_scores': [...],
            'emotion': str,
            'emotion_score': float
        }
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = f"sentiment:{self.model_name}:{hash(text)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Run sentiment analysis
            sentiment_results = await asyncio.to_thread(self.classifier, text)
            
            # Process results
            if isinstance(sentiment_results[0], list):
                scores = sentiment_results[0]
            else:
                scores = [sentiment_results[0]]
            
            # Find best sentiment
            best_sentiment = max(scores, key=lambda x: x['score'])
            
            # Normalize labels
            label = self._normalize_sentiment_label(best_sentiment['label'])
            
            result = {
                'label': label,
                'score': best_sentiment['score'],
                'confidence': best_sentiment['score'],
                'all_scores': scores,
                'model': self.model_name,
                'emotion': None,
                'emotion_score': None
            }
            
            # Add emotion analysis if available
            if self.emotion_classifier:
                try:
                    emotion_results = await asyncio.to_thread(self.emotion_classifier, text)
                    if isinstance(emotion_results[0], list):
                        emotion_scores = emotion_results[0]
                    else:
                        emotion_scores = [emotion_results[0]]
                    
                    best_emotion = max(emotion_scores, key=lambda x: x['score'])
                    result['emotion'] = best_emotion['label']
                    result['emotion_score'] = best_emotion['score']
                    
                except Exception as e:
                    logger.warning(f"Emotion analysis failed: {e}")
            
            # Cache result
            cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            # Return default neutral sentiment
            return {
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'all_scores': [],
                'model': self.model_name,
                'emotion': None,
                'emotion_score': None,
                'error': str(e)
            }
    
    def _normalize_sentiment_label(self, label: str) -> str:
        """Normalize sentiment labels to consistent format"""
        label_lower = label.lower()
        
        # Map various sentiment labels to standard format
        positive_labels = ['positive', 'pos', 'label_2', '2']
        negative_labels = ['negative', 'neg', 'label_0', '0']
        neutral_labels = ['neutral', 'neu', 'label_1', '1']
        
        if label_lower in positive_labels:
            return 'positive'
        elif label_lower in negative_labels:
            return 'negative'
        elif label_lower in neutral_labels:
            return 'neutral'
        else:
            return 'neutral'  # Default fallback
    
    async def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""
        if not self.is_initialized:
            await self.initialize()
        
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_sentiment_summary(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for a list of sentiment analyses"""
        if not sentiments:
            return {}
        
        labels = [s['label'] for s in sentiments]
        scores = [s['score'] for s in sentiments]
        emotions = [s.get('emotion') for s in sentiments if s.get('emotion')]
        
        return {
            'total_messages': len(sentiments),
            'positive_count': labels.count('positive'),
            'negative_count': labels.count('negative'),
            'neutral_count': labels.count('neutral'),
            'average_score': np.mean(scores) if scores else 0.0,
            'score_std': np.std(scores) if len(scores) > 1 else 0.0,
            'dominant_sentiment': max(set(labels), key=labels.count) if labels else 'neutral',
            'emotions_distribution': {emotion: emotions.count(emotion) for emotion in set(emotions)} if emotions else {}
        }

# Alternative simple sentiment analyzer using VADER (fallback)
class VADERSentimentAnalyzer:
    """Simple sentiment analyzer using VADER (backup option)"""
    
    def __init__(self):
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Download required VADER data
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.analyzer = SentimentIntensityAnalyzer()
            self.is_initialized = True
            logger.info("VADER sentiment analyzer initialized")
        except ImportError:
            logger.warning("NLTK not available, VADER analyzer not initialized")
            self.is_initialized = False
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER"""
        if not self.is_initialized:
            return {
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'error': 'VADER not available'
            }
        
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Determine label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'label': label,
                'score': abs(compound),
                'confidence': abs(compound),
                'all_scores': scores,
                'model': 'vader'
            }
        except Exception as e:
            logger.error(f"VADER sentiment analysis error: {e}")
            return {
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }

# Global sentiment analyzer instances
hf_sentiment_analyzer = SentimentAnalyzer()
vader_analyzer = VADERSentimentAnalyzer()

async def analyze_sentiment(text: str, use_hf: bool = True) -> Dict[str, Any]:
    """
    Main sentiment analysis function
    
    Args:
        text: Text to analyze
        use_hf: Whether to use HuggingFace (True) or VADER (False)
    
    Returns:
        Sentiment analysis result
    """
    try:
        if use_hf:
            return await hf_sentiment_analyzer.analyze_sentiment(text)
        else:
            return await vader_analyzer.analyze_sentiment(text)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        # Fallback to VADER if HuggingFace fails
        if use_hf:
            logger.info("Falling back to VADER sentiment analysis")
            return await vader_analyzer.analyze_sentiment(text)
        else:
            # Return neutral if all fails
            return {
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
