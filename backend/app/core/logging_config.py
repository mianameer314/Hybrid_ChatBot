"""
Comprehensive Logging Configuration
"""
import logging
import logging.config
import sys
from datetime import datetime
import os
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file
    if not log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_file = f"{log_dir}/chatbot_{timestamp}.log"
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[{asctime}] {levelname:8s} {name:15s} | {message}",
                "style": "{",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "{levelname:8s} | {message}",
                "style": "{"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": f"{log_dir}/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": "DEBUG",
                "handlers": ["console", "file"]
            },
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            },
            "transformers": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            },
            "sentence_transformers": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set up root logger
    logger = logging.getLogger("app")
    logger.info("Logging configuration initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")

class StructuredLogger:
    """Structured logger for better log analysis"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"app.{name}")
    
    def log_api_request(self, endpoint: str, method: str, user_id: str = None, processing_time: float = None):
        """Log API request"""
        self.logger.info(
            f"API Request | {method} {endpoint} | User: {user_id or 'anonymous'} | "
            f"Time: {processing_time:.3f}s" if processing_time else f"Time: N/A"
        )
    
    def log_llm_call(self, provider: str, model: str, tokens: int = None, cost: float = None):
        """Log LLM API call"""
        self.logger.info(
            f"LLM Call | {provider}:{model} | Tokens: {tokens or 'N/A'} | "
            f"Cost: ${cost:.4f}" if cost else "Cost: N/A"
        )
    
    def log_rag_query(self, query: str, results_count: int, processing_time: float):
        """Log RAG query"""
        self.logger.info(
            f"RAG Query | Query: '{query[:50]}...' | Results: {results_count} | "
            f"Time: {processing_time:.3f}s"
        )
    
    def log_sentiment_analysis(self, text: str, sentiment: str, confidence: float, model: str):
        """Log sentiment analysis"""
        self.logger.info(
            f"Sentiment Analysis | Text: '{text[:30]}...' | Sentiment: {sentiment} | "
            f"Confidence: {confidence:.3f} | Model: {model}"
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        context_str = f" | Context: {context}" if context else ""
        self.logger.error(f"Error: {str(error)} | Type: {type(error).__name__}{context_str}", exc_info=True)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        extra_info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        self.logger.info(f"Performance | {operation} | Duration: {duration:.3f}s | {extra_info}")

# Performance monitoring decorator
def log_performance(operation_name: str):
    """Decorator to log function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(operation_name, duration, status="success")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(operation_name, duration, status="error", error=str(e))
                logger.log_error(e)
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(operation_name, duration, status="success")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(operation_name, duration, status="error", error=str(e))
                logger.log_error(e)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
