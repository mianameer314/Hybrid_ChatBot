"""
Redis Cache Configuration and Management
"""
import redis
import json
import pickle
from typing import Any, Optional, Union
import logging
from datetime import timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache manager"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except (redis.RedisError, ConnectionError, OSError) as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except redis.RedisError:
            return False
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache"""
        if not self.is_available():
            return False
        
        try:
            # Use pickle for complex objects, json for simple ones
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value).encode('utf-8')
            else:
                serialized_value = pickle.dumps(value)
            
            ttl = ttl or settings.CACHE_TTL
            self.redis_client.setex(key, ttl, serialized_value)
            return True
        except (redis.RedisError, json.JSONEncodeError, pickle.PickleError) as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        if not self.is_available():
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(value)
                
        except (redis.RedisError, pickle.PickleError) as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        if not self.is_available():
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except redis.RedisError as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        if not self.is_available():
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except redis.RedisError as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.is_available():
            return {"available": False}
        
        try:
            info = self.redis_client.info()
            return {
                "available": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except redis.RedisError as e:
            logger.error(f"Cache stats error: {e}")
            return {"available": False, "error": str(e)}

# Global cache instance
cache = RedisCache()

def get_cache() -> RedisCache:
    """Get cache instance"""
    return cache

# Cache key generators
def make_chat_history_key(session_id: str) -> str:
    """Generate cache key for chat history"""
    return f"chat_history:{session_id}"

def make_embedding_key(text: str, model: str) -> str:
    """Generate cache key for embeddings"""
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"embedding:{model}:{text_hash}"

def make_rag_key(query: str, docs_hash: str) -> str:
    """Generate cache key for RAG results"""
    import hashlib
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return f"rag:{query_hash}:{docs_hash}"
