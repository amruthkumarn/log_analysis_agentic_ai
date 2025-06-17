import os
import redis

def get_redis_client():
    """
    Initializes and returns a Redis client using a consistent URL from environment variables.
    """
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(redis_url)

def get_redis_url():
    """
    Returns the Redis URL from environment variables for libraries that require the URL directly.
    """
    return os.getenv("REDIS_URL", "redis://redis:6379/0") 