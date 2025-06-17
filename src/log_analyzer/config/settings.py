"""
Configuration settings for the log analyzer package.
"""

import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analysis_output"
DOCUMENTATION_DIR = BASE_DIR / "documentation"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)

# Analysis output settings
ANALYSIS_RETENTION_DAYS = int(os.getenv("ANALYSIS_RETENTION_DAYS", "7"))
ANALYSIS_OUTPUT_FORMAT = "%Y%m%d_%H%M%S"

def get_analysis_output_dir() -> Path:
    """Get the current analysis output directory with timestamp."""
    timestamp = datetime.now().strftime(ANALYSIS_OUTPUT_FORMAT)
    output_dir = ANALYSIS_OUTPUT_DIR / timestamp
    output_dir.mkdir(exist_ok=True)
    return output_dir

def get_analysis_file_path(session_id: str, analysis_type: str) -> Path:
    """Get the path for an analysis output file."""
    output_dir = get_analysis_output_dir()
    timestamp = datetime.now().strftime(ANALYSIS_OUTPUT_FORMAT)
    return output_dir / f"{analysis_type}_{session_id}_{timestamp}.json"

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Elasticsearch settings
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "demo-logs")

# Analysis settings
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT = 30

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        "base_dir": str(BASE_DIR),
        "logs_dir": str(LOGS_DIR),
        "analysis_output_dir": str(ANALYSIS_OUTPUT_DIR),
        "documentation_dir": str(DOCUMENTATION_DIR),
        "analysis": {
            "retention_days": ANALYSIS_RETENTION_DAYS,
            "output_format": ANALYSIS_OUTPUT_FORMAT,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_retries": MAX_RETRIES,
            "timeout": TIMEOUT,
        },
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": REDIS_DB,
        },
        "elasticsearch": {
            "host": ELASTICSEARCH_HOST,
            "port": ELASTICSEARCH_PORT,
            "index": ELASTICSEARCH_INDEX,
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT,
        },
        "llm": {
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
    } 