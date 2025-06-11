import argparse
import os
import sys
from datetime import datetime, timezone, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.redis_log_analysis_agent import main as analysis_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full log analysis pipeline.")
    
    # Arguments for log source
    parser.add_argument(
        "--log-source", 
        type=str, 
        default=os.getenv("LOG_SOURCE", "file"), 
        choices=["file", "elk"],
        help="Source of the logs ('file' or 'elk')."
    )

    # Arguments for file-based logs
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=os.getenv("LOG_DIR", "logs/"),
        help="Directory containing log files if source is 'file'."
    )
    
    # Arguments for Elasticsearch
    parser.add_argument(
        "--elk-host", 
        type=str, 
        default=os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200"),
        help="Elasticsearch host URL."
    )
    parser.add_argument(
        "--elk-index", 
        type=str, 
        default=os.getenv("ELK_INDEX", "logs-test_data-default"),
        help="Elasticsearch index pattern to search."
    )
    parser.add_argument(
        "--elk-user",
        type=str,
        default=os.getenv("ELASTICSEARCH_USER"),
        help="Username for Elasticsearch authentication."
    )
    parser.add_argument(
        "--elk-password",
        type=str,
        default=os.getenv("ELASTICSEARCH_PASSWORD"),
        help="Password for Elasticsearch authentication."
    )
    parser.add_argument(
        "--elk-max-results",
        type=int,
        default=os.getenv("ELK_MAX_RESULTS", 10000),
        help="Maximum number of results to fetch from Elasticsearch."
    )
    parser.add_argument(
        "--elk-time-field",
        type=str,
        default=os.getenv("ELK_TIME_FIELD", "@timestamp"),
        help="Field name for the timestamp in Elasticsearch documents."
    )
    parser.add_argument(
        "--elk-service-field",
        type=str,
        default=os.getenv("ELK_SERVICE_FIELD", "service.name"),
        help="Field name for the service name in Elasticsearch documents."
    )
    parser.add_argument(
        "--elk-message-field",
        type=str,
        default=os.getenv("ELK_MESSAGE_FIELD", "message"),
        help="Field name for the log message in Elasticsearch documents."
    )
    parser.add_argument(
        "--elk-level-field",
        type=str,
        default=os.getenv("ELK_LEVEL_FIELD", "log.level"),
        help="Field name for the log level in Elasticsearch documents."
    )

    # Time window for analysis
    DEFAULT_END_TIME = datetime.now(timezone.utc).isoformat()
    DEFAULT_START_TIME = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    parser.add_argument(
        "--start-time", 
        type=datetime.fromisoformat, 
        default=os.getenv("START_TIME", DEFAULT_START_TIME),
        help="Start time for log analysis (ISO 8601 format)."
    )
    parser.add_argument(
        "--end-time", 
        type=datetime.fromisoformat, 
        default=os.getenv("END_TIME", DEFAULT_END_TIME),
        help="End time for log analysis (ISO 8601 format)."
    )

    args = parser.parse_args()
    
    # Run the main analysis function from the new agent
    analysis_main(args) 