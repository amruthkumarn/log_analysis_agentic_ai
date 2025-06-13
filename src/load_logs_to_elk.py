import argparse
import logging
import os
import re
import time
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Log Parsing Logic (copied from redis_log_analysis_agent.py for consistency) ---

def parse_message_content(message: str, level: str) -> dict:
    """Parse log message content to extract structured information."""
    parsed = {
        'is_login': False, 'is_request': False, 'is_response': False,
        'metrics': {}
    }
    
    # Only add fields if they are actually present in the message
    session_match = re.search(r'session_id=([^,\s]+)', message)
    if session_match:
        parsed['session_id'] = session_match.group(1)
    
    urc_match = re.search(r'URC=([^,\s]+)', message)
    if urc_match:
        parsed['urc'] = urc_match.group(1)
        # If session_id not found but URC contains session pattern, extract it
        if 'session_id' not in parsed:
            # Extract session_id from URC pattern like urc-payment-req-gbx131-01
            urc_session_match = re.search(r'urc-[^-]+-[^-]+-([a-z0-9]+)-\d+', parsed['urc'])
            if urc_session_match:
                parsed['session_id'] = urc_session_match.group(1)
    
    uid_match = re.search(r'UID=([^,\s]+)', message)
    if uid_match:
        parsed['uid'] = uid_match.group(1)
        # If session_id not found but UID contains session pattern, extract it
        if 'session_id' not in parsed:
            # Extract session_id from UID pattern like urc-payment-req-gbx131-01
            uid_session_match = re.search(r'urc-[^-]+-[^-]+-([a-z0-9]+)-\d+', parsed['uid'])
            if uid_session_match:
                parsed['session_id'] = uid_session_match.group(1)
    
    # Extract other fields only if present
    cif_match = re.search(r'cif_id=([^,\s]+)', message)
    if cif_match:
        parsed['cif_id'] = cif_match.group(1)
    
    transaction_match = re.search(r'transaction_type=([^,\s]+)', message)
    if transaction_match:
        parsed['transaction_type'] = transaction_match.group(1)
    
    endpoint_match = re.search(r'endpoint=([^,\s]+)', message)
    if endpoint_match:
        parsed['api_endpoint'] = endpoint_match.group(1)
    
    # Check for login event
    if 'logged in' in message.lower():
        parsed['is_login'] = True
    
    # Check for API request/response
    if 'request received' in message.lower():
        parsed['is_request'] = True
    elif 'response sent' in message.lower():
        parsed['is_response'] = True
    
    # Extract error information only if it's an error
    is_error_log_level = level.upper() in ['ERROR', 'WARN']
    contains_error_keywords = 'error' in message.lower() or 'failed' in message.lower()
    
    if is_error_log_level or contains_error_keywords:
        if 'timeout' in message.lower():
            parsed['error_type'] = 'timeout'
        elif 'rate limit' in message.lower():
            parsed['error_type'] = 'rate_limit'
        elif 'authentication' in message.lower():
            parsed['error_type'] = 'authentication'
        elif 'connection' in message.lower():
            parsed['error_type'] = 'connection'
        elif 'validation' in message.lower():
            parsed['error_type'] = 'validation'
        elif 'business' in message.lower():
            parsed['error_type'] = 'business'
        elif is_error_log_level:
            parsed['error_type'] = 'system'
        else:
            parsed['error_type'] = 'unknown_error'
        
        # Set severity based on log level
        if level.upper() == 'ERROR':
            parsed['severity'] = 'HIGH'
        elif level.upper() == 'WARN':
            parsed['severity'] = 'MEDIUM'
    
    # Extract metrics if present
    duration_match = re.search(r'duration[:\s]+(\d+)', message, re.IGNORECASE)
    if duration_match:
        parsed['metrics']['duration_ms'] = int(duration_match.group(1))
    
    return parsed

def parse_log_line(line: str) -> dict:
    """Parse a log line into its components."""
    pattern = r'\[(.*?)\] \[(.*?)\] (.*?): (.*)'
    match = re.match(pattern, line)
    if match:
        timestamp_str, level, source, message = match.groups()
        try:
            # Handle different timestamp formats gracefully
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').isoformat() + "Z"
        except ValueError:
            timestamp = datetime.now().isoformat() + "Z" # Fallback

        parsed_log = {
            '@timestamp': timestamp,
            'log.level': level,
            'service.name': source,
            'message': message,
        }

        # --- FIX: Parse the message content and merge the results ---
        parsed_content = parse_message_content(message, level)
        parsed_log.update(parsed_content)
        # --- END FIX ---

        return parsed_log
    return None

# --- Elasticsearch Utilities ---

def get_es_client(host):
    """Connect to Elasticsearch, with retries."""
    es_url = f"http://{host}:9200"
    for i in range(10):
        try:
            client = Elasticsearch(hosts=[es_url], verify_certs=False)
            # The ping() method seems to be the source of the issue.
            # We will rely on the subsequent operations to validate the connection.
            logging.info(f"Elasticsearch client initialized for {es_url}. Connection will be verified by first operation.")
            return client
        except Exception as e:
            logging.warning(f"Connection attempt {i+1} to {es_url} failed. Retrying in 10 seconds... Error: {e}")
            time.sleep(10)
    logging.error(f"Could not connect to Elasticsearch at {es_url} after multiple retries.")
    return None

def create_index(client, index_name):
    """Create an Elasticsearch index if it doesn't exist."""
    if not client.indices.exists(index=index_name):
        try:
            client.indices.create(index=index_name)
            logging.info(f"Index '{index_name}' created.")
        except Exception as e:
            logging.error(f"Failed to create index '{index_name}': {e}")
            raise
    else:
        logging.info(f"Index '{index_name}' already exists.")


# --- Main Execution ---
def main(args):
    es_client = get_es_client(args.elk_host)
    if not es_client:
        return

    create_index(es_client, args.elk_index)

    actions = []
    for log_file_path in args.log_files:
        if not os.path.exists(log_file_path):
            logging.warning(f"Log file not found: {log_file_path}. Skipping.")
            continue
        
        logging.info(f"Processing log file: {log_file_path}")
        with open(log_file_path, 'r') as f:
            for line in f:
                parsed_log = parse_log_line(line.strip())
                if parsed_log:
                    action = {
                        "_op_type": "index",
                        "_index": args.elk_index,
                        "_source": parsed_log
                    }
                    actions.append(action)

    if actions:
        logging.info(f"Bulk indexing {len(actions)} documents...")
        try:
            helpers.bulk(es_client, actions)
            logging.info("Successfully indexed documents.")
        except helpers.BulkIndexError as e:
            logging.error(f"Bulk indexing failed for {len(e.errors)} documents.")
            for i, error in enumerate(e.errors):
                 # Log first 5 errors as examples
                if i >= 5:
                    logging.error("...and more errors.")
                    break
                logging.error(f"Error {i+1}: {error}")
    else:
        logging.info("No documents to index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load static log files into Elasticsearch.")
    parser.add_argument(
        "--log-files", 
        nargs='+', 
        required=True,
        help="List of log file paths to ingest."
    )
    parser.add_argument(
        "--elk-host", 
        default=os.getenv("ELASTICSEARCH_HOST", "elasticsearch"),
        help="Elasticsearch host name."
    )
    parser.add_argument(
        "--elk-index", 
        default=os.getenv("ELK_INDEX", "test-logs-default"),
        help="Elasticsearch index name."
    )
    args = parser.parse_args()
    main(args) 