import elasticsearch
from elasticsearch import Elasticsearch
import logging
import time
import os
from datetime import datetime, timedelta
import random
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELK_INDEX", "logs-test_data-default")

def get_es_client():
    """Connect to Elasticsearch, with retries."""
    for i in range(10):
        try:
            client = Elasticsearch(
                hosts=[ELASTICSEARCH_HOST],
                verify_certs=False
            )
            if client.ping():
                logging.info("Successfully connected to Elasticsearch.")
                return client
        except elasticsearch.ConnectionError as e:
            logging.warning(f"Connection attempt {i+1} failed. Retrying in 10 seconds... Error: {e}")
            time.sleep(10)
    logging.error("Could not connect to Elasticsearch after multiple retries.")
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

def generate_log(session_id, urc, uid=None):
    """Generates a structured log entry."""
    service_name = random.choice(["api-gateway", "user-service", "payment-service", "3scale-api-gateway"])
    log_level = random.choices(["INFO", "WARN", "ERROR"], weights=[0.8, 0.15, 0.05], k=1)[0]
    
    message = f"Request received for session_id={session_id}, URC={urc}"
    if uid:
        message += f", UID={uid}"
        
    if log_level == "ERROR":
        error_type = random.choice(["timeout", "authentication", "connection"])
        message += f", error='{error_type} failure'"

    return {
        "@timestamp": datetime.utcnow().isoformat() + "Z",
        "service.name": service_name,
        "log.level": log_level,
        "message": message,
    }

def main():
    """Main function to generate and push logs."""
    es_client = get_es_client()
    if not es_client:
        return

    create_index(es_client, ELASTICSEARCH_INDEX)

    # Generate logs for a few sessions
    for i in range(3): # 3 user sessions
        session_id = f"session-{uuid.uuid4()}"
        root_urc = f"urc-root-{uuid.uuid4()}"
        
        # 1. Login event
        login_log = generate_log(session_id, root_urc)
        login_log["message"] = f"User logged in, session_id={session_id}, URC={root_urc}"
        es_client.index(index=ELASTICSEARCH_INDEX, document=login_log)
        
        # 2. A few API calls
        parent_urc = root_urc
        for j in range(random.randint(2, 4)): # 2-4 child calls
             child_urc = f"urc-child-{uuid.uuid4()}"
             api_call_log = generate_log(session_id, child_urc, uid=parent_urc)
             es_client.index(index=ELASTICSEARCH_INDEX, document=api_call_log)
             parent_urc = child_urc
             # Maybe an error on the last call
             if j > 0 and random.random() < 0.3:
                 error_log = generate_log(session_id, child_urc, uid=parent_urc)
                 error_log["log.level"] = "ERROR"
                 error_log["message"] = f"Backend service timeout for service_id=123, session_id={session_id}, URC={child_urc}, timeout=5000ms"
                 es_client.index(index=ELASTICSEARCH_INDEX, document=error_log)


    logging.info("Finished generating and pushing logs.")

if __name__ == "__main__":
    main() 