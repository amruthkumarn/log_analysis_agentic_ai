from typing import Annotated, Sequence, TypedDict, Dict, List, Set
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime, timedelta
import re
from collections import defaultdict
import json
from document_processor import DocumentProcessor
import logging
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.output_parsers import OutputFixingParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define UPART attributes
class UPARTAttributes(TypedDict):
    user: str
    parameters: Dict[str, str]
    action: str
    resource: str
    time: datetime

# Define our state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    raw_logs: Annotated[Dict[str, List[dict]], "Raw log entries"]
    normalized_logs: Annotated[Dict[str, List[UPARTAttributes]], "Normalized logs with UPART attributes"]
    log_content: Annotated[Dict[str, str], "The log content from different sources"]
    correlations: Annotated[List[Dict], "Correlated events across logs"]
    impact_analysis: Annotated[Dict, "Analysis of impact and dependencies"]
    root_causes: Annotated[List[Dict], "Identified root causes"]

# Define specialized agent states
class ParserState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    raw_logs: Annotated[Dict[str, List[dict]], "Raw log entries"]
    normalized_logs: Annotated[Dict[str, List[UPARTAttributes]], "Normalized logs with UPART attributes"]

class CorrelationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    normalized_logs: Annotated[Dict[str, List[UPARTAttributes]], "Normalized logs"]
    correlated_events: Annotated[List[Dict], "Correlated events with temporal/spatial relationships"]

class AnomalyState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    correlated_events: Annotated[List[Dict], "Correlated events"]
    anomalies: Annotated[List[Dict], "Detected anomalies with deviation scores"]

class RootCauseState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    anomalies: Annotated[List[Dict], "Detected anomalies"]
    causal_graph: Annotated[Dict, "Causal inference graph"]
    root_causes: Annotated[List[Dict], "Identified root causes"]

# RAG Integration
class RAGManager:
    def __init__(self, docs_dir: str = "documentation"):
        self.docs_dir = docs_dir
        self.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        self.vectorstore = None
        self.document_processor = DocumentProcessor(docs_dir)
        self.initialize_vectorstore()
    
    def initialize_vectorstore(self):
        """Initialize the vector store with documentation."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            return
        
        # Load and split documents
        documents = []
        
        # Process text and markdown files
        for file in os.listdir(self.docs_dir):
            if file.endswith(('.txt', '.md')):
                with open(os.path.join(self.docs_dir, file), 'r') as f:
                    documents.append(f.read())
        
        # Process PDF files using DocumentProcessor
        self.document_processor.process_documents()
        pdf_docs = self.document_processor.vector_store.similarity_search("", k=1000)  # Get all PDF documents
        documents.extend([doc.page_content for doc in pdf_docs])
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.create_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
    
    def get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documentation context for a query."""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

def parse_log_line(line: str) -> dict:
    """Parse a log line into its components."""
    pattern = r'\[(.*?)\] \[(.*?)\] (.*?): (.*)'
    match = re.match(pattern, line)
    if match:
        timestamp, level, source, message = match.groups()
        parsed = {
            'timestamp': timestamp,
            'level': level,
            'source': source,
            'message': message,
            'parsed_message': parse_message_content(message, level)
        }
        return parsed
    return None

def read_log_file(file_path: str) -> List[dict]:
    """Read and parse a log file."""
    log_entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = parse_log_line(line.strip())
            if entry:
                log_entries.append(entry)
    return log_entries

def parse_message_content(message: str, level: str) -> Dict:
    """Parse log message content to extract structured information."""
    parsed = {
        'is_login': False,
        'is_request': False,
        'is_response': False,
        'session_id': None,
        'cif_id': None,
        'urc': None,
        'uid': None,
        'transaction_type': None,
        'api_endpoint': None,
        'error_type': None,
        'severity': None,
        'metrics': {}
    }
    
    # Extract session_id
    session_match = re.search(r'session_id=([^,\s]+)', message)
    if session_match:
        parsed['session_id'] = session_match.group(1)
        logger.info(f"Extracted session_id: {parsed['session_id']}")
    
    # Extract CIF ID
    cif_match = re.search(r'cif_id=([^,\s]+)', message)
    if cif_match:
        parsed['cif_id'] = cif_match.group(1)
    
    # Extract URC
    urc_match = re.search(r'URC=([^,\s]+)', message)
    if urc_match:
        parsed['urc'] = urc_match.group(1)
        logger.info(f"Extracted URC: {parsed['urc']}")
    
    # Extract UID
    uid_match = re.search(r'UID=([^,\s]+)', message)
    if uid_match:
        parsed['uid'] = uid_match.group(1)
        logger.info(f"Extracted UID: {parsed['uid']}")
    
    # Check for login event
    if 'logged in' in message.lower():
        parsed['is_login'] = True
        logger.info("Detected login event")
        # If this is a login event, ensure we have a session_id
        if not parsed['session_id']:
            # Try to extract session_id from the message
            session_match = re.search(r'session[:\s]+([^,\s]+)', message, re.IGNORECASE)
            if session_match:
                parsed['session_id'] = session_match.group(1)
                logger.info(f"Extracted session_id from login event: {parsed['session_id']}")
    
    # Check for API request/response
    if 'request received' in message.lower():
        parsed['is_request'] = True
        logger.info("Detected API request")
    elif 'response sent' in message.lower():
        parsed['is_response'] = True
        logger.info("Detected API response")
    
    # Extract API endpoint
    endpoint_match = re.search(r'endpoint=([^,\s]+)', message)
    if endpoint_match:
        parsed['api_endpoint'] = endpoint_match.group(1)
    
    # Extract transaction type
    tx_match = re.search(r'transaction_type=([^,\s]+)', message)
    if tx_match:
        parsed['transaction_type'] = tx_match.group(1)
    
    # Extract error information
    is_error_log_level = level.upper() in ['ERROR', 'WARN']
    contains_error_keywords = 'error' in message.lower() or 'failed' in message.lower()

    if is_error_log_level or contains_error_keywords:
        # Determine error type based on keywords if available
        if 'timeout' in message.lower():
            parsed['error_type'] = 'timeout'
        elif 'rate limit' in message.lower():
            parsed['error_type'] = 'rate_limit'
        elif 'authentication' in message.lower(): # This will catch "Authentication failed"
            parsed['error_type'] = 'authentication'
        elif 'connection' in message.lower():
            parsed['error_type'] = 'connection'
        elif 'validation' in message.lower():
            parsed['error_type'] = 'validation'
        elif 'business' in message.lower():
            parsed['error_type'] = 'business'
        elif is_error_log_level and not parsed.get('error_type'): # If it's an ERROR/WARN level but no specific keyword matched
            parsed['error_type'] = 'system' # Default for error level logs
        elif not parsed.get('error_type') and contains_error_keywords: # If keywords like "error" were present but no specific type matched
             parsed['error_type'] = 'unknown_error'


        # Extract severity if explicitly mentioned
        severity_match = re.search(r'severity=([^,\\s]+)', message, re.IGNORECASE)
        if severity_match:
            parsed['severity'] = severity_match.group(1).upper()
        else:
            # Default severity based on log level if not specified in message
            if level.upper() == 'ERROR':
                parsed['severity'] = 'HIGH' # Default for ERROR
            elif level.upper() == 'WARN':
                parsed['severity'] = 'MEDIUM' # Default for WARN
            elif parsed.get('error_type') in ['timeout', 'rate_limit', 'authentication', 'connection']: # Check parsed error_type
                 parsed['severity'] = 'HIGH' # Fallback if level was INFO but error_type indicates high severity
            elif parsed.get('error_type'): # If an error_type is set, but not covered above
                 parsed['severity'] = 'MEDIUM' # Default for other errors
    
    # Extract metrics
    metrics = {}
    duration_match = re.search(r'duration[:\s]+(\d+)', message, re.IGNORECASE)
    if duration_match:
        metrics['duration_ms'] = int(duration_match.group(1))
    
    response_time_match = re.search(r'response_time=(\d+)', message)
    if response_time_match:
        metrics['response_time_ms'] = int(response_time_match.group(1))
    
    if metrics:
        parsed['metrics'] = metrics
    
    logger.info(f"Final parsed message (level: {level}): {parsed}")
    return parsed

def analyze_error_chain(entries: List[dict], chain_scope_id: str) -> Dict:
    """Analyze a chain of errors and their relationships within a defined scope (e.g., session)."""
    logger.info(f"Analyzing error chain for scope: {chain_scope_id}")
    
    error_chain = []
    for entry in entries:
        if entry['level'] in ['ERROR', 'WARN']:
            logger.info(f"Processing error: {entry['message']} at {entry['timestamp']}")
            
            # Get severity from parsed message
            severity = entry['parsed_message'].get('severity', 'MEDIUM')
            
            # Analyze error relationships
            error_context = {
                'timestamp': entry['timestamp'],
                'level': entry['level'],
                'source': entry['source'],
                'message': entry['message'],
                'error_type': entry['parsed_message']['error_type'],
                'session_id': entry['parsed_message']['session_id'],
                'urc': entry['parsed_message']['urc'],
                'uid': entry['parsed_message']['uid'],
                'transaction_type': entry['parsed_message']['transaction_type'],
                'api_endpoint': entry['parsed_message']['api_endpoint'],
                'related_errors': [],
                'impact': {
                    'severity': severity,
                    'affected_components': {},
                    'performance_impact': False,
                    'security_impact': False
                }
            }
            
            # Add affected components
            if entry['parsed_message'].get('session_id'):
                error_context['impact']['affected_components']['session'] = entry['parsed_message']['session_id']
            if entry['parsed_message'].get('cif_id'):
                error_context['impact']['affected_components']['customer'] = entry['parsed_message']['cif_id']
            if entry['parsed_message'].get('api_endpoint'):
                error_context['impact']['affected_components']['endpoint'] = entry['parsed_message']['api_endpoint']
            if entry['parsed_message'].get('transaction_type'):
                error_context['impact']['affected_components']['transaction'] = entry['parsed_message']['transaction_type']
            
            # Set performance and security impact
            if entry['parsed_message']['error_type'] in ['timeout', 'performance', 'queue', 'backend_service_timeout']:
                error_context['impact']['performance_impact'] = True
            if entry['parsed_message']['error_type'] in ['authentication', 'rate_limit']:
                error_context['impact']['security_impact'] = True
            
            # Find related errors within the same transaction
            for other_entry in entries:
                if other_entry != entry and other_entry['level'] in ['ERROR', 'WARN']:
                    if is_related_error(entry, other_entry):
                        error_context['related_errors'].append({
                            'timestamp': other_entry['timestamp'],
                            'message': other_entry['message'],
                            'relationship': determine_error_relationship(entry, other_entry),
                            'api_endpoint': other_entry['parsed_message']['api_endpoint']
                        })
            
            error_chain.append(error_context)
            logger.info(f"Added error to chain: {error_context['error_type']} with {len(error_context['related_errors'])} related errors")
    
    return {
        'session_id': chain_scope_id,
        'errors': error_chain,
        'total_errors': len(error_chain),
        'impact_level': calculate_chain_impact(error_chain)
    }

def is_related_error(error1: dict, error2: dict) -> bool:
    """Determine if two errors are related within a transaction."""
    # Check if they're from the same session
    if error1['parsed_message']['session_id'] != error2['parsed_message']['session_id']:
        return False
    
    # Check if they're part of the same transaction (URC/UID relationship)
    urc1 = error1['parsed_message']['urc']
    uid1 = error1['parsed_message']['uid']
    urc2 = error2['parsed_message']['urc']
    uid2 = error2['parsed_message']['uid']
    
    same_transaction = (
        urc1 == urc2 or  # Same URC
        urc1 == uid2 or  # URC matches UID
        uid1 == urc2 or  # UID matches URC
        uid1 == uid2     # Same UID
    )
    
    if not same_transaction:
        return False
    
    # Check temporal relationship (within 5 minutes)
    time1 = datetime.strptime(error1['timestamp'], '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(error2['timestamp'], '%Y-%m-%d %H:%M:%S')
    time_diff = abs((time1 - time2).total_seconds())
    
    # Check error type relationship
    error_type_relationship = {
        'timeout': ['connection', 'performance'],
        'connection': ['timeout', 'queue'],
        'authentication': ['rate_limit', 'validation'],
        'rate_limit': ['authentication', 'performance'],
        'queue': ['connection', 'performance'],
        'performance': ['timeout', 'queue'],
        'validation': ['business', 'authentication'],
        'business': ['validation', 'system'],
        'system': ['business', 'performance']
    }
    
    related_types = (
        error1['parsed_message']['error_type'] in error_type_relationship.get(error2['parsed_message']['error_type'], []) or
        error2['parsed_message']['error_type'] in error_type_relationship.get(error1['parsed_message']['error_type'], [])
    )
    
    return time_diff <= 300 and related_types

def determine_error_relationship(error1: dict, error2: dict) -> str:
    """Determine the relationship between two errors."""
    time1 = datetime.strptime(error1['timestamp'], '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(error2['timestamp'], '%Y-%m-%d %H:%M:%S')
    
    if time1 < time2:
        return 'causes'
    elif time1 > time2:
        return 'caused_by'
    else:
        return 'concurrent'

def calculate_error_impact(error: dict) -> Dict:
    """Calculate the impact of a single error."""
    impact = {
        'severity': 'LOW',
        'affected_components': {},
        'performance_impact': False,
        'security_impact': False
    }
    
    # Determine severity based on both log level and error type
    error_type = error['parsed_message']['error_type']
    log_level = error['level']
    
    # Set base severity from log level
    if log_level == 'ERROR':
        impact['severity'] = 'HIGH'
    elif log_level == 'WARN':
        impact['severity'] = 'MEDIUM'
    
    # Adjust severity based on error type
    critical_errors = ['rate_limit', 'authentication', 'backend_service_health_check', 'backend_service_timeout']
    high_impact_errors = ['timeout', 'connection', 'queue', 'performance']
    
    if error_type in critical_errors:
        impact['severity'] = 'CRITICAL'
    elif error_type in high_impact_errors:
        impact['severity'] = 'HIGH'
    
    # Check for performance impact
    if error_type in ['timeout', 'performance', 'queue', 'backend_service_timeout']:
        impact['performance_impact'] = True
    
    # Check for security impact
    if error_type in ['authentication', 'rate_limit']:
        impact['security_impact'] = True
    
    # Add affected components based on new log format
    parsed_msg = error['parsed_message']
    if parsed_msg.get('session_id'):
        impact['affected_components']['session'] = parsed_msg['session_id']
    if parsed_msg.get('cif_id'):
        impact['affected_components']['customer'] = parsed_msg['cif_id']
    if parsed_msg.get('api_endpoint'):
        impact['affected_components']['endpoint'] = parsed_msg['api_endpoint']
    if parsed_msg.get('transaction_type'):
        impact['affected_components']['transaction'] = parsed_msg['transaction_type']
    
    # Add performance metrics if available
    if error['parsed_message']['metrics']:
        impact['performance_metrics'] = error['parsed_message']['metrics']
    
    return impact

def calculate_chain_impact(error_chain: List[Dict]) -> str:
    """Calculate the overall impact level of an error chain."""
    if not error_chain:
        return 'INFO'
    
    high_impact = sum(1 for error in error_chain if error['impact']['severity'] == 'HIGH')
    medium_impact = sum(1 for error in error_chain if error['impact']['severity'] == 'MEDIUM')
    
    if high_impact >= 2:
        return 'CRITICAL'
    elif high_impact == 1 or medium_impact >= 2:
        return 'HIGH'
    elif medium_impact == 1:
        return 'MEDIUM'
    return 'LOW'

class APICallNode:
    """Represents a node in the API call tree."""
    def __init__(self, urc: str, entry: dict):
        self.urc = urc
        self.entry = entry
        self.children = []  # Child API calls
        self.response = None  # Response for this API call
        self.errors = []  # Errors related to this API call
        self.level = 0  # Level in the call tree

def build_api_call_tree(entries: List[dict]) -> Dict[str, APICallNode]:
    """Build a tree of API calls based on URC/UID relationships."""
    logger.info("Building API call tree")
    
    # First, find the login entry which contains the root URC
    root_entry = None
    for entry in entries:
        logger.info(f"Checking entry: {entry['message']}")
        logger.info(f"Parsed message: {entry['parsed_message']}")
        
        if entry['parsed_message']['is_login'] and entry['parsed_message']['urc']:
            root_entry = entry
            logger.info(f"Found root entry: {root_entry['message']}")
            logger.info(f"Root URC: {root_entry['parsed_message']['urc']}")
            logger.info(f"Root session_id: {root_entry['parsed_message']['session_id']}")
            break
    
    if not root_entry:
        logger.warning("No login entry with URC found")
        return {}
    
    current_session_id_for_debug = root_entry['parsed_message'].get('session_id')

    # Initialize the tree with the root node
    nodes = {}
    root_urc = root_entry['parsed_message']['urc']
    root_node_obj = APICallNode(root_urc, root_entry)
    root_node_obj.level = 0 # Root is level 0
    nodes[root_urc] = root_node_obj
    logger.info(f"Created root node with URC: {root_urc} at level 0")
    
    # Sort entries by timestamp to process in order
    sorted_entries = sorted(entries, 
                          key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))
    
    # Process each entry to build the tree
    for entry in sorted_entries:
        parsed = entry['parsed_message']
        
        # ---- REMOVE OLD DEBUG LOG ----
        # if current_session_id_for_debug == 'abc123' and \
        #    entry.get('level') in ['ERROR', 'WARN'] and \
        #    parsed.get('urc') in ['child2', 'child3']:
        #     logger.info(f"TREE_LOOP_DEBUG URC {parsed.get('urc')}: Message: {entry.get('message')}")
        #     logger.info(f"TREE_LOOP_DEBUG URC {parsed.get('urc')}: Parsed error_type: {parsed.get('error_type')}")
        #     logger.info(f"TREE_LOOP_DEBUG URC {parsed.get('urc')}: Parsed urc: {parsed.get('urc')}")
        # ---- END REMOVE OLD DEBUG LOG ----
        
        logger.info(f"\nProcessing entry: {entry['message']}")
        logger.info(f"Timestamp: {entry['timestamp']}")
        logger.info(f"URC: {parsed.get('urc')}")
        logger.info(f"UID: {parsed.get('uid')}")
        logger.info(f"Session ID: {parsed.get('session_id')}")
        
        if parsed['is_request'] and parsed['urc']:
            current_urc = parsed['urc']
            # Skip if this is the root URC being processed as a request again
            if current_urc == root_urc and nodes[current_urc].entry == entry:
                continue

            if current_urc not in nodes:
                parent_node = None
                parent_urc_from_uid = parsed.get('uid')

                if parent_urc_from_uid and parent_urc_from_uid in nodes:
                    parent_node = nodes[parent_urc_from_uid]
                    logger.info(f"Found parent node {parent_urc_from_uid} for {current_urc} via UID.")
                elif not parent_urc_from_uid and current_urc != root_urc:
                    # This case implies a new root-level request if no UID is present and it's not the main root_urc
                    # For this problem, we assume a single root URC from login, so other URCs must link via UID
                    logger.warning(f"Request URC {current_urc} has no UID to link to a parent. Treating as orphan unless it is root.")
                    # If it's a new URC without UID and not the root, it won't be added to the tree here.
                    # Or, decide to make it a new root if that's desired (not per current requirements)

                if parent_node and parent_node.level < 3: # Max depth check (0, 1, 2, 3 are 4 levels)
                    new_node = APICallNode(current_urc, entry)
                    new_node.level = parent_node.level + 1
                    parent_node.children.append(new_node)
                    nodes[current_urc] = new_node
                    logger.info(f"Added child node {current_urc} (level {new_node.level}) to parent {parent_node.urc} (level {parent_node.level})")
                elif not parent_node and current_urc == root_urc:
                    # This is the root URC's own request event, node already exists from login
                    # We might want to update its entry if this request log has more info, but APICallNode stores one 'entry'
                    logger.info(f"Processing request for already existing root node {current_urc}. Not creating new node or child.")
                elif parent_node and parent_node.level >= 3:
                    logger.info(f"Skipping child node {current_urc} because parent {parent_node.urc} is at max depth (level {parent_node.level}).")
                elif not parent_node and current_urc != root_urc:
                    logger.warning(f"Request for URC {current_urc} could not be linked to a parent node and is not the root. Ignoring for tree.")
            else:
                logger.info(f"Node for URC {current_urc} already exists. Request entry: {entry['message']}")
        
        elif parsed['is_response'] and parsed['uid']:
            # This is a response, find the corresponding request node
            if parsed['uid'] in nodes:
                nodes[parsed['uid']].response = entry
                logger.info(f"Added response to node with UID: {parsed['uid']}")
            else:
                logger.warning(f"Response UID {parsed['uid']} not found in nodes")
        
        elif parsed['error_type'] and parsed['urc']:
            # This is an error, add it to the corresponding node
            logger.info(f"ERROR_CHECK: Processing error entry for URC: '{parsed['urc']}'. Node keys: {list(nodes.keys())}")
            if parsed['urc'] in nodes:
                nodes[parsed['urc']].errors.append(entry)
                logger.info(f"SUCCESS: Added error TO NODE {parsed['urc']}: {entry['message']}")
            else:
                logger.warning(f"Error URC {parsed['urc']} not found in nodes")
    
    # Print final tree structure
    logger.info("\nFinal Tree Structure:")
    def print_tree(node: APICallNode, level: int = 0):
        indent = "  " * level
        logger.info(f"{indent}Node URC: {node.urc}")
        logger.info(f"{indent}Session ID: {node.entry['parsed_message'].get('session_id')}")
        logger.info(f"{indent}Level: {node.level}")
        logger.info(f"{indent}Children: {[child.urc for child in node.children]}")
        for child in node.children:
            print_tree(child, level + 1)
    
    root_node = next(iter(nodes.values()))
    print_tree(root_node)
    
    logger.info(f"Built API call tree with {len(nodes)} nodes")
    return nodes

def find_error_chains(api_tree: Dict[str, APICallNode]) -> List[Dict]:
    """
    Extracts and organizes error chains from the API call tree for a single session.
    It collects all errors and then formats them using analyze_error_chain.
    """
    logger.info("Finding error chains from API call tree")
    all_errors_for_session = []
    
    # Determine the session_id for the current API tree from its root node
    tree_session_id = None
    if api_tree:
        for node_obj in api_tree.values():
            if node_obj.level == 0: # Root node
                tree_session_id = node_obj.entry['parsed_message'].get('session_id')
                if tree_session_id:
                    logger.info(f"Determined session_id for API tree: {tree_session_id}")
                break
    
    if not tree_session_id:
        logger.warning("Could not determine session_id for the API tree in find_error_chains. Errors might not be correctly attributed to a session.")
        # Fallback: try to get session_id from any node, though this is less reliable
        if api_tree:
            first_node = next(iter(api_tree.values()), None)
            if first_node:
                tree_session_id = first_node.entry['parsed_message'].get('session_id', 'unknown_session')


    def extract_errors_from_api_node(node: APICallNode, session_id_for_chain: str) -> List[Dict]:
        """Helper to extract error entries from a single APICallNode for chain construction."""
        node_specific_errors = []
        logger.info(f"FIND_ERROR_CHAINS: Extracting errors from node {node.urc}. It has {len(node.errors)} direct errors.")
        # Add errors directly associated with the request node (node.errors)
        for error_entry in node.errors:
            # Ensure the error entry has the correct session_id for the chain
            current_parsed_message = error_entry.get('parsed_message', {})
            current_parsed_message['session_id'] = session_id_for_chain # Override or set session_id
            error_entry['parsed_message'] = current_parsed_message
            node_specific_errors.append(error_entry)

        # Add errors from the response, if any
        if node.response and node.response.get('level') in ['ERROR', 'WARN']:
            # Construct a full error entry from the response, similar to how other errors are structured
            response_parsed_message = node.response.get('parsed_message', {})
            response_parsed_message['session_id'] = session_id_for_chain # Override or set session_id
            
            response_error_entry = {
                'timestamp': node.response.get('timestamp'),
                'level': node.response.get('level'),
                'source': node.response.get('source'),
                'message': node.response.get('message'),
                'parsed_message': response_parsed_message
            }
            node_specific_errors.append(response_error_entry)
            logger.info(f"FIND_ERROR_CHAINS: Added response error from node {node.urc}: {response_error_entry['message']}")
        logger.info(f"FIND_ERROR_CHAINS: Extracted {len(node_specific_errors)} errors in total from node {node.urc}.")
        return node_specific_errors

    for node_urc in api_tree:
        node = api_tree[node_urc]
        # Use the determined tree_session_id for all errors. If still None, it will be handled by analyze_error_chain.
        errors_from_node = extract_errors_from_api_node(node, tree_session_id if tree_session_id else "unknown_session_in_find_error_chains")
        all_errors_for_session.extend(errors_from_node)

    final_formatted_chains = []
    if all_errors_for_session:
        # All collected errors for this session are passed as a single list to analyze_error_chain
        # The `transaction_id` for `analyze_error_chain` is the session_id of this tree.
        if tree_session_id:
            logger.info(f"Formatting error chain for session {tree_session_id} with {len(all_errors_for_session)} raw errors.")
            formatted_chain_dict = analyze_error_chain(all_errors_for_session, tree_session_id)
            if formatted_chain_dict and formatted_chain_dict.get('errors'): # Ensure there are actual errors after formatting
                 final_formatted_chains.append(formatted_chain_dict)
        else:
            logger.warning("No session_id available to format error chain in find_error_chains. Skipping formatting.")
    
    logger.info(f"Found and formatted {len(final_formatted_chains)} error chains for the API tree.")
    return final_formatted_chains

def calculate_avg_response_time(api_tree: Dict[str, APICallNode]) -> float:
    """Calculates the average response time from the API call tree."""
    total_response_time = 0
    num_responses = 0
    if not api_tree:
        return 0.0

    for node in api_tree.values():
        if node.response and node.response.get('parsed_message'):
            metrics = node.response['parsed_message'].get('metrics', {})
            if 'response_time_ms' in metrics:
                try:
                    # Ensure response_time_ms is treated as a number
                    response_time_val = metrics['response_time_ms']
                    if isinstance(response_time_val, str):
                        response_time_val = response_time_val.replace('ms', '') # Clean if 'ms' is part of string
                    total_response_time += int(response_time_val)
                    num_responses += 1
                except ValueError:
                    logger.warning(f"Could not parse response_time_ms: {metrics['response_time_ms']} in node {node.urc}")
                except TypeError:
                    logger.warning(f"Invalid type for response_time_ms: {metrics['response_time_ms']} in node {node.urc}")


    if num_responses == 0:
        logger.info("No responses with response_time_ms found in API tree for averaging.")
        return 0.0
    
    avg_time = total_response_time / num_responses
    logger.info(f"Calculated average response time: {avg_time:.2f}ms from {num_responses} responses.")
    return avg_time

def correlate_logs(log_entries: Dict[str, List[dict]]) -> List[dict]:
    """Correlate logs across different sources and build session-based analysis."""
    logger.info("Starting log correlation")

    # 1. Create a flat list of all log entries, tagged with source, sorted chronologically
    all_entries_chronological = []
    for source, entries in log_entries.items():
        for entry in entries:
            if entry: # Ensure entry is not None (e.g. from failed parsing)
                all_entries_chronological.append({**entry, 'source_file': source})
    
    all_entries_chronological.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))
    logger.info(f"Created a flat list of {len(all_entries_chronological)} log entries from all sources.")

    # Find all login events to start session-based analysis
    login_events = []
    for entry in all_entries_chronological:
        if entry['parsed_message'].get('is_login') and entry['parsed_message'].get('urc'):
            login_events.append(entry)
            logger.info(f"Found login event: {entry['message']} from {entry['source_file']}")
            logger.info(f"Session ID: {entry['parsed_message']['session_id']}, URC: {entry['parsed_message']['urc']}")

    if not login_events:
        logger.warning("No login events with URC found across all logs.")
        return []

    correlations = []
    # 2. For each login_event, iteratively collect all related entries
    for login_event in login_events:
        current_session_id = login_event['parsed_message']['session_id']
        root_urc = login_event['parsed_message']['urc']

        if not current_session_id:
            logger.warning(f"Login event has no session_id: {login_event['message']}. Skipping this session.")
            continue
        if not root_urc:
            logger.warning(f"Login event has no root URC: {login_event['message']}. Skipping this session.") # Should be caught by earlier check but good to have
            continue

        logger.info(f"Processing session {current_session_id} starting with root URC {root_urc}")

        session_specific_entries = []
        known_urcs_uids_for_session = {root_urc} # Start with the root URC
        if login_event['parsed_message'].get('uid'): # Sometimes login might have a UID too
            known_urcs_uids_for_session.add(login_event['parsed_message']['uid'])


        # Iteratively collect entries for the session (e.g., up to 5 passes for 4 levels + errors/responses)
        # Max passes to prevent infinite loops in case of unexpected data, and to cover 4 levels of calls.
        # A pass adds entries whose URC/UID is now known from a previous pass.
        MAX_PASSES = 5 
        processed_entry_keys = set() # Initialize the set here
        for pass_num in range(MAX_PASSES):
            new_entries_added_this_pass = False
            logger.info(f"Session {current_session_id}, Pass {pass_num + 1}: Known URCs/UIDs: {known_urcs_uids_for_session}")
            
            for entry in all_entries_chronological:
                # Avoid adding duplicates
                key = (entry['timestamp'], entry['message'], entry['source_file'])
                if key in processed_entry_keys:
                    continue
                processed_entry_keys.add(key)

                parsed_msg = entry['parsed_message']
                entry_session_id = parsed_msg.get('session_id')
                entry_urc = parsed_msg.get('urc')
                entry_uid = parsed_msg.get('uid')
                
                linked_to_session = False
                if entry_session_id == current_session_id:
                    linked_to_session = True
                    logger.debug(f"Linking entry by session_id ({current_session_id}): {entry['message']}")
                # Link if entry's URC is a UID we know OR entry's UID is a URC we know
                # This covers request-response (URC matches UID of response) and parent-child (UID matches URC of parent)
                elif entry_urc and entry_urc in known_urcs_uids_for_session:
                    linked_to_session = True
                    logger.debug(f"Linking entry by its URC ({entry_urc}) being in known set: {entry['message']}")
                elif entry_uid and entry_uid in known_urcs_uids_for_session:
                    linked_to_session = True
                    logger.debug(f"Linking entry by its UID ({entry_uid}) being in known set: {entry['message']}")

                if linked_to_session:
                    session_specific_entries.append(entry)
                    new_entries_added_this_pass = True
                    # Add its URC and UID to known set for next iteration/pass
                    if entry_urc:
                        known_urcs_uids_for_session.add(entry_urc)
                    if entry_uid:
                        known_urcs_uids_for_session.add(entry_uid)
                    # If we link by session_id, and this entry also has URC/UID, add them too
                    if entry_session_id == current_session_id:
                        if entry_urc: known_urcs_uids_for_session.add(entry_urc)
                        if entry_uid: known_urcs_uids_for_session.add(entry_uid)


            if not new_entries_added_this_pass and pass_num > 0: # Allow at least one full pass after initial login event
                logger.info(f"Session {current_session_id}: No new entries added in pass {pass_num + 1}. Concluding entry collection.")
                break
        
        # Ensure entries are unique and sorted for build_api_call_tree,
        # as the iterative collection might add them out of order if not careful.
        # The chronological sort of all_entries_chronological helps, but re-sorting session_specific_entries ensures it.
        unique_session_entries_dict = {}
        for entry in session_specific_entries:
            key = (entry['timestamp'], entry['message'], entry['source_file'])
            unique_session_entries_dict[key] = entry
        
        final_session_entries = sorted(list(unique_session_entries_dict.values()), 
                                       key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))

        logger.info(f"Collected {len(final_session_entries)} entries for session {current_session_id} with root URC {root_urc}.")
        for e in final_session_entries:
            logger.debug(f"  - {e['timestamp']} {e['message']} (URC:{e['parsed_message'].get('urc')}, UID:{e['parsed_message'].get('uid')})")

        if not final_session_entries:
            logger.warning(f"No entries collected for session {current_session_id} beyond login. Skipping tree build.")
            continue
            
        # 3. Pass this comprehensive list to build_api_call_tree
        api_tree = build_api_call_tree(final_session_entries)
        
        if not api_tree:
            logger.warning(f"API call tree is empty for session {current_session_id}. Skipping further analysis for this session.")
            continue

        error_chains = find_error_chains(api_tree)
        
        correlation = {
            'session_id': current_session_id,
            'root_urc': root_urc,
            'login_event': login_event, # The original login event that started this session
            'api_calls': [], # Populate this based on api_tree
            'error_chains': error_chains,
            'metrics': {
                'total_calls': len(api_tree), # Number of nodes in the tree
                'error_count': sum(len(node.errors) + (1 if node.response and node.response.get('level') in ['ERROR','WARN'] else 0) for node in api_tree.values()),
                'avg_response_time': calculate_avg_response_time(api_tree)
            }
        }
        
        for node_urc, node_obj in api_tree.items():
            api_call = {
                'urc': node_obj.urc,
                'request_log_entry': node_obj.entry, # The log entry that defined this node (request or login)
                'response_log_entry': node_obj.response,
                'associated_error_log_entries': node_obj.errors,
                'children_urcs': [child.urc for child in node_obj.children],
                'level_in_tree': node_obj.level
            }
            correlation['api_calls'].append(api_call)
        
        correlations.append(correlation)
        logger.info(f"Finished processing session {current_session_id}. Correlation: {json.dumps(correlation, default=str, indent=2)}")
    
    logger.info(f"Finished log correlation. Found {len(correlations)} sessions with data.")
    return correlations

# Initialize the LLM
llm = OllamaLLM(model="llama3.2:1b")

# Create the log analyzer agent with enhanced prompt
log_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert log analyzer. Your task is to analyze logs and identify:
    1. Error patterns and their relationships across different services
    2. Performance issues and their impact on system health
    3. Security concerns and authentication issues
    4. System health indicators and resource utilization
    5. Correlated events between different log sources
    6. Service dependencies and failure chains
    7. Impact analysis of errors on different components
    
    Use the provided documentation context to validate your analysis and provide more accurate insights.
    Focus on identifying root causes and their cascading effects."""),
    MessagesPlaceholder(variable_name="messages"),
])

log_analyzer_chain = log_analyzer_prompt | llm | StrOutputParser()

# Create the recommendation agent with enhanced prompt
recommendation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Based on the log analysis, correlations, and impact analysis, provide actionable recommendations for:
    1. Immediate actions needed to resolve critical issues
    2. Long-term improvements to prevent similar issues
    3. Preventive measures and monitoring improvements
    4. System integration and architecture improvements
    5. Performance optimization opportunities
    6. Security hardening measures
    
    Be specific and practical in your recommendations.
    Prioritize recommendations based on impact level and affected services."""),
    MessagesPlaceholder(variable_name="messages"),
])

recommendation_chain = recommendation_prompt | llm | StrOutputParser()

# Initialize RAG manager
rag_manager = RAGManager()

# Create specialized prompts with RAG integration
def create_parser_prompt(rag_manager: RAGManager):
    return ChatPromptTemplate.from_messages([
        ("system", """You are a specialized log parsing agent. Your task is to:
        1. Parse different log formats
        2. Extract UPART attributes:
           - User: Who performed the action
           - Parameters: Input parameters and configuration
           - Action: What operation was performed
           - Resource: What was affected
           - Time: When it occurred
        3. Normalize log entries
        4. Identify log patterns
        5. Categorize log types
        
        Use the provided documentation context to understand log formats and patterns."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_correlation_prompt(rag_manager: RAGManager):
    return ChatPromptTemplate.from_messages([
        ("system", """You are a correlation analysis agent. Your task is to:
        1. Identify related events across logs using:
           - Temporal relationships (time-based)
           - Spatial relationships (service/component-based)
           - Causal relationships (event chains)
        2. Establish event dependencies
        3. Map service interactions
        4. Detect cascading failures
        
        Use the provided documentation context to understand service relationships and dependencies."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_anomaly_prompt(rag_manager: RAGManager):
    return ChatPromptTemplate.from_messages([
        ("system", """You are an anomaly detection agent. Your task is to:
        1. Establish baselines for:
           - Response times
           - Error rates
           - Resource usage
           - Service patterns
        2. Detect deviations using:
           - Statistical analysis
           - Pattern recognition
           - Trend analysis
        3. Calculate anomaly scores
        4. Categorize anomalies
        
        Use the provided documentation context to understand normal system behavior and thresholds."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_root_cause_prompt(rag_manager: RAGManager):
    system_message_intro = (
        "You are an expert system and network administrator tasked with identifying a root cause for a SPECIFIC TRIGGERING ERROR EVENT provided to you."
        "Analyze ONLY the provided single triggering error event, its immediate details, and its position within the larger error chain (also provided). "
        "Focus on the primary trigger for THIS SPECIFIC failure. "
        "Your response MUST BE A SINGLE, VALID JSON OBJECT. No other text, no markdown, just the JSON. "
        "The JSON object must contain the following fields, populated with information derived from the provided error event details:"
        # Note: Explicitly list fields instead of saying 'adhere to schema' initially
    )
    
    fields_description_content = {
        "problem_description": "[String: Brief 1-2 sentence description of the core problem observed FOR THE PROVIDED TRIGGERING ERROR EVENT. Use details like the service name and error type FROM THE TRIGGERING ERROR. E.g., 'Timeout occurred in 3scale API Gateway while processing session abc123.' NOT 'Transaction X failed.']",
        "probable_root_cause_summary": "[String: Concise summary of the MOST LIKELY root cause FOR THE PROVIDED TRIGGERING ERROR EVENT. This summary MUST explain what likely caused THIS SPECIFIC error message in THIS specific service. E.g., 'The 3scale API Gateway timed out because the backend service it called (service_id=123) did not respond within the 5000ms window.' NOT a generic summary or a summary of a different error in the chain.]",
        "key_error_log_messages": "[[List containing EXACTLY ONE string: the UNMODIFIED 'EXACT Triggering Error Message' provided in the input. No other messages. No modifications.]]",
        "confidence_score": "[Float or null: Score 0.0-1.0 for confidence, e.g., 0.85, or null if unsure.]",
        "associated_identifiers": {
            "session_id": "[String or null: EXACT session_id from input.]",
            "urc": "[String or null: EXACT URC from the triggering error event.]",
            "uid": "[String or null: EXACT UID from the triggering error event.]",
            "service_name": "[String or null: EXACT service_name where the triggering error was logged.]",
            "api_endpoint": "[String or null: EXACT API endpoint from the triggering error event.]",
            "transaction_type": "[String or null: EXACT transaction type from the triggering error event.]"
        }
    }
    # Rephrased: less like a formal schema, more like a field guide
    fields_description = f"Description of fields to include in your JSON response (ensure all are present):\\n{json.dumps(fields_description_content, indent=2)}"

    example_json_instruction = (
        "\\nIMPORTANT: Follow the structure of this EXAMPLE JSON for your output. "
        "Populate all fields using the ACTUAL data from THE SPECIFIC TRIGGERING ERROR EVENT provided in the input, NOT these example values. "
        "Focus all descriptions and summaries on THAT specific triggering error. "
        "The 'key_error_log_messages' field must contain ONLY the exact triggering message string from the input.\\n"
    )
    
    example_json_content = {
        "problem_description": "A timeout error occurred in the '3scale API Gateway' service for session 'abc123' and URC 'root123'.",
        "probable_root_cause_summary": "The '3scale API Gateway' service experienced a timeout because the backend service (identified by service_id=123) failed to respond within the configured 5000ms timeout period, as stated in the triggering error message.",
        "key_error_log_messages": ["Backend service timeout for service_id=123, session_id=abc123, URC=root123, timeout=5000ms"],
        "confidence_score": 0.9,
        "associated_identifiers": {
            "session_id": "abc123",
            "urc": "root123",
            "uid": None,
            "service_name": "3scale API Gateway",
            "api_endpoint": None,
            "transaction_type": None
        }
    }
    example_json = f"Example JSON structure to follow:\\n```json\\n{json.dumps(example_json_content, indent=2)}\\n```"
    
    # Construct the full system message
    # Ensuring RAG context is clearly separated and its role defined later in the user message part by analyze_root_cause
    final_system_message = f"{system_message_intro}\\n\\n{fields_description}\\n\\n{example_json_instruction}\\n{example_json}"

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=final_system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Create specialized agents with RAG integration
def create_parser_agent(model: OllamaLLM):
    return create_parser_prompt(rag_manager) | model | StrOutputParser()

def create_correlation_agent(model: OllamaLLM):
    return create_correlation_prompt(rag_manager) | model | StrOutputParser()

def create_anomaly_agent(model: OllamaLLM):
    return create_anomaly_prompt(rag_manager) | model | StrOutputParser()

def create_root_cause_agent(model: OllamaLLM):
    prompt = create_root_cause_prompt(rag_manager)
    # Original parser
    json_parser = JsonOutputParser(pydantic_object=LLMRootCauseAnalysis)
    # Wrap with OutputFixingParser
    output_fixing_parser = OutputFixingParser.from_llm(llm=model, parser=json_parser)
    return prompt | model | output_fixing_parser

def create_recommendation_prompt(rag_manager: RAGManager):
    system_message_intro = (
        "You are an expert system support engineer. "
        "You will be given a root cause analysis for a SINGLE, SPECIFIC triggering error event. "
        "Your task is to provide recommendations that DIRECTLY ADDRESS THIS ANALYZED ERROR. "
        "Suggest 2-3 distinct recommendations: 1-2 for immediate remediation of THIS error, and 1 for prevention of THIS type of error in THIS context. "
        "Your response MUST BE A SINGLE, VALID JSON OBJECT. No other text allowed. "
        "The JSON object must have ONE top-level key: \"recommendations\", value is a list of recommendation objects. "
        "Adhere strictly to the schema. Identifiers MUST be EXACT values from input. "
        "Recommendations MUST be highly specific to the analyzed error (e.g., the specific service like '3scale API Gateway', the specific error type like 'timeout'). Generic advice is NOT acceptable."
    )

    # Define schema description with corrected newlines for the prompt
    recommendation_item_schema = {
        "recommendation_type": "[String: Type: 'Immediate Remediation' or 'Preventive Measure'.]",
        "recommendation_description": "[String: Clear, concise description of action that DIRECTLY addresses the SPECIFIC PROBLEM identified in the provided analysis (e.g., 'Investigate network latency between 3scale API Gateway and its backend service with service_id=123', 'Implement a retry mechanism with backoff in 3scale API Gateway for calls to service_id=123').]",
        "action_steps": ["[List of strings: Specific, actionable steps for THIS recommendation. Max 3-4 steps. E.g., ['Check firewall rules for 3scale API Gateway to backend service_id=123', 'Analyze historical performance metrics for service_id=123'].]", "[List of strings: Specific, actionable steps for THIS recommendation. Max 3-4 steps. E.g., ['Check firewall rules for 3scale API Gateway to backend service_id=123', 'Analyze historical performance metrics for service_id=123'].]", "[List of strings: Specific, actionable steps for THIS recommendation. Max 3-4 steps. E.g., ['Check firewall rules for 3scale API Gateway to backend service_id=123', 'Analyze historical performance metrics for service_id=123'].]"],
        "relevant_documentation": ["[List of strings: Docs relevant to THIS specific recommendation and error type, e.g., ['3scale API Gateway Timeout Configuration Guide', 'Network Troubleshooting for Backend Services']. Use [] if none.]", "[List of strings: Docs relevant to THIS specific recommendation and error type, e.g., ['3scale API Gateway Timeout Configuration Guide', 'Network Troubleshooting for Backend Services']. Use [] if none.]", "[List of strings: Docs relevant to THIS specific recommendation and error type, e.g., ['3scale API Gateway Timeout Configuration Guide', 'Network Troubleshooting for Backend Services']. Use [] if none.]"],
        "applicable_to_identifiers": {
            "session_id": "[String or null: EXACT session_id from input analysis.]",
            "service_name": "[String or null: EXACT service_name from input analysis (e.g., '3scale API Gateway'). MUST BE THE SERVICE FROM THE ANALYSIS.]",
            "urc": "[String or null: EXACT URC from input analysis.]"
        }
    }
    schema_description = (
        f"Schema for EACH object in the 'recommendations' list:\n{json.dumps(recommendation_item_schema, indent=2)}\n"
    )

    example_json_instruction = (
        "\nBelow is an EXAMPLE. Populate with ACTUAL data from the input analysis. Recommendations MUST be specific to the error details provided in that analysis.\n"
    )

    # Define example_json using triple quotes for easier multiline JSON string
    example_json_content = {
        "recommendations": [
            {
                "recommendation_type": "Immediate Remediation",
                "recommendation_description": "Investigate connectivity and current load on the backend service (service_id=123) that the '3scale API Gateway' failed to reach within the timeout period.",
                "action_steps": ["Check logs for backend service (service_id=123) around the time of the incident.", "Verify network path connectivity between '3scale API Gateway' and service_id=123.", "Assess current resource utilization (CPU, memory, network) of service_id=123."],
                "relevant_documentation": ["Troubleshooting Timeouts for 3scale API Gateway", "Backend Service (service_id=123) Operations Guide"],
                "applicable_to_identifiers": {
                    "session_id": "abc123",
                    "service_name": "3scale API Gateway",
                    "urc": "root123"
                }
            },
            {
                "recommendation_type": "Preventive Measure",
                "recommendation_description": "Consider increasing the timeout configuration for backend service_id=123 within '3scale API Gateway' if legitimate calls are expected to take longer, or implement circuit breaker patterns.",
                "action_steps": ["Analyze typical response times for service_id=123 to determine an appropriate timeout.", "Update timeout settings in '3scale API Gateway' configuration for this specific backend.", "Explore adding a circuit breaker library to the gateway's interaction with this backend."],
                "relevant_documentation": ["3scale API Gateway Configuration: Timeouts", "Circuit Breaker Pattern for Microservices"],
                "applicable_to_identifiers": {
                    "session_id": "abc123",
                    "service_name": "3scale API Gateway",
                    "urc": "root123"
                }
            }
        ]
    }
    example_json = f"```json\n{json.dumps(example_json_content, indent=2)}\n```"
    
    final_system_message = f"{system_message_intro}\n\n{schema_description}\n{example_json_instruction}\n{example_json}"

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=final_system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_recommendation_agent(model: OllamaLLM):
    prompt = create_recommendation_prompt(rag_manager)
    # Original parser
    json_parser = JsonOutputParser(pydantic_object=LLMRecommendations)
    # Wrap with OutputFixingParser
    output_fixing_parser = OutputFixingParser.from_llm(llm=model, parser=json_parser)
    return prompt | model | output_fixing_parser

# Define agent nodes with specialized functionality
def parse_logs(state: AgentState) -> AgentState:
    """Parse and normalize logs with UPART attributes."""
    messages = state["messages"]
    raw_logs = state["raw_logs"]
    
    # Get documentation context for log formats
    doc_context = rag_manager.get_relevant_context("log format patterns and UPART attributes")
    
    # Format logs for analysis
    log_content = {}
    for source, entries in raw_logs.items():
        log_content[source] = "\n".join([
            f"[{entry['timestamp']}] [{entry['level']}] {entry['source']}: {entry['message']}"
            for entry in entries
        ])
    
    # Get AI analysis with documentation context
    analysis = create_parser_agent(llm).invoke({
        "messages": messages + [
            HumanMessage(content=f"Documentation Context:\n{''.join(doc_context)}\n\nLog Content:\n{str(log_content)}")
        ]
    })
    
    # Parse and normalize logs with UPART attributes
    normalized_logs = {}
    for source, entries in raw_logs.items():
        normalized_logs[source] = []
        for entry in entries:
            # Extract UPART attributes
            upart = {
                'user': extract_user(entry['message']),
                'parameters': extract_parameters(entry['message']),
                'action': extract_action(entry['message']),
                'resource': extract_resource(entry['message']),
                'time': datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
            }
            normalized_logs[source].append(upart)
    
    return {
        "messages": messages + [HumanMessage(content=analysis)],
        "raw_logs": raw_logs,
        "normalized_logs": normalized_logs,
        "log_content": log_content,
        "correlations": state.get("correlations", []),
        "impact_analysis": state.get("impact_analysis", {}),
        "root_causes": state.get("root_causes", [])
    }

def correlate_events(state: AgentState) -> AgentState:
    """Correlate events using temporal and spatial relationships."""
    messages = state["messages"]
    normalized_logs = state["normalized_logs"]
    
    # Get documentation context for service relationships
    doc_context = rag_manager.get_relevant_context("service dependencies and relationships")
    
    # Correlate events using temporal and spatial rules
    correlated_events = []
    for source1, entries1 in normalized_logs.items():
        for entry1 in entries1:
            related_events = []
            
            # Find temporally related events (within 5 minutes)
            for source2, entries2 in normalized_logs.items():
                for entry2 in entries2:
                    if entry1 != entry2:
                        time_diff = abs((entry1['time'] - entry2['time']).total_seconds())
                        if time_diff <= 300:  # 5 minutes
                            # Check spatial relationships
                            if is_spatially_related(entry1, entry2):
                                related_events.append({
                                    'event': entry2,
                                    'relationship': determine_relationship(entry1, entry2),
                                    'time_diff': time_diff
                                })
            
            if related_events:
                correlated_events.append({
                    'primary_event': entry1,
                    'related_events': related_events,
                    'temporal_window': '5min',
                    'spatial_scope': get_spatial_scope(entry1, related_events)
                })
    
    # Get AI analysis with documentation context
    analysis = create_correlation_agent(llm).invoke({
        "messages": messages + [
            HumanMessage(content=f"Documentation Context:\n{''.join(doc_context)}\n\nCorrelated Events:\n{str(correlated_events)}")
        ]
    })
    
    return {
        "messages": messages + [HumanMessage(content=analysis)],
        "raw_logs": state["raw_logs"],
        "normalized_logs": normalized_logs,
        "log_content": state["log_content"],
        "correlations": correlated_events,
        "impact_analysis": state.get("impact_analysis", {}),
        "root_causes": state.get("root_causes", [])
    }

def detect_anomalies(state: AgentState) -> AgentState:
    """Detect anomalies using unsupervised learning."""
    messages = state["messages"]
    correlated_events = state["correlations"]
    
    # Get documentation context for normal behavior
    doc_context = rag_manager.get_relevant_context("normal system behavior and thresholds")
    
    # Calculate baselines
    baselines = calculate_baselines(correlated_events)
    
    # Detect anomalies
    anomalies = []
    for event_group in correlated_events:
        # Calculate deviation scores
        deviation_scores = calculate_deviation_scores(event_group, baselines)
        
        # Identify anomalies
        if is_anomaly(deviation_scores):
            anomalies.append({
                'event_group': event_group,
                'deviation_scores': deviation_scores,
                'anomaly_type': determine_anomaly_type(deviation_scores),
                'confidence': calculate_anomaly_confidence(deviation_scores)
            })
    
    # Get AI analysis with documentation context
    analysis = create_anomaly_agent(llm).invoke({
        "messages": messages + [
            HumanMessage(content=f"Documentation Context:\n{''.join(doc_context)}\n\nDetected Anomalies:\n{str(anomalies)}")
        ]
    })
    
    return {
        "messages": messages + [HumanMessage(content=analysis)],
        "raw_logs": state["raw_logs"],
        "normalized_logs": state["normalized_logs"],
        "log_content": state["log_content"],
        "correlations": correlated_events,
        "impact_analysis": state.get("impact_analysis", {}),
        "root_causes": state.get("root_causes", [])
    }

def analyze_root_cause(state: AgentState) -> AgentState:
    messages = state["messages"]
    correlations = state["correlations"]
    logger.info("--- DEBUG: Correlations passed to RCA ---")
    logger.info(json.dumps(correlations, indent=2, default=str))

    doc_context_for_analysis = rag_manager.get_relevant_context("system architecture, known issues, error types and patterns")
    
    programmatically_identified_root_causes = []
    static_error_types = [
        'rate_limit', 'authentication', 'backend_service_health_check',
        'backend_service_timeout', 'timeout', 'connection', 'queue',
        'performance', 'validation', 'business', 'system'
    ]
    # Dynamic error types could be added here if needed from RAG
    root_cause_error_types = static_error_types

    for correlation in correlations:
        for error_chain in correlation.get('error_chains', []):
            if error_chain.get('impact_level') in ['HIGH', 'CRITICAL', 'MEDIUM', 'LOW']:
                for error_details in error_chain.get('errors', []):
                    current_error_type = error_details.get('error_type')
                    if current_error_type in root_cause_error_types:
                        programmatically_identified_root_causes.append({
                            'timestamp': error_details.get('timestamp'),
                            'triggering_error_message': error_details.get('message'),
                            'triggering_error_type': current_error_type,
                            'source_service_from_log': error_details.get('source'), # Original source from log
                            'session_id': correlation.get('session_id'),
                            'root_urc_of_session': correlation.get('root_urc'),
                            'error_specific_urc': error_details.get('urc'),
                            'error_specific_uid': error_details.get('uid'),
                            'error_specific_api_endpoint': error_details.get('api_endpoint'),
                            'error_specific_transaction_type': error_details.get('transaction_type'),
                            'overall_chain_impact': error_chain.get('impact_level'),
                            'full_error_chain_details_for_context': error_chain 
                        })
                        break # Process this chain once
    
    logger.info(f"RCA_DEBUG: Programmatically identified {len(programmatically_identified_root_causes)} potential root cause events.")

    final_rca_results_with_llm_analysis = []

    for idx, rc_item in enumerate(programmatically_identified_root_causes):
        logger.info(f"RCA_DEBUG: Processing individual root cause item {idx+1}/{len(programmatically_identified_root_causes)}: URC {rc_item.get('root_urc_of_session')} - Type {rc_item.get('triggering_error_type')}")

        # Prepare specific input for LLMRootCauseAnalysis
        # Ensure all expected keys for RootCauseIdentifiers are present, defaulting to None if missing from rc_item
        identifiers_for_llm = RootCauseIdentifiers(
            session_id=rc_item.get('session_id'),
            urc=rc_item.get('error_specific_urc') or rc_item.get('root_urc_of_session'), # Prioritize error specific URC
            uid=rc_item.get('error_specific_uid'),
            service_name=rc_item.get('source_service_from_log'), # Use the original log source as service name
            api_endpoint=rc_item.get('error_specific_api_endpoint'),
            transaction_type=rc_item.get('error_specific_transaction_type')
        ).model_dump(exclude_none=True) # Pass as dict, exclude_none for cleaner input if some are None

        # Construct the detailed input message for initial analysis LLM
        # This approach helps with complex f-string formatting and escaping
        initial_analysis_parts = [
            f"Analyze the following specific error event for its root cause. ",
            f"Focus EXCLUSIVELY on THIS triggering error event and its immediate details. ",
            f"The 'Full Error Chain' and 'Documentation Context' are for overall context ONLY; your core analysis (problem_description, probable_root_cause_summary) MUST be about THIS specific error.\\n",
            f"Key Details for THIS error (Your analysis MUST use these exact details):",
            f"- Session ID: {rc_item.get('session_id')}",
            f"- URC of this error event: {rc_item.get('error_specific_urc', 'N/A')}",
            f"- Root URC of session (for context only): {rc_item.get('root_urc_of_session', 'N/A')}",
            f"- UID of this error event: {rc_item.get('error_specific_uid', 'N/A')}",
            f"- Service where error logged (use this for service_name in output): {rc_item.get('source_service_from_log')}",
            f"- Triggering Error Type (use this for descriptions): {rc_item.get('triggering_error_type')}",
            f"- EXACT Triggering Error Message (this is the primary subject of your analysis): '{rc_item.get('triggering_error_message')}'", # No extra quotes around the message itself here
            f"- API Endpoint (if any): {rc_item.get('error_specific_api_endpoint', 'N/A')}",
            f"- Transaction Type (if any): {rc_item.get('error_specific_transaction_type', 'N/A')}\\n",
            f"Full Error Chain for broader context (DO NOT make this the subject of your 'problem_description' or 'probable_root_cause_summary'):",
            json.dumps(rc_item.get('full_error_chain_details_for_context'), indent=2, default=str) + "\\n",
            f"Documentation Context (for general understanding of error types or components - ONLY use if directly relevant to explaining THIS SPECIFIC error event):",
            ''.join(doc_context_for_analysis) + "\\n",
            f"Instructions for filling the JSON object:",
            f"1. Populate ALL fields according to schema.",
            f"2. 'associated_identifiers': Use EXACT values from 'Key Details for THIS error' (session_id, error_specific_urc, error_specific_uid, source_service_from_log as service_name, etc.).",
            f"3. 'key_error_log_messages': MUST be a list with ONE string: the UNMODIFIED 'EXACT Triggering Error Message' (which is '{rc_item.get('triggering_error_message')}'). No other text/prefixes.",
            f"4. 'problem_description': Describe THE PROBLEM shown by the 'EXACT Triggering Error Message' in the 'Service where error logged'. Be specific to THIS event.",
            f"5. 'probable_root_cause_summary': Explain THE LIKELY CAUSE of the 'EXACT Triggering Error Message' occurring in the 'Service where error logged'. Relate directly to THIS specific message and service."
        ]
        initial_analysis_input_content = "\n".join(initial_analysis_parts)

        logger.info(f"RCA_DEBUG: Input content for initial analysis LLM call:\\n{initial_analysis_input_content}")

        raw_initial_analysis_data = create_root_cause_agent(llm).invoke({
            "messages": [HumanMessage(content=initial_analysis_input_content)]
        })
        logger.info(f"RCA_DEBUG: Raw initial analysis data from LLM for item {idx+1}: {raw_initial_analysis_data}")

        validated_initial_analysis_obj = None
        llm_generated_rca_dict = None
        try:
            validated_initial_analysis_obj = LLMRootCauseAnalysis.model_validate(raw_initial_analysis_data)
            llm_generated_rca_dict = validated_initial_analysis_obj.model_dump(exclude_none=True)
            logger.info(f"RCA_DEBUG: Successfully validated initial LLM analysis for item {idx+1}: {llm_generated_rca_dict}")
        except Exception as e:
            logger.error(f"RCA_DEBUG: Failed to validate LLM initial_analysis_data for item {idx+1}. Data: {raw_initial_analysis_data}. Error: {e}")
            llm_generated_rca_dict = {
                "error": "Failed to validate LLM output for initial analysis against required schema.",
                "raw_llm_output": raw_initial_analysis_data,
                "validation_error_details": str(e)
            }

        # Prepare for recommendation LLM call
        recommendation_parts = [
            f"You are given the following root cause analysis FOR A SINGLE, SPECIFIC TRIGGERING ERROR:\\n",
            json.dumps(llm_generated_rca_dict, indent=2, default=str) + "\\n",
            f"Key context for THIS specific error event (Your recommendations MUST address this specific context):",
            f"- Original Triggering Error Message: {rc_item.get('triggering_error_message')}",
            f"- Original Error Type: {rc_item.get('triggering_error_type')}",
            f"- Session ID: {rc_item.get('session_id')}",
            f"- Service where error logged (recommendations should target this service): {rc_item.get('source_service_from_log')}\\n",
            f"Documentation Context (for general understanding - recommendations must be specific to the analyzed error, NOT generic documentation points unless they directly solve THIS specific analyzed error):",
            ''.join(doc_context_for_analysis) + "\\n",
            f"Instructions for generating recommendations JSON:",
            f"1. Create recommendations that DIRECTLY address the 'problem_description' and 'probable_root_cause_summary' from the provided analysis for the SPECIFIC error (e.g., if analysis says 'Timeout in X due to Y', recommend how to fix Y or handle timeouts in X related to Y).",
            f"2. Populate ALL fields per schema.",
            f"3. 'recommendation_description': Must be a direct consequence of the analyzed root cause FOR THIS SPECIFIC ERROR. E.g., if the problem is a timeout in '{rc_item.get('source_service_from_log')}', recommendations must relate to fixing/handling timeouts in '{rc_item.get('source_service_from_log')}'.",
            f"4. 'applicable_to_identifiers': Use EXACT values from the 'Key context' (Session ID, and '{rc_item.get('source_service_from_log')}' as service_name). URC should be from the analysis if relevant to the recommendation for THIS error.",
            f"5. Recommendations MUST be specific to the problem outlined in the provided 'llm_initial_analysis' and the 'Key context for this specific error event'. Do not recommend solutions for other problems that might be in the broader documentation or error chain unless they are the direct cause/solution for THIS specific triggering error."
        ]
        recommendation_input_content = "\n".join(recommendation_parts)
        
        logger.info(f"RCA_DEBUG: Input content for recommendation LLM call for item {idx+1}:\\n{recommendation_input_content}")

        raw_recommendations_data = create_recommendation_agent(llm).invoke({
            "messages": [HumanMessage(content=recommendation_input_content)]
        })
        logger.info(f"RCA_DEBUG: Raw recommendations data from LLM for item {idx+1}: {raw_recommendations_data}")
        
        llm_generated_recommendations_dict = None
        try:
            validated_recommendations_obj = LLMRecommendations.model_validate(raw_recommendations_data)
            llm_generated_recommendations_dict = validated_recommendations_obj.model_dump(exclude_none=True)
            logger.info(f"RCA_DEBUG: Successfully validated LLM recommendations for item {idx+1}: {llm_generated_recommendations_dict}")
        except Exception as e:
            logger.error(f"RCA_DEBUG: Failed to validate LLM recommendations_data for item {idx+1}. Data: {raw_recommendations_data}. Error: {e}")
            llm_generated_recommendations_dict = {
                "error": "Failed to validate LLM output for recommendations against required schema.",
                "raw_llm_output": raw_recommendations_data,
                "validation_error_details_recommendation": str(e) # Changed key to avoid clash
            }
        
        # Combine original programmatic data with LLM analysis and recommendations
        final_rc_item = {
            **rc_item, # Original data like timestamps, triggering messages, full chain
            "llm_initial_analysis": llm_generated_rca_dict,
            "llm_recommendations": llm_generated_recommendations_dict
        }
        final_rca_results_with_llm_analysis.append(final_rc_item)
    
    final_summary_for_state_message = f"Processed {len(programmatically_identified_root_causes)} root cause events, generating individual LLM analysis and recommendations for each."
    logger.info(f"RCA_DEBUG: analyze_root_cause returning {len(final_rca_results_with_llm_analysis)} items.")

    return {
        "messages": messages + [HumanMessage(content=final_summary_for_state_message)],
        "raw_logs": state["raw_logs"],
        "normalized_logs": state["normalized_logs"],
        "log_content": state["log_content"],
        "correlations": correlations,
        "impact_analysis": state.get("impact_analysis", {}),
        "root_causes": final_rca_results_with_llm_analysis
    }

# Helper functions for UPART extraction
def extract_user(message: str) -> str:
    """Extract user information from log message."""
    patterns = [
        r'user=([^\s]+)',
        r'client_id=([^\s]+)',
        r'user_id=([^\s]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return "unknown"

def extract_parameters(message: str) -> Dict[str, str]:
    """Extract parameters from log message."""
    params = {}
    param_pattern = r'(\w+)=([^\s]+)'
    for match in re.finditer(param_pattern, message):
        key, value = match.groups()
        if key not in ['user', 'client_id', 'user_id']:
            params[key] = value
    return params

def extract_action(message: str) -> str:
    """Extract action from log message."""
    action_patterns = [
        r'action=([^\s]+)',
        r'operation=([^\s]+)',
        r'performing ([^\s]+)',
        r'executing ([^\s]+)'
    ]
    for pattern in action_patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return "unknown"

def extract_resource(message: str) -> str:
    """Extract resource information from log message."""
    resource_patterns = [
        r'resource=([^\s]+)',
        r'endpoint=([^\s]+)',
        r'path=([^\s]+)',
        r'service=([^\s]+)'
    ]
    for pattern in resource_patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return "unknown"

# Helper functions for correlation analysis
def is_spatially_related(event1: Dict, event2: Dict) -> bool:
    """Check if two events are spatially related."""
    # Check service relationships
    if event1['resource'] == event2['resource']:
        return True
    
    # Check parameter relationships
    common_params = set(event1['parameters'].keys()) & set(event2['parameters'].keys())
    if common_params:
        return True
    
    return False

def determine_relationship(event1: Dict, event2: Dict) -> str:
    """Determine the relationship between two events."""
    if event1['time'] < event2['time']:
        return 'precedes'
    elif event1['time'] > event2['time']:
        return 'follows'
    else:
        return 'concurrent'

def get_spatial_scope(event: Dict, related_events: List[Dict]) -> Dict:
    """Get the spatial scope of an event and its related events."""
    services = {event['resource']}
    parameters = set(event['parameters'].keys())
    
    for related in related_events:
        services.add(related['event']['resource'])
        parameters.update(related['event']['parameters'].keys())
    
    return {
        'services': list(services),
        'parameters': list(parameters)
    }

# Helper functions for anomaly detection
def calculate_baselines(events: List[Dict]) -> Dict:
    """Calculate baselines for various metrics."""
    baselines = {
        'response_time': [],
        'error_rate': [],
        'resource_usage': [],
        'service_patterns': defaultdict(int)
    }
    
    for event in events:
        # Extract metrics from parsed_message
        if 'primary_event' in event:
            parsed_msg = event['primary_event'].get('parsed_message', {})
            metrics = parsed_msg.get('metrics', {})
            
            # Add metrics to baselines
            if 'latency' in metrics:
                baselines['response_time'].append(float(metrics['latency'][0]))
            if 'error_rate' in metrics:
                baselines['error_rate'].append(float(metrics['error_rate'][0]))
            if 'memory' in metrics:
                baselines['resource_usage'].append(float(metrics['memory'][0]))
            
            # Track service patterns
            service_name = parsed_msg.get('service_name', 'unknown')
            baselines['service_patterns'][service_name] += 1
            
            # Also process related events
            for related in event.get('related_events', []):
                related_parsed = related['event'].get('parsed_message', {})
                related_metrics = related_parsed.get('metrics', {})
                
                if 'latency' in related_metrics:
                    baselines['response_time'].append(float(related_metrics['latency'][0]))
                if 'error_rate' in related_metrics:
                    baselines['error_rate'].append(float(related_metrics['error_rate'][0]))
                if 'memory' in related_metrics:
                    baselines['resource_usage'].append(float(related_metrics['memory'][0]))
                
                related_service = related_parsed.get('service_name', 'unknown')
                baselines['service_patterns'][related_service] += 1
    
    # Calculate statistics
    for metric in ['response_time', 'error_rate', 'resource_usage']:
        if baselines[metric]:
            baselines[metric] = {
                'mean': sum(baselines[metric]) / len(baselines[metric]),
                'std': calculate_std(baselines[metric])
            }
        else:
            baselines[metric] = {'mean': 0, 'std': 1}  # Default values if no data
    
    return baselines

def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if not values:
        return 0
    mean = sum(values) / len(values)
    squared_diff_sum = sum((x - mean) ** 2 for x in values)
    return (squared_diff_sum / len(values)) ** 0.5

def calculate_deviation_scores(event_group: Dict, baselines: Dict) -> Dict:
    """Calculate deviation scores for an event group."""
    scores = {}
    
    # Get metrics from primary event
    parsed_msg = event_group['primary_event'].get('parsed_message', {})
    metrics = parsed_msg.get('metrics', {})
    
    # Calculate response time deviation
    if 'latency' in metrics:
        rt = float(metrics['latency'][0])
        rt_baseline = baselines['response_time']
        scores['response_time'] = abs(rt - rt_baseline['mean']) / rt_baseline['std']
    
    # Calculate error rate deviation
    if 'error_rate' in metrics:
        er = float(metrics['error_rate'][0])
        er_baseline = baselines['error_rate']
        scores['error_rate'] = abs(er - er_baseline['mean']) / er_baseline['std']
    
    # Calculate resource usage deviation
    if 'memory' in metrics:
        ru = float(metrics['memory'][0])
        ru_baseline = baselines['resource_usage']
        scores['resource_usage'] = abs(ru - ru_baseline['mean']) / ru_baseline['std']
    
    return scores

def is_anomaly(deviation_scores: Dict) -> bool:
    """Determine if an event group is anomalous."""
    threshold = 2.0  # 2 standard deviations
    return any(score > threshold for score in deviation_scores.values())

def determine_anomaly_type(deviation_scores: Dict) -> str:
    """Determine the type of anomaly."""
    max_score = max(deviation_scores.items(), key=lambda x: x[1])
    return f"{max_score[0]}_anomaly"

def calculate_anomaly_confidence(deviation_scores: Dict) -> float:
    """Calculate confidence score for an anomaly."""
    max_score = max(deviation_scores.values())
    return min(1.0, max_score / 5.0)  # Normalize to [0,1]

# Update the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_logs", parse_logs)
# workflow.add_node("correlate_events", correlate_events)
# workflow.add_node("detect_anomalies", detect_anomalies)
workflow.add_node("analyze_root_cause", analyze_root_cause)

# Add edges
# workflow.add_edge("parse_logs", "correlate_events")
# workflow.add_edge("correlate_events", "detect_anomalies")
# workflow.add_edge("detect_anomalies", "analyze_root_cause")
workflow.add_edge("parse_logs", "analyze_root_cause")

# Set entry point
workflow.set_entry_point("parse_logs")

# Compile the workflow
app = workflow.compile()

def save_analysis_to_json(analysis_results: Dict, output_dir: str = "analysis_output") -> Dict[str, str]:
    """Save analysis results to JSON files in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert sets to lists and datetime objects to strings for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    # Convert messages to serializable format
    def convert_messages(messages):
        return [
            {
                "type": msg.type,
                "content": msg.content
            } for msg in messages
        ]
    
    # Prepare the session-keyed data
    sessions_data = {}
    for correlation_item in analysis_results.get("correlations", []):
        session_id = correlation_item.get("session_id", "unknown_session")
        # Ensure all parts of the correlation item are JSON serializable
        sessions_data[session_id] = convert_for_json(correlation_item)

    # Prepare the full analysis data with new structure
    full_analysis = {
        "timestamp": timestamp,
        "general_log_analysis": {
            "messages": convert_messages(analysis_results.get("messages", [])),
            "log_content": analysis_results.get("log_content", {}), # log_content is already a dict of strings
            "documentation_context": analysis_results.get("documentation_context", {}) # Assuming this is already serializable
        },
        "sessions": sessions_data,
        # Keep impact_analysis if it exists and is structured by session or general
        # Based on previous AgentState, impact_analysis was a general Dict, not necessarily session-keyed directly.
        # If it needs to be session-keyed, that logic would be in its generation, not here.
        # For now, let's assume it's a general analysis part or handle its structure if known.
        # Given it was removed from saving earlier, this key might not be present or used.
        # "impact_analysis": convert_for_json(analysis_results.get("impact_analysis", {})) # Retaining for now if present
    }
    
    # Save full analysis
    full_analysis_path = os.path.join(output_dir, f"full_analysis_{timestamp}.json")
    with open(full_analysis_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    return {
        "full_analysis": full_analysis_path
    }

def format_log_content(log_entries: Dict[str, List[dict]]) -> Dict[str, str]:
    """Format log entries into a structured format for analysis."""
    formatted_content = {}
    for source, entries in log_entries.items():
        # Group entries by timestamp windows (1-minute intervals)
        time_windows = defaultdict(list)
        for entry in entries:
            if entry:  # Skip None entries (failed parsing)
                timestamp = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
                window_key = timestamp.strftime('%Y-%m-%d %H:%M')
                time_windows[window_key].append(entry)
        
        # Format entries by time window
        formatted_windows = {}
        for window, entries in time_windows.items():
            formatted_windows[window] = {
                'entries': entries,
                'error_count': sum(1 for e in entries if e['level'] in ['ERROR', 'WARN']),
                'info_count': sum(1 for e in entries if e['level'] == 'INFO'),
                'services': list(set(e['source'] for e in entries))
            }
        
        formatted_content[source] = formatted_windows
    
    return formatted_content

def save_root_cause_to_json(analysis_results: Dict, output_dir: str = "analysis_output") -> str:
    """Save root cause analysis results to a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_causes = analysis_results.get("root_causes", [])
    logger.info(f"SAVE_JSON_DEBUG: root_causes list received in save_root_cause_to_json: {json.dumps(root_causes, indent=2, default=str)}") # ADDED
    root_cause_path = os.path.join(output_dir, f"root_cause_analysis_{timestamp}.json")
    with open(root_cause_path, 'w') as f:
        json.dump({"timestamp": timestamp, "root_causes": root_causes}, f, indent=2)
    return root_cause_path

# Pydantic models for structured LLM output - REVISED
class RootCauseIdentifiers(BaseModel):
    session_id: Optional[str] = Field(None, description="The EXACT session_id from the input data relevant to this root cause, if applicable.")
    urc: Optional[str] = Field(None, description="The EXACT URC from the input data if specifically relevant to this root cause.")
    uid: Optional[str] = Field(None, description="The EXACT UID from the input data if specifically relevant to this root cause.")
    service_name: Optional[str] = Field(None, description="The EXACT service name (e.g., 'payment_service', 'api_gateway') from the input data relevant to this root cause.")
    api_endpoint: Optional[str] = Field(None, description="The EXACT API endpoint from the input data, if relevant.")
    transaction_type: Optional[str] = Field(None, description="The EXACT transaction type from the input data, if relevant.")

class LLMRootCauseAnalysis(BaseModel):
    problem_description: str = Field(description="A brief (1-2 sentence) description of the core problem observed.")
    probable_root_cause_summary: str = Field(description="A concise summary of the MOST LIKELY root cause (e.g., 'Authentication failure in payment_service due to invalid credentials'). Do NOT include specific identifiers like session_id here; use the 'associated_identifiers' field for that.")
    key_error_log_messages: List[str] = Field(description="List of 1-3 EXACT log message strings from the input error chain that are most indicative of THIS specific root cause.")
    confidence_score: Optional[float] = Field(None, description="A score between 0.0 and 1.0 indicating confidence in this identified root cause (e.g., 0.85). Use null if not determinable.")
    associated_identifiers: RootCauseIdentifiers = Field(description="Key EXACT identifiers from the input data that are directly associated with this specific root cause analysis.")

    @validator('key_error_log_messages')
    def check_log_messages(cls, v):
        if not v or not all(isinstance(item, str) for item in v):
            raise ValueError('key_error_log_messages must be a non-empty list of strings')
        if any('[' in item or ']' in item for item in v):
             logger.warning(f"LLM_VALIDATION_WARN: key_error_log_messages item may contain placeholder-like brackets: {v}")
        return v

class RecommendationIdentifiers(BaseModel):
    session_id: Optional[str] = Field(None, description="The EXACT session_id this recommendation applies to, if specific. Use null if general.")
    service_name: Optional[str] = Field(None, description="The EXACT service name this recommendation applies to, if specific. Use null if general.")
    urc: Optional[str] = Field(None, description="The EXACT URC this recommendation applies to, if specific. Use null if general.")

class RecommendationItem(BaseModel):
    recommendation_type: str = Field(description="Type of recommendation (e.g., 'Immediate Remediation', 'Preventive Measure', 'Further Investigation').")
    recommendation_description: str = Field(description="A clear, concise description of the recommended action. Do NOT include specific identifiers like session_id here; use 'applicable_to_identifiers' for that.")
    action_steps: Optional[List[str]] = Field(default_factory=list, description="Specific, actionable steps to implement the recommendation. Max 3-4 steps.")
    relevant_documentation: Optional[List[str]] = Field(default_factory=list, description="References to specific titles or sections in documentation that support this recommendation (e.g., 'Firewall Configuration Guide, Section 2.3').")
    applicable_to_identifiers: Optional[RecommendationIdentifiers] = Field(None, description="Key EXACT identifiers from input data this recommendation primarily targets. Use null or omit if recommendation is general.")

class LLMRecommendations(BaseModel):
    recommendations: List[RecommendationItem] = Field(description="A list of structured recommendations based on the root cause analysis.")

def main():
    """Main function to run the log analysis workflow."""
    try:
        # Read log files
        log_files = {
            '3scale_api_gateway': 'logs/3scale_api_gateway.log',
            'tibco_businessworks': 'logs/tibco_businessworks.log',
            'payment_service': 'logs/payment_service.log'
        }
        
        # Parse logs into a dictionary of log entries
        raw_logs = {}
        for service, file_path in log_files.items():
            with open(file_path, 'r') as f:
                # Parse each line into a structured format
                raw_logs[service] = [
                    parse_log_line(line.strip())
                    for line in f.readlines()
                    if line.strip()  # Skip empty lines
                ]
        
        # Format log content for analysis
        log_content = format_log_content(raw_logs)
        
        # Correlate events across logs and analyze impact
        correlations = correlate_logs(raw_logs)
        
        # Initialize state with all required fields
        initial_state = {
            "messages": [],
            "raw_logs": raw_logs,
            "normalized_logs": {},
            "log_content": log_content,
            "correlations": correlations,
            "impact_analysis": {},
            "root_causes": []
        }
        
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        logger.info(f"MAIN_DEBUG: final_state['root_causes'] before saving: {json.dumps(final_state.get('root_causes', []), indent=2, default=str)}") # ADDED
        # Save analysis results
        output_files = save_analysis_to_json(final_state)
        
        # Save root cause analysis separately
        root_cause_path = save_root_cause_to_json(final_state)
        print(f"Root cause analysis saved to: {root_cause_path}")
        
        return output_files
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {str(e)}")
        raise
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 