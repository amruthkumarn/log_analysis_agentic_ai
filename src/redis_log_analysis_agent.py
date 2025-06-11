from typing import Annotated, Sequence, TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama import OllamaEmbeddings
import os
import re
from collections import defaultdict
import json
from .document_processor import DocumentProcessor
import logging
from pydantic import BaseModel, Field, validator
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from langgraph.checkpoint.redis import RedisSaver
import uuid
import threading
import time
from langchain.output_parsers import OutputFixingParser
from .redis_client import get_redis_url

# --- Setup ---
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

project_root = get_project_root()
LOG_FILE_PATH = project_root / "logs" / "redis_log_analysis.log"
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- Pydantic Models for Structured LLM Output ---
class RootCauseIdentifiers(BaseModel):
    session_id: Optional[str] = Field(None, description="The EXACT session_id from the input data relevant to this root cause, if applicable.")
    urc: Optional[str] = Field(None, description="The EXACT URC from the input data if specifically relevant to this root cause.")
    uid: Optional[str] = Field(None, description="The EXACT UID from the input data if specifically relevant to this root cause.")
    service_name: Optional[str] = Field(None, description="The EXACT service name (e.g., 'payment_service', 'api_gateway') from the input data relevant to this root cause.")
    api_endpoint: Optional[str] = Field(None, description="The EXACT API endpoint from the input data, if relevant.")
    transaction_type: Optional[str] = Field(None, description="The EXACT transaction type from the input data, if relevant.")

class LLMRootCauseAnalysis(BaseModel):
    problem_description: str = Field(description="A brief (1-2 sentence) description of the core problem observed.")
    probable_root_cause_summary: str = Field(description="A concise summary of the MOST LIKELY root cause FOR THE PROVIDED TRIGGERING ERROR EVENT.]")
    key_error_log_messages: List[str] = Field(description="List of 1-3 EXACT log message strings from the input error chain that are most indicative of THIS specific root cause.")
    confidence_score: Optional[float] = Field(None, description="A score between 0.0 and 1.0 indicating confidence in this identified root cause (e.g., 0.85). Use null if not determinable.")
    associated_identifiers: RootCauseIdentifiers = Field(description="Key EXACT identifiers from the input data that are directly associated with this specific root cause analysis.")

    @validator('key_error_log_messages', pre=True)
    def ensure_list_of_strings(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and all(isinstance(s, str) for s in v):
            return v
        raise TypeError('key_error_log_messages must be a string or a list of strings')

    @validator('key_error_log_messages')
    def check_log_messages(cls, v):
        if not v or not all(isinstance(item, str) for item in v):
            raise ValueError('key_error_log_messages must be a non-empty list of strings')
        return v

class RecommendationIdentifiers(BaseModel):
    session_id: Optional[str] = Field(None, description="The EXACT session_id this recommendation applies to, if specific. Use null if general.")
    service_name: Optional[str] = Field(None, description="The EXACT service name this recommendation applies to, if specific. Use null if general.")
    urc: Optional[str] = Field(None, description="The EXACT URC this recommendation applies to, if specific. Use null if general.")

class RecommendationItem(BaseModel):
    recommendation_type: str = Field(description="Type of recommendation (e.g., 'Immediate Remediation', 'Preventive Measure').")
    recommendation_description: str = Field(description="A clear, concise description of the recommended action.")
    action_steps: Optional[List[str]] = Field(default_factory=list, description="Specific, actionable steps to implement the recommendation.")
    relevant_documentation: Optional[List[str]] = Field(default_factory=list, description="References to documentation.")
    applicable_to_identifiers: Optional[RecommendationIdentifiers] = Field(None, description="Identifiers this recommendation applies to.")

class LLMRecommendations(BaseModel):
    recommendations: List[RecommendationItem] = Field(description="A list of structured recommendations.")


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    raw_logs: Annotated[Dict[str, List[dict]], "Raw log entries for the session"]
    log_content: Annotated[Dict[str, str], "The log content from different sources"]
    correlations: Annotated[List[Dict], "Correlated events across logs"]
    root_causes: Annotated[List[Dict], "Identified root causes"]
    session_id: str


# --- RAG Manager (Adapted for Redis) ---
class RAGManager:
    def __init__(self, docs_dir: str = "documentation"):
        self.project_root = get_project_root()
        self.docs_dir = self.project_root / docs_dir
        self.document_processor = DocumentProcessor()
        # The process_documents now happens on a per-session basis if needed
        # but the document_processor itself is initialized here.

    def get_relevant_context(self, session_id: str, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documentation context for a query for a given session."""
        logger.info(f"[{session_id}] RAGManager: Getting relevant context for query: '{query}'")
        # Use query_documentation directly instead of get_documentation_context
        relevant_docs = self.document_processor.query_documentation(session_id, query, k=k)
        
        # Extract content from the returned documents
        context_list = []
        for doc in relevant_docs:
            if isinstance(doc, dict) and 'content' in doc:
                context_list.append(doc['content'])
        
        logger.info(f"[{session_id}] RAGManager: Found {len(context_list)} relevant context snippets.")
        return context_list

# Initialize once
rag_manager = RAGManager()

# --- LLM and Prompt Setup ---
def get_llm():
    return OllamaLLM(model="llama3.2:1b", base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))

def create_root_cause_prompt():
    # This is the detailed prompt from log_analysis_agent.py
    system_message = """
    You are an expert system and network administrator tasked with identifying a root cause for a SPECIFIC TRIGGERING ERROR EVENT provided to you.
    Analyze ONLY the provided single triggering error event, its immediate details, and its position within the larger error chain (also provided).
    Your response MUST BE A SINGLE, VALID JSON OBJECT.

    The JSON object must contain the following fields:
    - "problem_description": [String: Brief 1-2 sentence description of the core problem observed FOR THE PROVIDED TRIGGERING ERROR EVENT.]
    - "probable_root_cause_summary": [String: Concise summary of the MOST LIKELY root cause FOR THE PROVIDED TRIGGERING ERROR EVENT.]
    - "key_error_log_messages": [List containing EXACTLY ONE string. This string itself MUST be the UNMODIFIED 'EXACT Triggering Error Message' from the input.]
    - "confidence_score": [Float or null: Score 0.0-1.0 for confidence.]
    - "associated_identifiers": {
        "session_id": "[String or null: EXACT session_id from input.]",
        "urc": "[String or null: EXACT URC from the triggering error event.]",
        "uid": "[String or null: EXACT UID from the triggering error event.]",
        "service_name": "[String or null: EXACT service_name where the triggering error was logged.]",
        "api_endpoint": "[String or null: EXACT API endpoint from the triggering error event.]",
        "transaction_type": "[String or null: EXACT transaction type from the triggering error event.]"
    }
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

def create_recommendation_prompt():
    # This is the detailed prompt from log_analysis_agent.py
    system_message = """
    You are an expert system support engineer. You will be given a root cause analysis for a SINGLE, SPECIFIC triggering error event.
    Your task is to provide recommendations that DIRECTLY ADDRESS THIS ANALYZED ERROR.
    Your response MUST BE A SINGLE, VALID JSON OBJECT with ONE top-level key: "recommendations", which is a list of recommendation objects.
    
    Each recommendation object in the list must follow this schema:
    - "recommendation_type": "[String: 'Immediate Remediation' or 'Preventive Measure'.]"
    - "recommendation_description": "[String: Clear, concise description of action that DIRECTLY addresses the SPECIFIC PROBLEM.]"
    - "action_steps": "[List of strings: Specific, actionable steps. Max 3-4 steps.]"
    - "relevant_documentation": "[List of strings: Docs relevant to THIS specific recommendation. Use [] if none.]"
    - "applicable_to_identifiers": {
        "session_id": "[String or null: EXACT session_id from input analysis.]",
        "service_name": "[String or null: EXACT service_name from input analysis.]",
        "urc": "[String or null: EXACT URC from input analysis.]"
    }
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])

# --- Agent and Chain Creation ---
def create_root_cause_agent(llm: OllamaLLM):
    prompt = create_root_cause_prompt()
    parser = JsonOutputParser(pydantic_object=LLMRootCauseAnalysis)
    output_fixing_parser = OutputFixingParser.from_llm(llm=llm, parser=parser)
    return prompt | llm | output_fixing_parser

def create_recommendation_agent(llm: OllamaLLM):
    prompt = create_recommendation_prompt()
    parser = JsonOutputParser(pydantic_object=LLMRecommendations)
    output_fixing_parser = OutputFixingParser.from_llm(llm=llm, parser=parser)
    return prompt | llm | output_fixing_parser

# --- Log Parsing and Correlation Logic (from log_analysis_agent.py) ---
def parse_log_line(line: str) -> dict:
    pattern = r'\[(.*?)\] \[(.*?)\] (.*?): (.*)'
    match = re.match(pattern, line)
    if match:
        timestamp, level, source, message = match.groups()
        return {
            'timestamp': timestamp, 'level': level, 'source': source, 'message': message,
            'parsed_message': parse_message_content(message, level)
        }
    return None

def read_log_file(file_path: str) -> List[dict]:
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
    
    # Extract URC
    urc_match = re.search(r'URC=([^,\s]+)', message)
    if urc_match:
        parsed['urc'] = urc_match.group(1)
    
    # Extract UID
    uid_match = re.search(r'UID=([^,\s]+)', message)
    if uid_match:
        parsed['uid'] = uid_match.group(1)
    
    # Check for login event
    if 'logged in' in message.lower():
        parsed['is_login'] = True
    
    # Check for API request/response
    if 'request received' in message.lower():
        parsed['is_request'] = True
    elif 'response sent' in message.lower():
        parsed['is_response'] = True
    
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
        elif 'authentication' in message.lower():
            parsed['error_type'] = 'authentication'
        elif 'connection' in message.lower():
            parsed['error_type'] = 'connection'
        elif 'validation' in message.lower():
            parsed['error_type'] = 'validation'
        elif 'business' in message.lower():
            parsed['error_type'] = 'business'
        elif is_error_log_level and not parsed.get('error_type'):
            parsed['error_type'] = 'system'
        elif not parsed.get('error_type') and contains_error_keywords:
             parsed['error_type'] = 'unknown_error'

        # Default severity based on log level
        if level.upper() == 'ERROR':
            parsed['severity'] = 'HIGH'
        elif level.upper() == 'WARN':
            parsed['severity'] = 'MEDIUM'
    
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
    
    return parsed

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
    
    root_entry = next((e for e in entries if e['parsed_message'].get('is_login') and e['parsed_message'].get('urc')), None)
    
    if not root_entry:
        logger.warning("No login entry with URC found in the provided entries for this session.")
        return {}
    
    nodes = {}
    root_urc = root_entry['parsed_message']['urc']
    root_node_obj = APICallNode(root_urc, root_entry)
    nodes[root_urc] = root_node_obj
    
    sorted_entries = sorted(entries, key=lambda x: datetime.fromisoformat(x.get('@timestamp', '1970-01-01T00:00:00Z').replace('Z','+00:00')))

    for entry in sorted_entries:
        parsed = entry.get('parsed_message', parse_message_content(entry.get('message', ''), entry.get('log.level', 'INFO')))
        
        urc, uid = parsed.get('urc'), parsed.get('uid')

        if parsed.get('is_request') and urc:
            if urc not in nodes:
                parent_urc_from_uid = uid
                if parent_urc_from_uid and parent_urc_from_uid in nodes:
                    parent_node = nodes[parent_urc_from_uid]
                    if parent_node.level < 4:
                        new_node = APICallNode(urc, entry)
                        new_node.level = parent_node.level + 1
                        parent_node.children.append(new_node)
                        nodes[urc] = new_node
        
        elif parsed.get('is_response') and uid in nodes:
            nodes[uid].response = entry
            
        elif parsed.get('error_type'):
            error_associated_to_node = urc or uid
            if error_associated_to_node and error_associated_to_node in nodes:
                nodes[error_associated_to_node].errors.append(entry)

    return nodes

def analyze_error_chain(entries: List[dict], chain_scope_id: str) -> Dict:
    """Analyze a chain of errors and their relationships within a defined scope (e.g., session)."""
    error_chain = []
    for entry in entries:
        if entry.get('log.level', '').upper() in ['ERROR', 'WARN']:
            error_context = {
                'timestamp': entry.get('@timestamp'),
                'level': entry.get('log.level'),
                'source': entry.get('service.name'),
                'message': entry.get('message'),
                'error_type': entry.get('error_type', 'unknown'),
                'urc': entry.get('urc'),
                'uid': entry.get('uid')
            }
            error_chain.append(error_context)
    
    impact_level = 'LOW'
    if any(e['level'] == 'HIGH' for e in error_chain):
        impact_level = 'HIGH'
    if any(e['level'] == 'CRITICAL' for e in error_chain):
        impact_level = 'CRITICAL'
        
    return {
        'session_id': chain_scope_id,
        'errors': error_chain,
        'total_errors': len(error_chain),
        'impact_level': impact_level
    }

def find_error_chains(api_tree: Dict[str, APICallNode]) -> List[Dict]:
    """Extracts and organizes error chains from the API call tree for a single session."""
    all_errors_for_session = []
    session_id = None
    
    if api_tree:
        root_node = next((n for n in api_tree.values() if n.level == 0), None)
        if root_node:
            session_id = root_node.entry.get('session_id')

    if not session_id:
        # Fallback if no clear root node
        first_node = next(iter(api_tree.values()), None)
        if first_node:
            session_id = first_node.entry.get('session_id', 'unknown_session')
        else:
            return []

    for node in api_tree.values():
        all_errors_for_session.extend(node.errors)
        if node.response and node.response.get('log.level', '').upper() in ['ERROR', 'WARN']:
            all_errors_for_session.append(node.response)

    if all_errors_for_session:
        return [analyze_error_chain(all_errors_for_session, session_id)]
    return []

def correlate_logs_for_session(session_logs: Dict[str, List[dict]], session_id: str) -> List[dict]:
    """Correlate logs for a single session to build API tree and find error chains."""
    all_entries = [entry for entries in session_logs.values() for entry in entries]
    
    if not all_entries:
        return []

    api_tree = build_api_call_tree(all_entries)
    if not api_tree:
        return []

    error_chains = find_error_chains(api_tree)
    root_urc_node = next((n for n in api_tree.values() if n.level == 0), None)
    root_urc = root_urc_node.urc if root_urc_node else None

    correlation = {
        'session_id': session_id,
        'root_urc': root_urc,
        'api_calls': [{'urc': n.urc, 'children': [c.urc for c in n.children], 'level': n.level} for n in api_tree.values()],
        'error_chains': error_chains,
        'metrics': {'total_calls': len(api_tree)}
    }
    logger.info(f"Finished log correlation for session {session_id}. Found {len(api_tree)} API calls.")
    return [correlation]

# --- Agent Nodes ---
def ingest_and_process_docs(state: AgentState) -> AgentState:
    session_id = state['session_id']
    logger.info(f"[{session_id}] Ingesting and processing documentation...")
    # The DocumentProcessor is initialized in RAGManager. Here we trigger processing for the session.
    # In the new design, processing happens when get_documentation_context is called.
    # We can use this step to pre-load or check documentation if needed.
    rag_manager.document_processor.process_documents(session_id)
    logger.info(f"[{session_id}] Documentation processing complete.")
    return state

def analyze_root_cause(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    correlations = state["correlations"]
    logger.info(f"[{session_id}] Starting root cause analysis...")
    llm = get_llm()
    rca_agent = create_root_cause_agent(llm)
    
    doc_context = rag_manager.get_relevant_context(session_id, "system architecture, error patterns")
    
    final_rca_results = []
    for correlation in correlations:
        for error_chain in correlation.get('error_chains', []):
            if not error_chain.get('errors'):
                continue
            
            # The first error in the chain is often the primary trigger
            triggering_error = error_chain['errors'][0]
            
            input_content = f"""
            Analyze the following error for root cause.
            Documentation Context: {' '.join(doc_context)}
            Error Chain Context: {json.dumps(error_chain, default=str)}
            EXACT Triggering Error Message: '{triggering_error['message']}'
            """
            
            try:
                raw_analysis = rca_agent.invoke({"messages": [HumanMessage(content=input_content)]})
                
                # Add the initial analysis to a structure that will be passed to the recommendation step
                final_rca_results.append({
                    "triggering_error": triggering_error,
                    "llm_initial_analysis": raw_analysis,
                    "llm_recommendations": {} # Placeholder
                })
            except Exception as e:
                logger.error(f"[{session_id}] Error during LLM call for root cause analysis: {e}")

    logger.info(f"[{session_id}] Completed root cause analysis with {len(final_rca_results)} results.")
    return {**state, "root_causes": final_rca_results}

def generate_recommendations(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    root_causes = state["root_causes"]
    logger.info(f"[{session_id}] Generating recommendations...")
    llm = get_llm()
    rec_agent = create_recommendation_agent(llm)

    doc_context = rag_manager.get_relevant_context(session_id, "remediation steps, configuration guides")

    for rca_item in root_causes:
        analysis = rca_item.get("llm_initial_analysis")
        if not analysis:
            continue
            
        input_content = f"""
        Given the following root cause analysis, provide recommendations.
        Documentation Context: {' '.join(doc_context)}
        Root Cause Analysis: {json.dumps(analysis, default=str)}
        """

        try:
            recommendations = rec_agent.invoke({"messages": [HumanMessage(content=input_content)]})
            rca_item["llm_recommendations"] = recommendations
        except Exception as e:
            logger.error(f"[{session_id}] Error during LLM call for recommendations: {e}")
            
    logger.info(f"[{session_id}] Completed recommendation generation.")
    return state


# --- Graph Definition ---
def get_log_analysis_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("ingest_and_process_docs", ingest_and_process_docs)
    workflow.add_node("analyze_root_cause", analyze_root_cause)
    workflow.add_node("generate_recommendations", generate_recommendations)

    workflow.set_entry_point("ingest_and_process_docs")
    workflow.add_edge("ingest_and_process_docs", "analyze_root_cause")
    workflow.add_edge("analyze_root_cause", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "__end__")

    return workflow.compile(checkpointer=checkpointer)

# --- File Saving ---
def save_analysis_to_json(analysis_results: Dict, output_dir: str = "analysis_output"):
    output_path = get_project_root() / output_dir
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = analysis_results.get("session_id", "unknown_session")

    full_analysis_path = output_path / f"full_analysis_{session_id}_{timestamp}.json"
    with open(full_analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    logger.info(f"Full analysis for session {session_id} saved to {full_analysis_path}")

def save_root_cause_to_json(analysis_results: Dict, output_dir: str = "analysis_output"):
    output_path = get_project_root() / output_dir
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = analysis_results.get("session_id", "unknown_session")
    
    root_causes = analysis_results.get("root_causes", [])
    if not root_causes:
        return

    root_cause_path = output_path / f"root_cause_analysis_{session_id}_{timestamp}.json"
    with open(root_cause_path, 'w') as f:
        json.dump({"session_id": session_id, "timestamp": timestamp, "root_causes": root_causes}, f, indent=2, default=str)
    logger.info(f"Root cause analysis for session {session_id} saved to {root_cause_path}")


# --- Log Fetching ---
def fetch_logs_from_elk(args: argparse.Namespace) -> Dict[str, List[dict]]:
    es_url = f"http://{args.elk_host}:9200"
    logger.info(f"Connecting to Elasticsearch at {es_url}")
    es_auth = (args.elk_user, args.elk_password) if args.elk_user and args.elk_password else None
    
    try:
        es_client = Elasticsearch(
            hosts=[es_url],
            basic_auth=es_auth,
            request_timeout=30
        )
        # The ping() method seems to be the source of the issue.
        # We will rely on the subsequent search operation to validate the connection.
        logger.info(f"Elasticsearch client initialized for {es_url}. Connection will be verified by search operation.")
    except Exception as e:
        logger.error(f"Elasticsearch client initialization failed: {e}")
        return {}
    
    query = {"size": args.elk_max_results, "sort": [{args.elk_time_field: "asc"}], "query": {"match_all": {}}}
    if args.start_time:
        start_time = args.start_time
        end_time = args.end_time or (start_time + timedelta(minutes=2))
        query["query"] = {"range": {args.elk_time_field: {"gte": start_time.isoformat(), "lte": end_time.isoformat()}}}
        
    try:
        response = es_client.search(index=args.elk_index, body=query)
    except Exception as e:
        logger.error(f"Elasticsearch query failed: {e}")
        return {}

    hits = response.get('hits', {}).get('hits', [])
    if not hits:
        return {}

    # Group logs by session_id.
    grouped_logs = defaultdict(lambda: {'file': [], 'elk': []})
    for hit in hits:
        source = hit['_source']
        # Use existing session_id, but only if it exists. Otherwise, skip the log.
        session_id = source.get('session_id')
        if session_id:
            # We also need to parse the message content right away to get URC, UID, etc.
            # for the correlation logic later.
            source['parsed_message'] = parse_message_content(source.get('message', ''), source.get('log.level', 'INFO'))
            grouped_logs[session_id]['elk'].append(source)

    logger.info(f"Fetched {len(hits)} log entries from ELK, grouped into {len(grouped_logs)} sessions with explicit session_ids.")
    return grouped_logs

def filter_logs_by_time(raw_logs: Dict[str, List[dict]], start_time: datetime, end_time: datetime) -> Dict[str, List[dict]]:
    filtered_logs = defaultdict(list)
    for service, logs in raw_logs.items():
        for log in logs:
            log_time = datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S')
            if start_time <= log_time <= end_time:
                filtered_logs[service].append(log)
    return dict(filtered_logs)


# --- Main Application Logic ---
def run_session_analysis(session_id: str, session_logs: Dict[str, List[dict]], app):
    logger.info(f"Starting analysis for session: {session_id}")
    config = {"configurable": {"thread_id": session_id}}
    
    # Correlate logs for this session to build the initial state
    correlations = correlate_logs_for_session(session_logs, session_id)
    if not correlations:
        logger.warning(f"[{session_id}] No correlations found, ending analysis for this session.")
        return

    initial_state = {
        "messages": [],
        "raw_logs": session_logs,
        "log_content": {k: "\n".join(e['message'] for e in v) for k, v in session_logs.items()},
        "correlations": correlations, 
        "root_causes": [], 
        "session_id": session_id
    }
    
    try:
        final_state = app.invoke(initial_state, config)
        logger.info(f"Final state for session {session_id} retrieved.")
        # Save analysis files
        save_analysis_to_json(final_state)
        save_root_cause_to_json(final_state)
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)


def main(args: argparse.Namespace):
    # --- CHECKPOINTING: Use RedisSaver and call setup() to initialize indices ---
    with RedisSaver.from_conn_string(get_redis_url()) as checkpointer:
        checkpointer.setup()
        app = get_log_analysis_graph(checkpointer)

        # Safely check for elk_index attribute before using it
        user_sessions_elk = fetch_logs_from_elk(args) if getattr(args, 'elk_index', None) else {}

        # Safely check for log_files attribute before using it
        user_sessions_files = defaultdict(lambda: defaultdict(list))
        if getattr(args, 'log_files', None):
            raw_logs_files = {}
            for file_path_str in args.log_files:
                service_name = Path(file_path_str).stem.split('.')[0]
                raw_logs_files[service_name] = read_log_file(file_path_str)
            
            # Group file logs by session
            for service, logs in raw_logs_files.items():
                for log in logs:
                    session_id = log['parsed_message'].get('session_id')
                    if session_id:
                        user_sessions_files[session_id][service].append(log)

        # Combine sessions from ELK and files
        # Note: this is a simple merge; more sophisticated merging might be needed if session IDs overlap
        combined_sessions = user_sessions_elk
        for session_id, services in user_sessions_files.items():
            if session_id not in combined_sessions:
                combined_sessions[session_id] = defaultdict(list)
            for service, logs in services.items():
                combined_sessions[session_id][service].extend(logs)
                
        if not combined_sessions:
            logger.error("No logs loaded or no sessions found. Exiting.")
            return

        logger.info(f"Grouped logs into {len(combined_sessions)} sessions.")
        
        threads = []
        for session_id, session_logs in combined_sessions.items():
            # The session_logs from ELK are already in the right format.
            # For file logs, they are now grouped correctly as well.
            thread = threading.Thread(target=run_session_analysis, args=(session_id, session_logs, app))
            threads.append(thread)
            thread.start()
            time.sleep(1) # Stagger thread starts slightly

        for thread in threads:
            thread.join()

        logger.info("All session analyses complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log Analysis AI Agent with Redis and Checkpointing")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--log-files', type=str, nargs='+', help='Paths to log files.')
    source_group.add_argument('--elk-index', type=str, help='Elasticsearch index name.')

    time_group = parser.add_argument_group('Time Filtering')
    time_group.add_argument('--start-time', type=datetime.fromisoformat, help='Start time (YYYY-MM-DDTHH:MM:SS).')
    time_group.add_argument('--end-time', type=datetime.fromisoformat, help='End time (YYYY-MM-DDTHH:MM:SS).')

    elk_group = parser.add_argument_group('ELK Configuration')
    elk_group.add_argument('--elk-host', type=str, default=os.getenv("ELASTICSEARCH_HOST", "elasticsearch"))
    elk_group.add_argument('--elk-user', type=str, default=os.getenv("ELASTICSEARCH_USER"))
    elk_group.add_argument('--elk-password', type=str, default=os.getenv("ELASTICSEARCH_PASSWORD"))
    elk_group.add_argument('--elk-max-results', type=int, default=10000)
    elk_group.add_argument('--elk-time-field', type=str, default='@timestamp')
    elk_group.add_argument('--elk-service-field', type=str, default='service.name')
    elk_group.add_argument('--elk-message-field', type=str, default='message')
    elk_group.add_argument('--elk-level-field', type=str, default='log.level')

    args = parser.parse_args()
    if args.end_time and not args.start_time:
        parser.error("--end-time requires --start-time.")
    
    main(args) 