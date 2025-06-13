# Log Analysis AI Agent

## Overview

This project implements a sophisticated multi-agent system designed for comprehensive analysis of application and system logs. It leverages the power of Large Language Models (LLMs) through LangGraph and LangChain, integrated with Ollama for local model execution (e.g., `llama3.2:1b`). The primary goal is to automate the process of parsing diverse log formats, correlating events across multiple log sources, detecting anomalies, performing in-depth root cause analysis (RCA) for errors, and generating actionable recommendations for remediation and prevention.

The system is built to handle complex error scenarios, including those involving chained URC/UIDs (Unique Request/Identifier Chains) for tracking transactions across microservices. It uses a Retrieval Augmented Generation (RAG) mechanism to enrich LLM analysis with contextual information from provided documentation, leading to more accurate and relevant insights. **The agent can ingest logs from both local files and a centralized Elasticsearch (ELK) stack.**

## Key Features

### 🔥 **Enhanced Critical Error Detection & Impact Assessment**
*   **Intelligent Severity Calculation**: Automatically classifies errors as CRITICAL, HIGH, MEDIUM, or LOW based on error content analysis
*   **Critical Error Indicators**: Detects system-wide failures like "database cluster unavailable", "data corruption detected", "emergency fallback failed"
*   **Comprehensive Impact Analysis**: Provides detailed severity breakdown with counts of critical, high, medium, and low severity errors per session
*   **Real-time Severity Context**: LLM receives calculated severity levels to guide analysis criticality

### 🚀 **Advanced Multi-Agent Architecture**
*   **Redis-Based State Management**: Uses Redis for robust checkpointing and session isolation
*   **LangGraph Workflow**: Orchestrates specialized agents for different stages of log analysis
*   **Fine-Tuned Prompts**: Extensively optimized prompts for accurate root cause analysis and recommendations
*   **Enhanced JSON Parsing**: Robust handling of LLM outputs with markdown code block processing

### 📊 **Comprehensive Data Processing**
*   **Flexible Data Ingestion**: Supports log analysis from both local log files and direct querying of an **Elasticsearch** index
*   **Session-Based Correlation**: Groups log entries by session ID and correlates events within each session by analyzing URC/UID chains
*   **API Call Tree Construction**: Builds hierarchical trees of API calls based on URC/UID relationships up to 4 levels deep
*   **Error Chain Analysis**: Identifies and analyzes chains of errors within sessions with proper impact assessment

### 🤖 **AI-Powered Analysis**
*   **LLM-Powered Root Cause Analysis (RCA)**: Detailed analysis of critical error events with confidence scoring
*   **Actionable LLM-Generated Recommendations**: Specific, actionable recommendations with implementation steps
*   **Retrieval Augmented Generation (RAG)**: Integrates with project documentation for contextual analysis
*   **Smart Error Context**: Provides LLMs with calculated severity and detailed error context for accurate analysis

### 🔧 **Production-Ready Features**
*   **Docker Compose Setup**: Complete containerized environment with ELK stack, Redis, and Ollama integration
*   **Robust Error Handling**: Comprehensive error handling with graceful degradation
*   **Structured JSON Output**: Well-structured analysis results with detailed metadata
*   **Command-Line Interface**: Flexible CLI with extensive configuration options
*   **Documentation Integration**: Supports .txt, .md, .pdf files for RAG context

## Critical Error Detection Capabilities

The system now includes advanced error severity detection that automatically identifies:

### 🔴 **CRITICAL Level Errors**
- Database cluster unavailable / all nodes down
- Data corruption detected
- Emergency fallback failed / backup systems offline
- Critical database errors
- System-wide failures and infrastructure outages

### 🟠 **HIGH Level Errors**
- Connection pool exhausted
- Database deadlocks
- Gateway timeouts / upstream service unavailable
- Authentication failures
- 5xx status codes (500, 503, 504)
- Resource locks and maintenance mode issues

### 🟡 **MEDIUM Level Errors**
- Validation failures (422 status codes)
- Missing required fields
- Bad requests (400 status codes)
- Internal server errors

### ⚪ **LOW Level Errors**
- General application errors
- Warning-level issues
- Minor operational problems

## Enhanced Analysis Output

The system now provides:

### **Impact Assessment**
```json
{
  "overall_chain_impact": "CRITICAL",
  "severity_breakdown": {
    "critical": 4,
    "high": 3,
    "medium": 0,
    "low": 0
  }
}
```

### **Detailed Root Cause Analysis**
```json
{
  "problem_description": "Critical database cluster failure affecting all payment processing operations",
  "probable_root_cause_summary": "Database cluster became unavailable due to all nodes going down simultaneously, causing cascading failures across payment services",
  "confidence_score": 0.95,
  "calculated_severity": "CRITICAL"
}
```

### **Actionable Recommendations**
```json
{
  "recommendations": [
    {
      "recommendation_type": "Immediate Remediation",
      "recommendation_description": "Implement database cluster health monitoring with automatic failover",
      "action_steps": [
        "Configure cluster health checks with 30-second intervals",
        "Set up automatic failover to backup cluster",
        "Implement circuit breaker patterns for database connections"
      ]
    }
  ]
}
```

## Prerequisites

*   Python >=3.11
*   **Docker and Docker Compose**: Required for running the complete stack
*   Ollama installed and running locally
*   The `llama3.2:1b` model (or compatible) pulled via Ollama: `ollama pull llama3.2:1b`
*   `uv` (Python packaging tool): Install with `pip install uv` if not present

## Project Structure

```
.gitignore
README.md
docker-compose.yml          # Complete stack: ELK, Redis, Ollama, log-analyzer
analysis_output/             # Timestamped JSON analysis results
documentation/               # RAG documentation (.txt, .md, .pdf)
    red_hat_3scale_guide.pdf        # API management documentation
    unravel_spark_guide.pdf         # Spark troubleshooting guide
    system_architecture.md          # System architecture documentation
    troubleshooting_guide.md        # General troubleshooting guide
logs/                       # Input log files and operational logs
    demo_logs.json                  # Comprehensive demo dataset with critical errors
    3scale_api_gateway.log          # Example API gateway logs
    payment_service.log             # Example payment service logs
    tibco_businessworks.log         # Example integration logs
src/
    __init__.py
    redis_log_analysis_agent.py     # Main analysis engine with Redis integration
    document_processor.py           # RAG document processing
    redis_client.py                 # Redis connection management
    load_logs_to_elk.py            # Log ingestion utility
pyproject.toml              # Project dependencies and configuration
```

## Quick Start

### 1. Start the Complete Stack
```bash
docker-compose up --build -d
```
This starts:
- **Elasticsearch** (port 9200)
- **Kibana** (port 5601) 
- **Redis** (port 6379)
- **Ollama** (port 11434)
- **Log Analyzer** (ready for commands)

### 2. Load Demo Data with Critical Errors
```bash
docker-compose exec log-analyzer python src/load_logs_to_elk.py --demo-data
```
This loads comprehensive demo data including:
- **Session rlt509**: Critical system failures (database cluster down, data corruption, emergency fallback failures)
- **Session gbx131**: Database issues (deadlocks, constraint violations, timeouts)
- **Session nwj331**: Authentication failures (token expired, insufficient permissions)
- **Session iym306**: Validation errors (missing fields, format issues)
- **Session vcw023**: Successful operations with performance warnings

### 3. Run Critical Error Analysis
```bash
docker-compose exec log-analyzer python -m src.redis_log_analysis_agent --elk-index demo-logs
```

### 4. View Results
Check the `analysis_output/` directory for:
- `full_analysis_<session_id>_<timestamp>.json` - Complete analysis
- `root_cause_analysis_<session_id>_<timestamp>.json` - Focused RCA with severity assessment

## Advanced Usage

### Custom Log Analysis
```bash
# Analyze specific time range
docker-compose exec log-analyzer python -m src.redis_log_analysis_agent \
  --elk-index your-index \
  --start-time "2024-01-01T00:00:00Z" \
  --end-time "2024-01-02T00:00:00Z"

# Load custom logs
docker-compose exec log-analyzer python src/load_logs_to_elk.py \
  --log-files logs/your_app.log \
  --index-name your-index
```

### RAG Documentation Enhancement
Place your documentation in the `documentation/` directory:
```bash
# The system automatically processes:
documentation/
├── api_guides.pdf           # API documentation
├── troubleshooting.md       # Troubleshooting guides
├── architecture.txt         # System architecture
└── runbooks/               # Operational runbooks
```

## Analysis Output Structure

### Full Analysis File
```json
{
  "session_id": "rlt509",
  "timestamp": "20250612_091500",
  "correlations": [
    {
      "session_id": "rlt509",
      "root_urc": "urc-root-rlt509",
      "api_calls": [...],
      "error_chains": [
        {
          "impact_level": "CRITICAL",
          "severity_breakdown": {
            "critical": 4,
            "high": 3,
            "medium": 0,
            "low": 0
          }
        }
      ]
    }
  ]
}
```

### Root Cause Analysis File
```json
{
  "session_id": "rlt509",
  "root_causes": [
    {
      "overall_chain_impact": "CRITICAL",
      "triggering_error_message": "Database cluster unavailable - all nodes down",
      "llm_initial_analysis": {
        "problem_description": "Critical database infrastructure failure",
        "probable_root_cause_summary": "Complete database cluster failure due to...",
        "confidence_score": 0.95,
        "calculated_severity": "CRITICAL"
      },
      "llm_recommendations": {
        "recommendations": [
          {
            "recommendation_type": "Immediate Remediation",
            "recommendation_description": "Implement database cluster monitoring",
            "action_steps": ["Configure health checks", "Set up failover"]
          }
        ]
      }
    }
  ]
}
```

## Architecture

### Multi-Agent Workflow
1. **Document Ingestion Agent**: Processes RAG documentation
2. **Root Cause Analysis Agent**: Performs detailed error analysis with severity context
3. **Recommendation Agent**: Generates actionable remediation steps

### Technology Stack
- **LangGraph**: Multi-agent orchestration
- **Ollama**: Local LLM execution
- **Redis**: State management and vector storage
- **Elasticsearch**: Log storage and querying
- **Docker**: Containerized deployment

### Error Severity Pipeline
1. **Content Analysis**: Scans error messages for severity indicators
2. **Classification**: Assigns CRITICAL/HIGH/MEDIUM/LOW levels
3. **Context Enhancement**: Provides severity context to LLM
4. **Impact Assessment**: Calculates overall session impact

## Monitoring and Observability

### Kibana Dashboard
Access Kibana at `http://localhost:5601` to:
- Visualize log patterns
- Monitor error trends
- Create custom dashboards
- Set up alerting rules

### Redis Insights
Monitor Redis state and vector storage:
- Session checkpoints
- Document embeddings
- Analysis progress

## Contributing

We welcome contributions! Areas for enhancement:
- Additional error pattern detection
- Custom severity rules
- Enhanced visualization
- Integration with monitoring tools

## License

[Your License Here]

