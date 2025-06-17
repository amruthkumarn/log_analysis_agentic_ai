# LangGraph AI Agents - Log Analysis System

A sophisticated AI-powered log analysis system using LangGraph agents, Redis checkpointing, and large language models to process, correlate, and provide intelligent insights from application logs.

## 🚀 Features

- **AI-Powered Analysis**: Uses Ollama with `llama3.2:1b` model for intelligent root cause analysis
- **Session-Based Correlation**: Groups and correlates logs by session ID with URC/UID hierarchy
- **Redis Checkpointing**: Persistent state management for analysis workflows
- **RAG Integration**: Document-based context retrieval for enhanced analysis
- **Elasticsearch Integration**: Scalable log storage and retrieval
- **Session Filtering**: Analyze all sessions or focus on specific session IDs
- **Structured Output**: JSON-formatted analysis results with timestamps
- **Docker Containerized**: Complete infrastructure with Docker Compose
- **Error Chain Analysis**: Identifies cascading failures and their relationships
- **Severity Classification**: Automatic error severity assessment (LOW/MEDIUM/HIGH/CRITICAL)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Elasticsearch │    │      Redis      │    │     Ollama      │
│   (Log Storage) │    │ (Checkpointing) │    │  (AI Models)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Log Analyzer   │
                    │   (LangGraph)   │
                    └─────────────────┘
```

## 📁 Project Structure

```
langgraph_ai_agents/
├── config/                          # Docker configuration
│   ├── docker-compose.yml          # Multi-service orchestration
│   └── Dockerfile                  # Log analyzer container
├── src/
│   └── log_analyzer/               # Main package
│       ├── core/                   # Core analysis logic
│       │   ├── redis_log_analysis_agent.py  # Main AI agent
│       │   └── document_processor.py        # RAG document processing
│       ├── utils/                  # Utility modules
│       │   ├── redis_client.py     # Redis connectivity
│       │   ├── load_logs_to_elk.py # Elasticsearch loader
│       │   ├── log_producer.py     # Test log generation
│       │   └── cleanup.py          # Maintenance utilities
│       ├── scripts/                # Entry points
│       │   └── run_analysis.py     # Analysis script
│       └── config/                 # Package configuration
│           └── settings.py
├── scripts/                        # Shell scripts
│   ├── load_logs.sh               # Load logs to Elasticsearch
│   ├── analyze_logs.sh            # Run analysis
│   ├── load_and_analyze.sh        # Combined workflow
│   └── manage_analysis.sh         # Output management
├── logs/                          # Log files directory
├── analysis_output/               # Analysis results
├── documentation/                 # Technical documentation
├── tests/                        # Test suite
│   └── data_generators/          # Test data generation
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

## 🔧 Prerequisites

- Docker and Docker Compose
- 8GB+ RAM (for AI models)
- 2GB+ disk space

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd langgraph_ai_agents
```

### 2. Start Infrastructure

```bash
# Start all services (Elasticsearch, Redis, Ollama, Log Analyzer)
docker-compose -f config/docker-compose.yml up -d

# Check service status
docker-compose -f config/docker-compose.yml ps
```

### 3. Load Sample Logs

```bash
# Load test logs into Elasticsearch (creates 'demo-logs' index)
./scripts/load_logs.sh
```

### 4. Run Analysis

```bash
# Analyze all sessions
./scripts/analyze_logs.sh

# OR analyze specific session
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --session-id gbx131
```

## 📊 Usage Examples

### Analyze All Sessions
```bash
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs
```

### Analyze Specific Session
```bash
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --session-id gbx131
```

### Analyze Log Files Directly
```bash
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --log-files /app/logs/api_gateway.log /app/logs/payment_service.log
```

### Time-Based Filtering
```bash
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00
```

## 🧠 AI Analysis Capabilities

### Root Cause Analysis
- **Problem Identification**: Detects core issues from error patterns
- **Confidence Scoring**: AI-assessed confidence levels (0.0-1.0)
- **Error Chain Correlation**: Links related failures across services
- **Severity Assessment**: Automatic classification of error impact

### Intelligent Recommendations
- **Immediate Remediation**: Urgent fixes for critical issues
- **Preventive Measures**: Long-term improvements to prevent recurrence
- **Action Steps**: Specific, actionable implementation guidance
- **Documentation References**: Links to relevant troubleshooting guides

### Session Correlation
- **URC/UID Hierarchy**: 4-level request correlation tracking
- **API Call Trees**: Visual representation of service interactions
- **Cross-Service Analysis**: Identifies failures spanning multiple services
- **Timeline Reconstruction**: Chronological error progression

## 📈 Analysis Output

### File Structure
```
analysis_output/
├── full_analysis_[session_id]_[timestamp].json     # Complete analysis
└── root_cause_analysis_[session_id]_[timestamp].json  # Focused RCA
```

### Sample Root Cause Analysis
```json
{
  "session_id": "gbx131",
  "timestamp": "20250617_195511",
  "root_causes": [
    {
      "triggering_error_message": "Database service unavailable - maintenance mode",
      "overall_chain_impact": "CRITICAL",
      "llm_initial_analysis": {
        "problem_description": "Database maintenance caused cascading failures",
        "probable_root_cause_summary": "Service unavailability during maintenance",
        "confidence_score": 0.95
      },
      "llm_recommendations": {
        "recommendations": [
          {
            "recommendation_type": "Immediate Remediation",
            "recommendation_description": "Implement circuit breaker patterns",
            "action_steps": ["Configure timeout settings", "Add fallback mechanisms"]
          }
        ]
      }
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Elasticsearch
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_USER=
ELASTICSEARCH_PASSWORD=

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434

# Analysis
ANALYSIS_RETENTION_DAYS=30
```

### Service Configuration
- **Elasticsearch**: Port 9200, stores log data
- **Redis**: Port 6379, handles checkpointing
- **Ollama**: Port 11434, provides AI models
- **Log Analyzer**: Custom service, runs analysis

## 🧪 Test Data Generation

Generate realistic test logs for development:

```bash
# Generate test logs with multiple sessions
python tests/data_generators/generate_test_logs.py

# Load generated logs
./scripts/load_logs.sh
```

Test data includes:
- 5 different session IDs
- Multi-service interactions (API Gateway, Payment Service, TIBCO BusinessWorks)
- Realistic error scenarios (timeouts, deadlocks, validation failures)
- URC/UID correlation chains
- Proper timestamp sequencing

## 🐳 Docker Services

### Service Dependencies
```yaml
services:
  elasticsearch:  # Log storage
  redis:         # State management  
  ollama:        # AI models
  log-analyzer:  # Main application
```

### Resource Requirements
- **Elasticsearch**: 1GB RAM minimum
- **Ollama**: 2GB+ RAM (for AI models)
- **Redis**: 256MB RAM
- **Log Analyzer**: 512MB RAM

## 🔍 Troubleshooting

### Common Issues

1. **Container startup failures**
   ```bash
   docker-compose -f config/docker-compose.yml logs [service-name]
   ```

2. **Missing AI models**
   ```bash
   docker-compose -f config/docker-compose.yml exec ollama ollama pull llama3.2:1b
   docker-compose -f config/docker-compose.yml exec ollama ollama pull nomic-embed-text
   ```

3. **Elasticsearch connection issues**
   ```bash
   docker-compose -f config/docker-compose.yml exec log-analyzer curl -X GET "elasticsearch:9200/_cluster/health"
   ```

4. **Redis connectivity**
   ```bash
   docker-compose -f config/docker-compose.yml exec redis redis-cli ping
   ```

### Log Analysis
```bash
# View analysis logs
docker-compose -f config/docker-compose.yml logs log-analyzer

# Check service health
docker-compose -f config/docker-compose.yml ps
```

## 🤝 Development

### Local Development Setup
```bash
# Install development dependencies
pip install -e .
pip install -r requirements.txt

# Run linting
ruff check src/
ruff format src/

# Run tests
python -m pytest tests/
```

### Adding New Features
1. Implement core logic in `src/log_analyzer/core/`
2. Add utilities in `src/log_analyzer/utils/`
3. Update configuration in `src/log_analyzer/config/`
4. Add tests in `tests/`
5. Update documentation

## 📋 Command Reference

### Analysis Commands
```bash
# Basic analysis
./scripts/analyze_logs.sh

# Load and analyze in one step
./scripts/load_and_analyze.sh

# Custom analysis
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis [options]
```

### Management Commands
```bash
# List analysis outputs
./scripts/manage_analysis.sh list

# Clean old outputs
./scripts/manage_analysis.sh clean [days]

# Service management
docker-compose -f config/docker-compose.yml [up|down|restart|logs]
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangGraph**: Agent orchestration framework
- **Ollama**: Local AI model serving
- **Redis**: In-memory data structure store
- **Elasticsearch**: Search and analytics engine

