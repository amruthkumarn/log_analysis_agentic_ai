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
# Analyze all sessions (default output directory)
./scripts/analyze_logs.sh --elk-index demo-logs

# Analyze specific session with custom output directory
./scripts/analyze_logs.sh --elk-index demo-logs --session-id gbx131 --output-dir my_results

# OR use Python script directly
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --session-id gbx131 --output-dir custom_results
```

## 📊 Usage Examples

### Analyze All Sessions
```bash
# Default output directory (analysis_output/)
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs

# Custom output directory
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --output-dir my_custom_results
```

### Analyze Specific Session
```bash
# Single session analysis with custom output
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --session-id gbx131 --output-dir session_analysis

# Using shell script (recommended)
./scripts/analyze_logs.sh --elk-index demo-logs --session-id gbx131 --output-dir results
```

### Analyze Log Files Directly
```bash
# Direct file analysis with custom output
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --log-files /app/logs/api_gateway.log /app/logs/payment_service.log \
  --output-dir file_analysis_results
```

### Time-Based Filtering
```bash
# Time-filtered analysis with custom output
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00 \
  --output-dir time_filtered_results
```

### Advanced Shell Script Usage
```bash
# Shell script with all options
./scripts/analyze_logs.sh \
  --elk-index demo-logs \
  --session-id gbx131 \
  --output-dir production_analysis \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00

# Help for all available options
./scripts/analyze_logs.sh --help
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

### Flexible Output Directory Configuration
The system supports flexible output directory configuration for organizing analysis results:

```bash
# Default behavior - saves to analysis_output/
./scripts/analyze_logs.sh --elk-index demo-logs

# Custom relative directory - saves to custom_results/
./scripts/analyze_logs.sh --elk-index demo-logs --output-dir custom_results

# Multiple custom directories for different analyses
./scripts/analyze_logs.sh --elk-index demo-logs --session-id session1 --output-dir session1_analysis
./scripts/analyze_logs.sh --elk-index demo-logs --session-id session2 --output-dir session2_analysis
```

### File Structure
```
langgraph_ai_agents/
├── analysis_output/                    # Default output directory
│   ├── full_analysis_[session_id]_[timestamp].json
│   └── root_cause_analysis_[session_id]_[timestamp].json
├── custom_results/                     # Custom output directory
│   ├── full_analysis_[session_id]_[timestamp].json
│   └── root_cause_analysis_[session_id]_[timestamp].json
└── production_analysis/                # Another custom directory
    ├── full_analysis_[session_id]_[timestamp].json
    └── root_cause_analysis_[session_id]_[timestamp].json
```

### Output Files
- **Full Analysis**: `full_analysis_[session_id]_[timestamp].json`
  - Complete analysis including correlations, raw logs, and AI insights
  - Detailed session information and API call trees
  - Comprehensive error chain analysis

- **Root Cause Analysis**: `root_cause_analysis_[session_id]_[timestamp].json`
  - Focused analysis of identified issues
  - AI-generated problem descriptions and root causes
  - Actionable recommendations for remediation

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

### Command Line Parameters

#### Core Analysis Parameters
```bash
# Data source (required - choose one)
--elk-index INDEX              # Analyze from Elasticsearch index
--log-files FILE1 FILE2...     # Analyze from local log files

# Session and time filtering
--session-id SESSION_ID        # Analyze specific session only
--start-time YYYY-MM-DDTHH:MM:SS  # Filter by start time
--end-time YYYY-MM-DDTHH:MM:SS    # Filter by end time

# Output configuration
--output-dir DIRECTORY         # Custom output directory (default: analysis_output)

# Elasticsearch configuration
--elk-host HOST               # Elasticsearch host (default: elasticsearch)
--elk-user USERNAME           # Elasticsearch username
--elk-password PASSWORD       # Elasticsearch password
--elk-max-results NUMBER      # Maximum results to fetch (default: 10000)
```

#### Shell Script Parameters
```bash
./scripts/analyze_logs.sh [OPTIONS]

Options:
  --elk-index INDEX         Analyze logs from Elasticsearch index
  --log-files FILES          Analyze logs from local files (space-separated)
  --session-id ID           Analyze only specific session ID
  --output-dir DIR          Output directory for results (default: analysis_output)
  --start-time TIME         Analysis start time
  --end-time TIME           Analysis end time
  --elk-host HOST           Elasticsearch host (default: elasticsearch)
  --help                    Show help message

Examples:
  # Basic analysis
  ./scripts/analyze_logs.sh --elk-index demo-logs
  
  # Session-specific analysis with custom output
  ./scripts/analyze_logs.sh --elk-index demo-logs --session-id gbx131 --output-dir session_results
  
  # Time-filtered analysis
  ./scripts/analyze_logs.sh --elk-index demo-logs --start-time 2024-03-20T10:00:00 --end-time 2024-03-20T11:00:00
```

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
DOCKER_CONTAINER=true          # Enables proper path resolution in containers
```

### Docker Volume Configuration
The system automatically mounts common output directories to the host filesystem:

```yaml
volumes:
  - ../analysis_output:/app/analysis_output           # Default directory
  - ../custom_analysis_output:/app/custom_analysis_output
  - ../my_custom_results:/app/my_custom_results
```

**Note**: For new custom directory names, add them to the docker-compose.yml volume mounts to ensure files appear on the host filesystem.

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

### Quick Start Commands
```bash
# Start all services
docker-compose -f config/docker-compose.yml up -d

# Load sample data
./scripts/load_logs.sh

# Run basic analysis
./scripts/analyze_logs.sh --elk-index demo-logs

# Check results
ls -ltr analysis_output/
```

### Analysis Commands

#### Shell Script (Recommended)
```bash
# Basic analysis with default output
./scripts/analyze_logs.sh --elk-index demo-logs

# Session-specific analysis with custom output directory
./scripts/analyze_logs.sh --elk-index demo-logs --session-id gbx131 --output-dir session_results

# Time-filtered analysis
./scripts/analyze_logs.sh --elk-index demo-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00 \
  --output-dir time_analysis

# File-based analysis
./scripts/analyze_logs.sh --log-files "logs/app1.log logs/app2.log" --output-dir file_results

# Show all available options
./scripts/analyze_logs.sh --help
```

#### Direct Python Script
```bash
# All sessions with custom output
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --output-dir comprehensive_analysis

# Single session analysis
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs --session-id gbx131 --output-dir single_session

# File-based analysis
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --log-files /app/logs/api_gateway.log /app/logs/payment_service.log \
  --output-dir direct_file_analysis

# Time-filtered with custom output
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.scripts.run_analysis \
  --elk-index demo-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00 \
  --output-dir time_filtered
```

### Output Management Commands
```bash
# List analysis outputs in default directory
ls -ltr analysis_output/

# List outputs in custom directory
ls -ltr my_custom_results/

# View latest analysis results
find analysis_output/ -name "*.json" -type f -exec ls -lt {} + | head -10

# Check analysis file sizes
du -sh analysis_output/* | sort -hr

# Search for specific session results
find . -name "*gbx131*" -type f

# Clean old analysis results (if using manage_analysis.sh)
./scripts/manage_analysis.sh clean 7  # Remove files older than 7 days
```

### Data Loading Commands
```bash
# Load test logs to Elasticsearch
./scripts/load_logs.sh

# Load and analyze in one step
./scripts/load_and_analyze.sh

# Load with custom index name
docker-compose -f config/docker-compose.yml exec log-analyzer \
  python -m src.log_analyzer.utils.load_logs_to_elk --index-name custom-logs
```

### Service Management Commands
```bash
# Start all services
docker-compose -f config/docker-compose.yml up -d

# Stop all services
docker-compose -f config/docker-compose.yml down

# Restart specific service
docker-compose -f config/docker-compose.yml restart log-analyzer

# View service logs
docker-compose -f config/docker-compose.yml logs log-analyzer
docker-compose -f config/docker-compose.yml logs ollama

# Check service status
docker-compose -f config/docker-compose.yml ps

# Pull latest AI models
docker-compose -f config/docker-compose.yml exec ollama ollama pull llama3.2:1b
docker-compose -f config/docker-compose.yml exec ollama ollama pull nomic-embed-text
```

### Development Commands
```bash
# Build log-analyzer container
docker-compose -f config/docker-compose.yml build log-analyzer

# Access container shell
docker-compose -f config/docker-compose.yml exec log-analyzer /bin/bash

# Run tests
docker-compose -f config/docker-compose.yml exec log-analyzer python -m pytest tests/

# Generate test data
python tests/data_generators/generate_test_logs.py

# Check container resources
docker stats
```

### Troubleshooting Commands
```bash
# Check Elasticsearch health
docker-compose -f config/docker-compose.yml exec log-analyzer \
  curl -X GET "elasticsearch:9200/_cluster/health"

# Test Redis connectivity
docker-compose -f config/docker-compose.yml exec redis redis-cli ping

# Check Ollama models
docker-compose -f config/docker-compose.yml exec ollama ollama list

# View detailed logs
docker-compose -f config/docker-compose.yml logs --tail=100 log-analyzer

# Check disk usage
docker system df
docker volume ls
```

### Cleanup Commands
```bash
# Remove old analysis outputs
find analysis_output/ -name "*.json" -mtime +7 -delete

# Clean Docker resources
docker system prune -f

# Remove unused volumes
docker volume prune -f

# Full cleanup (WARNING: removes all data)
docker-compose -f config/docker-compose.yml down -v
docker system prune -a -f
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangGraph**: Agent orchestration framework
- **Ollama**: Local AI model serving
- **Redis**: In-memory data structure store
- **Elasticsearch**: Search and analytics engine

