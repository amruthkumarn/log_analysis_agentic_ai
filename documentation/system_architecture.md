# System Architecture Documentation

## Production System Dependencies
- 3scale API Gateway depends on TIBCO BusinessWorks for backend processing
- TIBCO BusinessWorks connects to multiple databases and external services
- Services communicate via REST APIs and JMS queues

## AI-Powered Log Analysis System

### Architecture Overview
The log analysis system consists of four main components running in Docker containers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Elasticsearch │    │      Redis      │    │     Ollama      │
│   (Log Storage) │    │ (Checkpointing) │    │  (AI Models)    │
│   Port: 9200    │    │   Port: 6379    │    │  Port: 11434    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Log Analyzer   │
                    │ (Main Analysis) │
                    │   AI Agent      │
                    └─────────────────┘
```

### Component Responsibilities

#### Log Analyzer (Main Service)
- **Core Analysis Engine**: Performs correlation analysis and pattern detection
- **AI Integration**: Communicates with Ollama for root cause analysis and recommendations
- **Session Management**: Handles URC/UID hierarchy tracking across services
- **Output Generation**: Creates structured JSON reports with configurable output directories

#### Elasticsearch
- **Log Storage**: Stores ingested log data in searchable indices
- **Query Engine**: Provides time-based and session-based log filtering
- **Data Retention**: Manages log lifecycle and cleanup

#### Redis
- **Checkpointing**: Maintains analysis state for resumable operations
- **Session Tracking**: Caches session correlation data
- **Performance Optimization**: Reduces redundant processing

#### Ollama
- **AI Models**: Hosts llama3.2:1b for natural language analysis
- **Embeddings**: Provides nomic-embed-text for document similarity
- **Local Processing**: Ensures data privacy with on-premises AI

### Data Flow
1. **Log Ingestion**: Raw logs → Elasticsearch (demo-logs index)
2. **Analysis Trigger**: User initiates analysis via shell script or Python
3. **Data Retrieval**: Log Analyzer queries Elasticsearch for session data
4. **Correlation Analysis**: System builds URC/UID relationship trees
5. **AI Analysis**: Ollama processes error patterns for root cause analysis
6. **Output Generation**: Results saved to configurable output directories
7. **Checkpointing**: Redis stores progress for fault tolerance

### Network Configuration
- **Internal Networks**: Services communicate via Docker networks (redis, elk)
- **Volume Mounts**: Analysis output directories mounted to host filesystem
- **Port Exposure**: Only necessary ports exposed for external access

## Normal Behavior Patterns
- API Gateway response times: < 200ms
- Database connection pool utilization: 20-80%
- JMS queue size: < 1000 messages
- Error rate threshold: < 1%

## Analysis System Performance Metrics
- **Processing Speed**: ~100-500 log entries per second
- **AI Analysis Time**: 2-5 seconds per error chain
- **Memory Usage**: 2-4GB total across all containers
- **Storage Requirements**: ~10MB per 1000 log entries

## Known Issues
1. Database Connection Issues
   - Symptoms: Connection timeouts, pool exhaustion
   - Impact: Service degradation, increased latency
   - Resolution: Increase pool size, implement connection retry

2. Rate Limiting
   - Symptoms: 429 errors, increased latency
   - Impact: Service unavailability
   - Resolution: Adjust rate limits, implement circuit breaker

3. JMS Queue Issues
   - Symptoms: Queue size warnings, message processing delays
   - Impact: Message loss, processing delays
   - Resolution: Scale consumers, optimize message processing

## Analysis System Troubleshooting
1. **Container Startup Issues**
   - Check Docker network configuration
   - Verify volume mount permissions
   - Ensure required AI models are pulled

2. **Analysis Performance Issues**
   - Monitor Elasticsearch query performance
   - Check Redis memory usage
   - Verify Ollama model availability

3. **Output Directory Issues**
   - Confirm Docker volume mounts are configured
   - Check filesystem permissions
   - Verify output directory parameter usage 