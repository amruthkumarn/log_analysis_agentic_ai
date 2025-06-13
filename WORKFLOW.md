# Log Analysis Workflow

This document explains the separated workflow for loading logs and running AI-powered analysis.

## Overview

The log analysis system now has **two separate steps**:

1. **Load logs** into Elasticsearch (`load_logs.sh`)
2. **Run AI analysis** on the loaded logs (`analyze_logs.sh`)

This separation allows you to:
- Load logs once and run multiple analyses
- Analyze different time ranges without reloading
- Use different data sources (ELK vs files) flexibly

## Quick Start

### Option 1: Two-Step Process (Recommended)

```bash
# Step 1: Load logs into Elasticsearch
./load_logs.sh

# Step 2: Run AI analysis
./analyze_logs.sh --elk-index test-logs-default
```

### Option 2: Combined Process

```bash
# Run both steps together
./load_and_analyze.sh
```

## Detailed Usage

### 1. Loading Logs (`load_logs.sh`)

Load log files into Elasticsearch for analysis:

```bash
# Basic usage (uses default files and index)
./load_logs.sh

# Custom index and files
./load_logs.sh --elk-index my-logs --log-files "logs/app1.log logs/app2.log"

# Help
./load_logs.sh --help
```

**Options:**
- `--elk-index INDEX`: Elasticsearch index name (default: `test-logs-default`)
- `--log-files FILES`: Space-separated log file paths
- `--elk-host HOST`: Elasticsearch host (default: `elasticsearch`)

### 2. Running Analysis (`analyze_logs.sh`)

Run AI-powered analysis on logs:

```bash
# Analyze from Elasticsearch
./analyze_logs.sh --elk-index test-logs-default

# Analyze from local files directly
./analyze_logs.sh --log-files "logs/app1.log logs/app2.log"

# Analyze specific time range
./analyze_logs.sh --elk-index my-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00

# Help
./analyze_logs.sh --help
```

**Data Sources:**
- `--elk-index INDEX`: Analyze from Elasticsearch index
- `--log-files FILES`: Analyze from local files directly

**Options:**
- `--start-time TIME`: Analysis start time (ISO format)
- `--end-time TIME`: Analysis end time (ISO format)
- `--elk-host HOST`: Elasticsearch host

## Key Improvements

### ✅ **Fixed UID/URC Handling**

The system now preserves original message content:
- **Before**: Added `null` values for missing UID/URC fields
- **After**: Only includes UID/URC when actually present in source logs

### ✅ **Separated Execution**

- **Load once, analyze many**: Load logs into ELK once, run multiple analyses
- **Flexible data sources**: Analyze from ELK or files directly
- **Time range filtering**: Analyze specific time periods without reloading

### ✅ **Better Error Handling**

- Health checks for Elasticsearch and AI services
- Clear error messages and validation
- Graceful handling of missing services

## Output Files

Analysis results are saved in `analysis_output/`:

- `full_analysis_<session_id>_<timestamp>.json` - Complete analysis
- `root_cause_analysis_<session_id>_<timestamp>.json` - Root cause summary (if issues found)

## Examples

### Example 1: Standard Workflow

```bash
# 1. Start services
docker-compose up -d

# 2. Load logs
./load_logs.sh

# 3. Run analysis
./analyze_logs.sh --elk-index test-logs-default

# 4. Check results
ls -ltr analysis_output/
```

### Example 2: Custom Time Range

```bash
# Load logs (once)
./load_logs.sh --elk-index production-logs

# Analyze morning issues
./analyze_logs.sh --elk-index production-logs \
  --start-time 2024-03-20T08:00:00 \
  --end-time 2024-03-20T12:00:00

# Analyze afternoon issues  
./analyze_logs.sh --elk-index production-logs \
  --start-time 2024-03-20T13:00:00 \
  --end-time 2024-03-20T17:00:00
```

### Example 3: Direct File Analysis

```bash
# Analyze files directly (no ELK loading needed)
./analyze_logs.sh --log-files "logs/critical_error.log logs/system.log"
```

## Troubleshooting

### Services Not Running
```bash
# Check service status
docker-compose ps

# Start services
docker-compose up -d
```

### Elasticsearch Issues
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# View logs
docker-compose logs elasticsearch
```

### AI Service Issues
```bash
# Check Ollama models
docker-compose exec ollama ollama list

# View analysis logs
docker-compose logs log-analyzer
``` 