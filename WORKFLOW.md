# Log Analysis Workflow

This document explains the separated workflow for loading logs and running AI-powered analysis with flexible output directory configuration.

## Overview

The log analysis system now has **two separate steps**:

1. **Load logs** into Elasticsearch (`load_logs.sh`)
2. **Run AI analysis** on the loaded logs (`analyze_logs.sh`)

This separation allows you to:
- Load logs once and run multiple analyses
- Analyze different time ranges without reloading
- Use different data sources (ELK vs files) flexibly
- **Configure custom output directories** for organized results

## Quick Start

### Option 1: Two-Step Process (Recommended)

```bash
# Step 1: Load logs into Elasticsearch
./scripts/load_logs.sh

# Step 2: Run AI analysis with default output
./scripts/analyze_logs.sh --elk-index demo-logs

# Step 3: Run analysis with custom output directory
./scripts/analyze_logs.sh --elk-index demo-logs --output-dir my_analysis_results
```

### Option 2: Combined Process

```bash
# Run both steps together (when available)
./scripts/load_and_analyze.sh
```

## Detailed Usage

### 1. Loading Logs (`load_logs.sh`)

Load log files into Elasticsearch for analysis:

```bash
# Basic usage (uses default files and index)
./scripts/load_logs.sh

# Custom index and files
./scripts/load_logs.sh --elk-index my-logs --log-files "logs/app1.log logs/app2.log"

# Help
./scripts/load_logs.sh --help
```

**Options:**
- `--elk-index INDEX`: Elasticsearch index name (default: `demo-logs`)
- `--log-files FILES`: Space-separated log file paths
- `--elk-host HOST`: Elasticsearch host (default: `elasticsearch`)

### 2. Running Analysis (`analyze_logs.sh`)

Run AI-powered analysis on logs with flexible output configuration:

```bash
# Analyze from Elasticsearch (default output directory)
./scripts/analyze_logs.sh --elk-index demo-logs

# Analyze with custom output directory
./scripts/analyze_logs.sh --elk-index demo-logs --output-dir production_analysis

# Analyze specific session with custom output
./scripts/analyze_logs.sh --elk-index demo-logs --session-id gbx131 --output-dir session_results

# Analyze from local files directly with custom output
./scripts/analyze_logs.sh --log-files "logs/app1.log logs/app2.log" --output-dir file_analysis

# Analyze specific time range with organized output
./scripts/analyze_logs.sh --elk-index my-logs \
  --start-time 2024-03-20T10:00:00 \
  --end-time 2024-03-20T11:00:00 \
  --output-dir morning_analysis

# Help
./scripts/analyze_logs.sh --help
```

**Data Sources:**
- `--elk-index INDEX`: Analyze from Elasticsearch index
- `--log-files FILES`: Analyze from local files directly

**Filtering Options:**
- `--session-id SESSION`: Analyze specific session only
- `--start-time TIME`: Analysis start time (ISO format)
- `--end-time TIME`: Analysis end time (ISO format)

**Output Configuration:**
- `--output-dir DIRECTORY`: Custom output directory (default: `analysis_output`)

**Connection Options:**
- `--elk-host HOST`: Elasticsearch host (default: `elasticsearch`)

## Key Improvements

### ✅ **Flexible Output Directory Configuration**

The system now supports customizable output directories:
- **Default behavior**: Results saved to `analysis_output/`
- **Custom directories**: Specify any directory name with `--output-dir`
- **Organized analysis**: Separate results by session, time period, or analysis type
- **Host filesystem**: All results appear directly on host filesystem via Docker volume mounts

### ✅ **Session-Specific Analysis**

- **Single session focus**: Use `--session-id` to analyze specific sessions
- **Faster processing**: Skip irrelevant sessions for targeted analysis
- **Organized output**: Session-specific results in dedicated directories

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

## Output Files and Organization

### Output Directory Structure

```bash
langgraph_ai_agents/
├── analysis_output/                    # Default directory
│   ├── full_analysis_all_sessions_20250617_143022.json
│   └── root_cause_analysis_all_sessions_20250617_143022.json
├── production_analysis/                # Custom directory example
│   ├── full_analysis_gbx131_20250617_143155.json
│   └── root_cause_analysis_gbx131_20250617_143155.json
├── session_results/                    # Session-specific analysis
│   ├── full_analysis_gbx131_20250617_143301.json
│   └── root_cause_analysis_gbx131_20250617_143301.json
└── morning_analysis/                   # Time-filtered analysis
    ├── full_analysis_filtered_20250617_143445.json
    └── root_cause_analysis_filtered_20250617_143445.json
```

### File Types

- **Full Analysis**: `full_analysis_<session_id>_<timestamp>.json`
  - Complete analysis including correlations, raw logs, and AI insights
  - Detailed session information and API call trees
  - Comprehensive error chain analysis

- **Root Cause Analysis**: `root_cause_analysis_<session_id>_<timestamp>.json`
  - Focused analysis of identified issues (only created when issues are found)
  - AI-generated problem descriptions and root causes
  - Actionable recommendations for remediation

### Managing Output Directories

```bash
# List analysis results in different directories
ls -ltr analysis_output/
ls -ltr production_analysis/
ls -ltr session_results/

# Find specific session results across all directories
find . -name "*gbx131*" -type f

# Check sizes of different analysis directories
du -sh analysis_output/ production_analysis/ session_results/

# Clean old results from specific directory
find production_analysis/ -name "*.json" -mtime +7 -delete
```

## Examples

### Example 1: Standard Workflow

```bash
# 1. Start services
docker-compose -f config/docker-compose.yml up -d

# 2. Load logs
./scripts/load_logs.sh

# 3. Run analysis with default output
./scripts/analyze_logs.sh --elk-index demo-logs

# 4. Check results
ls -ltr analysis_output/
```

### Example 2: Organized Analysis by Session

```bash
# Load logs (once)
./scripts/load_logs.sh --elk-index production-logs

# Analyze each session separately with organized output
./scripts/analyze_logs.sh --elk-index production-logs --session-id gbx131 --output-dir session_gbx131
./scripts/analyze_logs.sh --elk-index production-logs --session-id abc123 --output-dir session_abc123
./scripts/analyze_logs.sh --elk-index production-logs --session-id xyz789 --output-dir session_xyz789

# Check results for each session
ls -ltr session_gbx131/
ls -ltr session_abc123/
ls -ltr session_xyz789/
```

### Example 3: Time-Based Analysis Organization

```bash
# Load logs (once)
./scripts/load_logs.sh --elk-index production-logs

# Analyze different time periods with organized output
./scripts/analyze_logs.sh --elk-index production-logs \
  --start-time 2024-03-20T08:00:00 \
  --end-time 2024-03-20T12:00:00 \
  --output-dir morning_issues

./scripts/analyze_logs.sh --elk-index production-logs \
  --start-time 2024-03-20T13:00:00 \
  --end-time 2024-03-20T17:00:00 \
  --output-dir afternoon_issues

# Compare results
ls -ltr morning_issues/
ls -ltr afternoon_issues/
```

### Example 4: Direct File Analysis with Custom Output

```bash
# Analyze files directly with organized output
./scripts/analyze_logs.sh \
  --log-files "logs/critical_error.log logs/system.log" \
  --output-dir critical_incident_analysis

# Check results
ls -ltr critical_incident_analysis/
```

### Example 5: Production Analysis Workflow

```bash
# Comprehensive production analysis workflow
./scripts/load_logs.sh --elk-index production-logs

# All sessions overview
./scripts/analyze_logs.sh --elk-index production-logs --output-dir full_production_analysis

# Critical session deep dive
./scripts/analyze_logs.sh --elk-index production-logs --session-id critical_session --output-dir critical_session_analysis

# Time-based incident analysis
./scripts/analyze_logs.sh --elk-index production-logs \
  --start-time 2024-03-20T14:30:00 \
  --end-time 2024-03-20T15:30:00 \
  --output-dir incident_1430_analysis

# Generate summary report
echo "Production Analysis Summary:" > production_summary.txt
echo "=========================" >> production_summary.txt
echo "Full Analysis Files: $(ls full_production_analysis/ | wc -l)" >> production_summary.txt
echo "Critical Session Files: $(ls critical_session_analysis/ | wc -l)" >> production_summary.txt
echo "Incident Analysis Files: $(ls incident_1430_analysis/ | wc -l)" >> production_summary.txt
cat production_summary.txt
```

## Troubleshooting

### Services Not Running
```bash
# Check service status
docker-compose -f config/docker-compose.yml ps

# Start services
docker-compose -f config/docker-compose.yml up -d
```

### Elasticsearch Issues
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# View logs
docker-compose -f config/docker-compose.yml logs elasticsearch
```

### AI Service Issues
```bash
# Check Ollama models
docker-compose -f config/docker-compose.yml exec ollama ollama list

# View analysis logs
docker-compose -f config/docker-compose.yml logs log-analyzer
```

### Output Directory Issues
```bash
# Check if custom directory exists and has write permissions
ls -ld my_custom_output/

# Verify Docker volume mounts
docker-compose -f config/docker-compose.yml exec log-analyzer ls -la /app/

# Check available disk space
df -h

# View recent analysis files across all directories
find . -name "*.json" -type f -mtime -1 -exec ls -lt {} +
``` 