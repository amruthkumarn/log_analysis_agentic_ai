#!/bin/bash

# Log Analysis Script
# Runs AI-powered analysis on logs (from ELK or files)

set -e  # Exit on any error

echo "🔍 Log Analysis with AI"
echo "======================="

# Default values
ELK_INDEX=""
LOG_FILES=""
START_TIME="2024-01-01T00:00:00"
END_TIME="2026-01-01T00:00:00"
ELK_HOST="elasticsearch"
OUTPUT_DIR="analysis_output"
SESSION_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --elk-index)
            ELK_INDEX="$2"
            shift 2
            ;;
        --log-files)
            LOG_FILES="$2"
            shift 2
            ;;
        --start-time)
            START_TIME="$2"
            shift 2
            ;;
        --end-time)
            END_TIME="$2"
            shift 2
            ;;
        --elk-host)
            ELK_HOST="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --session-id)
            SESSION_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run AI-powered analysis on log data."
            echo ""
            echo "Data Source (choose one):"
            echo "  --elk-index INDEX     Analyze logs from Elasticsearch index"
            echo "  --log-files FILES     Analyze logs from local files (space-separated)"
            echo ""
            echo "Options:"
            echo "  --start-time TIME     Analysis start time (default: $START_TIME)"
            echo "  --end-time TIME       Analysis end time (default: $END_TIME)"
            echo "  --elk-host HOST       Elasticsearch host (default: $ELK_HOST)"
            echo "  --output-dir DIR      Output directory for results (default: $OUTPUT_DIR)"
            echo "  --session-id ID       Analyze only specific session ID"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Analyze from Elasticsearch:"
            echo "  $0 --elk-index demo-logs"
            echo ""
            echo "  # Analyze specific session:"
            echo "  $0 --elk-index demo-logs --session-id gbx131"
            echo ""
            echo "  # Analyze with custom output directory:"
            echo "  $0 --elk-index demo-logs --output-dir custom_results"
            echo ""
            echo "  # Analyze from local files:"
            echo "  $0 --log-files \"logs/app1.log logs/app2.log\""
            echo ""
            echo "  # Analyze with time range:"
            echo "  $0 --elk-index my-logs --start-time 2024-03-20T10:00:00 --end-time 2024-03-20T11:00:00"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate input
if [[ -z "$ELK_INDEX" && -z "$LOG_FILES" ]]; then
    echo "❌ Error: Must specify either --elk-index or --log-files"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -n "$ELK_INDEX" && -n "$LOG_FILES" ]]; then
    echo "❌ Error: Cannot specify both --elk-index and --log-files"
    echo "Use --help for usage information"
    exit 1
fi

echo "📋 Configuration:"
if [[ -n "$ELK_INDEX" ]]; then
    echo "   Data Source: Elasticsearch"
    echo "   ELK Host: $ELK_HOST"
    echo "   ELK Index: $ELK_INDEX"
else
    echo "   Data Source: Local Files"
    echo "   Log Files: $LOG_FILES"
fi
echo "   Time Range: $START_TIME to $END_TIME"
echo "   Output Directory: $OUTPUT_DIR"
if [[ -n "$SESSION_ID" ]]; then
    echo "   Session Filter: $SESSION_ID"
fi
echo ""

# Check if Docker Compose is running
if ! docker-compose -f config/docker-compose.yml ps | grep -q "log-analyzer.*Up"; then
    echo "❌ Error: log-analyzer container is not running"
    echo "Please start the services first: docker-compose -f config/docker-compose.yml up -d"
    exit 1
fi

# Check if Ollama is healthy (for AI analysis)
echo "🔍 Checking AI services..."
if ! docker-compose -f config/docker-compose.yml exec ollama ollama list > /dev/null 2>&1; then
    echo "❌ Error: Ollama AI service is not responding"
    echo "Please ensure Ollama is running and healthy"
    exit 1
fi

echo "✅ AI services are ready"
echo ""

# Build the analysis command
ANALYSIS_CMD="docker-compose -f config/docker-compose.yml exec log-analyzer python -m src.log_analyzer.scripts.run_analysis"

if [[ -n "$ELK_INDEX" ]]; then
    # Elasticsearch source
    echo "🔍 Running analysis on Elasticsearch data..."
    echo "   Index: $ELK_INDEX"
    echo "   Time Range: $START_TIME to $END_TIME"
    
    ANALYSIS_CMD="$ANALYSIS_CMD --elk-index $ELK_INDEX --elk-host $ELK_HOST"
else
    # File source
    echo "🔍 Running analysis on local log files..."
    echo "   Files: $LOG_FILES"
    
    # Convert space-separated files to array for proper argument passing
    IFS=' ' read -ra FILE_ARRAY <<< "$LOG_FILES"
    ANALYSIS_CMD="$ANALYSIS_CMD --log-files ${FILE_ARRAY[*]}"
fi

# Add time range
ANALYSIS_CMD="$ANALYSIS_CMD --start-time $START_TIME --end-time $END_TIME"

# Add output directory
ANALYSIS_CMD="$ANALYSIS_CMD --output-dir $OUTPUT_DIR"

# Add session filter if specified
if [[ -n "$SESSION_ID" ]]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --session-id $SESSION_ID"
fi

echo ""
echo "🚀 Starting AI analysis..."
echo "   This may take several minutes depending on log volume"
echo ""

# Run the analysis
eval $ANALYSIS_CMD

echo ""
echo "🎉 Analysis complete!"
echo "📁 Results saved to $OUTPUT_DIR/ directory:"
echo "   - full_analysis_<session_id>_<timestamp>.json"
echo "   - root_cause_analysis_<session_id>_<timestamp>.json (if issues found)"
echo ""
echo "💡 Tip: Check the latest files with: ls -ltr $OUTPUT_DIR/" 