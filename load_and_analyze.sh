#!/bin/bash

# Log Analysis Workflow Script
# This script demonstrates the two-step process:
# 1. Load logs into Elasticsearch
# 2. Run the analysis

set -e  # Exit on any error

echo "🚀 Log Analysis Workflow"
echo "========================"

# Default values
ELK_INDEX="test-logs-default"
LOG_FILES="logs/3scale_api_gateway.log logs/payment_service.log logs/tibco_businessworks.log"
START_TIME="2024-01-01T00:00:00"
END_TIME="2026-01-01T00:00:00"

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
        --skip-load)
            SKIP_LOAD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --elk-index INDEX     Elasticsearch index name (default: $ELK_INDEX)"
            echo "  --log-files FILES     Space-separated log file paths (default: $LOG_FILES)"
            echo "  --start-time TIME     Analysis start time (default: $START_TIME)"
            echo "  --end-time TIME       Analysis end time (default: $END_TIME)"
            echo "  --skip-load           Skip the log loading step"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "   ELK Index: $ELK_INDEX"
echo "   Log Files: $LOG_FILES"
echo "   Time Range: $START_TIME to $END_TIME"
echo ""

# Step 1: Load logs into Elasticsearch (unless skipped)
if [[ "$SKIP_LOAD" != "true" ]]; then
    echo "📥 Step 1: Loading logs into Elasticsearch..."
    echo "   Index: $ELK_INDEX"
    echo "   Files: $LOG_FILES"
    
    docker-compose exec log-analyzer python src/load_logs_to_elk.py \
        --log-files $LOG_FILES \
        --elk-index "$ELK_INDEX"
    
    echo "✅ Logs loaded successfully!"
    echo ""
else
    echo "⏭️  Step 1: Skipped (--skip-load specified)"
    echo ""
fi

# Step 2: Run the analysis
echo "🔍 Step 2: Running log analysis..."
echo "   Source: ELK Index '$ELK_INDEX'"
echo "   Time Range: $START_TIME to $END_TIME"

docker-compose exec log-analyzer python src/run_analysis.py \
    --elk-index "$ELK_INDEX" \
    --start-time "$START_TIME" \
    --end-time "$END_TIME"

echo ""
echo "🎉 Analysis workflow complete!"
echo "📁 Check the analysis_output/ directory for results:"
echo "   - full_analysis_<session_id>_<timestamp>.json"
echo "   - root_cause_analysis_<session_id>_<timestamp>.json (if root causes found)" 