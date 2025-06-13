#!/bin/bash

# Log Loading Script
# Loads log files into Elasticsearch for later analysis

set -e  # Exit on any error

echo "📥 Log Loading to Elasticsearch"
echo "==============================="

# Default values
ELK_INDEX="test-logs-default"
LOG_FILES="logs/3scale_api_gateway.log logs/payment_service.log logs/tibco_businessworks.log"
ELK_HOST="elasticsearch"

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
        --elk-host)
            ELK_HOST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Load log files into Elasticsearch for analysis."
            echo ""
            echo "Options:"
            echo "  --elk-index INDEX     Elasticsearch index name (default: $ELK_INDEX)"
            echo "  --log-files FILES     Space-separated log file paths (default: $LOG_FILES)"
            echo "  --elk-host HOST       Elasticsearch host (default: $ELK_HOST)"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --elk-index my-logs --log-files \"logs/app1.log logs/app2.log\""
            echo "  $0 --elk-host localhost"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "   ELK Host: $ELK_HOST"
echo "   ELK Index: $ELK_INDEX"
echo "   Log Files: $LOG_FILES"
echo ""

# Check if Docker Compose is running
if ! docker-compose ps | grep -q "log-analyzer.*Up"; then
    echo "❌ Error: log-analyzer container is not running"
    echo "Please start the services first: docker-compose up -d"
    exit 1
fi

# Check if Elasticsearch is healthy
echo "🔍 Checking Elasticsearch health..."
if ! docker-compose exec elasticsearch curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    echo "❌ Error: Elasticsearch is not responding"
    echo "Please ensure Elasticsearch is running and healthy"
    exit 1
fi

echo "✅ Elasticsearch is healthy"
echo ""

# Load the logs
echo "📥 Loading logs into Elasticsearch..."
echo "   Target Index: $ELK_INDEX"

# Convert space-separated files to array for proper argument passing
IFS=' ' read -ra FILE_ARRAY <<< "$LOG_FILES"

docker-compose exec log-analyzer python src/load_logs_to_elk.py \
    --elk-host "$ELK_HOST" \
    --elk-index "$ELK_INDEX" \
    --log-files "${FILE_ARRAY[@]}"

echo ""
echo "🎉 Log loading complete!"
echo "📊 Logs are now available in Elasticsearch index: $ELK_INDEX"
echo ""
echo "Next steps:"
echo "  1. Run analysis: ./analyze_logs.sh --elk-index $ELK_INDEX"
echo "  2. Or view in Kibana: http://localhost:5601" 