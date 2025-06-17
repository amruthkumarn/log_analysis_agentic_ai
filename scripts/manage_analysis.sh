#!/bin/bash

# Script to manage analysis outputs

function show_help {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  list              List all analysis outputs"
    echo "  clean [days]      Clean up analysis outputs older than specified days (default: 7)"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 list                    # List all analysis outputs"
    echo "  $0 clean                   # Clean up outputs older than 7 days"
    echo "  $0 clean 30                # Clean up outputs older than 30 days"
}

case "$1" in
    "list")
        python -m log_analyzer.utils.cleanup --list
        ;;
    "clean")
        if [ -z "$2" ]; then
            python -m log_analyzer.utils.cleanup
        else
            python -m log_analyzer.utils.cleanup --days "$2"
        fi
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 