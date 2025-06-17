#!/usr/bin/env python3
"""
Standalone script for running log analysis.
This script is separate from the ELK loading process.
"""

import argparse
import os
import sys
from datetime import datetime

# Import the main analysis function
from log_analyzer.core.redis_log_analysis_agent import main as run_analysis_main

def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description="Log Analysis AI Agent with Redis and Checkpointing")
    
    # Log source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--log-files', type=str, nargs='+', help='Paths to log files.')
    source_group.add_argument('--elk-index', type=str, help='Elasticsearch index name.')

    # Session filtering
    parser.add_argument('--session-id', type=str, help='Analyze only this specific session ID.')

    # Time filtering
    time_group = parser.add_argument_group('Time Filtering')
    time_group.add_argument('--start-time', type=datetime.fromisoformat, help='Start time (YYYY-MM-DDTHH:MM:SS).')
    time_group.add_argument('--end-time', type=datetime.fromisoformat, help='End time (YYYY-MM-DDTHH:MM:SS).')

    # ELK configuration
    elk_group = parser.add_argument_group('ELK Configuration')
    elk_group.add_argument('--elk-host', type=str, default=os.getenv("ELASTICSEARCH_HOST", "elasticsearch"))
    elk_group.add_argument('--elk-user', type=str, default=os.getenv("ELASTICSEARCH_USER"))
    elk_group.add_argument('--elk-password', type=str, default=os.getenv("ELASTICSEARCH_PASSWORD"))
    elk_group.add_argument('--elk-max-results', type=int, default=10000)
    elk_group.add_argument('--elk-time-field', type=str, default='@timestamp')
    elk_group.add_argument('--elk-service-field', type=str, default='service.name')
    elk_group.add_argument('--elk-message-field', type=str, default='message')
    elk_group.add_argument('--elk-level-field', type=str, default='log.level')

    args = parser.parse_args()
    
    # Validation
    if args.end_time and not args.start_time:
        parser.error("--end-time requires --start-time.")
    
    print("🚀 Starting Log Analysis...")
    print(f"📊 Source: {'ELK Index: ' + args.elk_index if args.elk_index else 'Log Files: ' + ', '.join(args.log_files)}")
    
    if args.session_id:
        print(f"🔍 Filtering to session: {args.session_id}")
    
    if args.start_time:
        print(f"⏰ Time Range: {args.start_time} to {args.end_time or 'now'}")
    
    # Run the analysis
    run_analysis_main(args)
    
    print("✅ Analysis complete! Check the analysis_output/ directory for results.")

if __name__ == '__main__':
    main() 