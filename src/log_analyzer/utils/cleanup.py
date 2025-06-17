"""
Utility for cleaning up old analysis outputs.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from log_analyzer.config.settings import (
    ANALYSIS_OUTPUT_DIR,
    ANALYSIS_RETENTION_DAYS,
    ANALYSIS_OUTPUT_FORMAT
)

def cleanup_old_analyses(days: Optional[int] = None) -> int:
    """
    Clean up analysis output directories older than the specified number of days.
    
    Args:
        days: Number of days to keep analysis outputs. If None, uses ANALYSIS_RETENTION_DAYS.
    
    Returns:
        Number of directories removed
    """
    if days is None:
        days = ANALYSIS_RETENTION_DAYS
    
    cutoff_date = datetime.now() - timedelta(days=days)
    removed_count = 0
    
    for item in ANALYSIS_OUTPUT_DIR.iterdir():
        if not item.is_dir():
            continue
            
        try:
            # Parse directory name as timestamp
            dir_date = datetime.strptime(item.name, ANALYSIS_OUTPUT_FORMAT)
            if dir_date < cutoff_date:
                shutil.rmtree(item)
                removed_count += 1
        except ValueError:
            # Skip directories that don't match the timestamp format
            continue
    
    return removed_count

def list_analysis_outputs() -> list[tuple[str, datetime, int]]:
    """
    List all analysis output directories with their timestamps and file counts.
    
    Returns:
        List of tuples containing (directory_name, timestamp, file_count)
    """
    outputs = []
    
    for item in ANALYSIS_OUTPUT_DIR.iterdir():
        if not item.is_dir():
            continue
            
        try:
            # Parse directory name as timestamp
            dir_date = datetime.strptime(item.name, ANALYSIS_OUTPUT_FORMAT)
            file_count = len(list(item.glob("*.json")))
            outputs.append((item.name, dir_date, file_count))
        except ValueError:
            continue
    
    # Sort by timestamp, newest first
    return sorted(outputs, key=lambda x: x[1], reverse=True)

def main():
    """Command-line interface for cleanup utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up old analysis outputs")
    parser.add_argument(
        "--days",
        type=int,
        help=f"Number of days to keep analysis outputs (default: {ANALYSIS_RETENTION_DAYS})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all analysis outputs instead of cleaning up"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAnalysis Outputs:")
        print("----------------")
        for dir_name, timestamp, file_count in list_analysis_outputs():
            print(f"{dir_name}: {file_count} files (from {timestamp})")
    else:
        removed = cleanup_old_analyses(args.days)
        print(f"Removed {removed} old analysis output directories")

if __name__ == "__main__":
    main() 