import os
import logging
from pathlib import Path
from .log_analysis_agent import main

# Helper function to get project root
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def setup_logging():
    """Configure logging for the analysis run."""
    project_root = get_project_root()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(project_root / 'analysis_run.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def verify_log_files():
    """Verify that required log files exist."""
    project_root = get_project_root()
    required_files = [
        project_root / 'logs/3scale_api_gateway.log',
        project_root / 'logs/tibco_businessworks.log'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        missing_files_str = [str(f) for f in missing_files]
        raise FileNotFoundError(f"Missing required log files: {', '.join(missing_files_str)}")
    
    return required_files

def verify_documentation():
    """Verify that documentation exists."""
    project_root = get_project_root()
    doc_path = project_root / 'documentation/system_architecture.md'
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Missing required documentation: {str(doc_path)}")

if __name__ == "__main__":
    logger = setup_logging()
    project_root = get_project_root()
    
    try:
        # Ensure required directories exist
        os.makedirs(project_root / "logs", exist_ok=True)
        os.makedirs(project_root / "documentation", exist_ok=True)
        os.makedirs(project_root / "analysis_output", exist_ok=True)
        
        # Verify required files
        logger.info("Verifying required files...")
        verify_log_files()
        verify_documentation()
        
        # Run the analysis
        logger.info("Starting log analysis...")
        main()
        logger.info("Analysis completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise 