import os
import logging
from log_analysis_agent import main

def setup_logging():
    """Configure logging for the analysis run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis_run.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def verify_log_files():
    """Verify that required log files exist."""
    required_files = [
        'logs/3scale_api_gateway.log',
        'logs/tibco_businessworks.log'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required log files: {', '.join(missing_files)}")
    
    return required_files

def verify_documentation():
    """Verify that documentation exists."""
    if not os.path.exists('documentation/system_architecture.md'):
        raise FileNotFoundError("Missing required documentation: documentation/system_architecture.md")

if __name__ == "__main__":
    logger = setup_logging()
    
    try:
        # Ensure required directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("documentation", exist_ok=True)
        os.makedirs("analysis_output", exist_ok=True)
        
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