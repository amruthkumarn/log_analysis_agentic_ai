# Log Analysis AI Agent

This is a multi-agent AI application built with LangGraph and LangChain for analyzing logs from 3scale and TIBCO systems. The application uses the Ollama LLM (llama2:3.1b) to provide intelligent log analysis and recommendations.

## Features

- Log analysis for 3scale and TIBCO systems
- Error pattern detection
- Performance issue identification
- Security concern analysis
- Actionable recommendations generation

## Prerequisites

- Python 3.8+
- Ollama with llama2:3.1b model installed
- UV package manager

## Installation

1. Clone the repository
2. Install dependencies using UV:
```bash
uv pip install -r requirements.txt
```

## Usage

Run the application with sample logs:
```bash
python log_analysis_agent.py
```

To analyze your own logs, modify the `sample_log` variable in `log_analysis_agent.py` with your log content.

## Architecture

The application uses a multi-agent system with two main components:

1. Log Analyzer Agent: Analyzes logs for patterns, errors, and issues
2. Recommendation Agent: Generates actionable recommendations based on the analysis

The agents communicate through a LangGraph workflow, ensuring a structured and efficient analysis process.

## Contributing

Feel free to submit issues and enhancement requests!

