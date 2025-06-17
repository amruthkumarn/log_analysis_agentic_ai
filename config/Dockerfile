# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Force a rebuild by changing an argument
ARG CACHEBUST=1

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY scripts/ scripts/

# Install the package and its dependencies
RUN pip install -r requirements.txt
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "log_analyzer.scripts.run_analysis"] 