# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools and ollama CLI
RUN apt-get update && apt-get install -y build-essential curl && \
    curl -fsSL https://ollama.com/install.sh | sh

# Upgrade core packaging tools to handle modern builds
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# The command to run the application will be specified in docker-compose.yml 