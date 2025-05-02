FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create cache directory
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    SENTENCE_TRANSFORMERS_HOME=/app/cache \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gradio

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
