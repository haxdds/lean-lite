# Lean-Lite Dockerfile
# A lightweight QuantConnect LEAN runtime

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# If poetry.lock doesn't exist, install from requirements.txt
RUN if [ ! -f poetry.lock ]; then pip install -r requirements.txt; fi

# Copy application code
COPY src/ ./src/
COPY strategies/ ./strategies/
COPY requirements.txt ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash lean && \
    chown -R lean:lean /app
USER lean

# Expose port (if needed for web interface)
EXPOSE 8000

# Set default command
CMD ["python", "src/main.py"] 