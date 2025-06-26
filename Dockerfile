# Production Dockerfile for Data Analytics Portfolio
# Multi-stage build for optimized image size and security

# Stage 1: Base dependencies
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    libgomp1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base AS dependencies

# Install poetry for dependency management
RUN pip install poetry==${POETRY_VERSION}

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* requirements.txt* ./

# Install production dependencies only
RUN if [ -f pyproject.toml ]; then \
        poetry config virtualenvs.create false && \
        poetry install --no-interaction --no-ansi --no-dev; \
    elif [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Stage 3: Application layer
FROM base AS application

# Create non-root user for security
RUN groupadd -r analytics && useradd -r -g analytics analytics

# Set work directory
WORKDIR /app

# Copy installed dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=analytics:analytics . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/outputs && \
    chown -R analytics:analytics /app

# Security hardening
RUN chmod -R 755 /app && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \;

# Switch to non-root user
USER analytics

# Expose port for API (if applicable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command - can be overridden
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Production optimized (optional, for minimal size)
FROM python:3.11-slim-bullseye AS production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r analytics && useradd -r -g analytics analytics

# Set work directory
WORKDIR /app

# Copy only necessary files from application stage
COPY --from=application --chown=analytics:analytics /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=application --chown=analytics:analytics /usr/local/bin /usr/local/bin
COPY --from=application --chown=analytics:analytics /app/src /app/src
COPY --from=application --chown=analytics:analytics /app/models /app/models
COPY --from=application --chown=analytics:analytics /app/config /app/config

# Create runtime directories
RUN mkdir -p /app/logs /app/data /app/outputs && \
    chown -R analytics:analytics /app

# Set environment variables for production
ENV PYTHONPATH=/app \
    ENVIRONMENT=production \
    LOG_LEVEL=WARNING

# Switch to non-root user
USER analytics

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "warning", \
     "--access-log", \
     "--use-colors"]