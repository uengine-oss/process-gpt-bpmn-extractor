# ProcessGPT BPMN Extractor Docker Image
# PDF to BPMN Converter with Agent Server
#
# Build: docker build --no-cache -t ghcr.io/uengine-oss/process-gpt-bpmn-extractor:dev .
# Run: docker run -p 8000:8000 -p 8001:8001 --env-file .env ghcr.io/uengine-oss/process-gpt-bpmn-extractor:dev

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    # Office → PDF conversion
    libreoffice \
    ca-certificates \
    locales \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-nanum \
    # OCR (Korean + English)
    tesseract-ocr \
    tesseract-ocr-kor \
    # Some libs commonly needed by renderers
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Korean locale (helps some rendering/text handling)
RUN sed -i 's/^# *ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

# Install uv for faster dependency resolution
RUN pip install uv

# Copy requirements files
COPY pyproject.toml ./
COPY uv.lock ./
COPY requirements-agent.txt ./
# Needed for `-e .` (pyproject readme points to README.md)
COPY README.md ./

# Copy application source (needed for `-e .` in requirements-agent.txt)
COPY src/ ./src/
COPY run.py ./
COPY pdf2bpmn_agent_executor.py ./
COPY pdf2bpmn_agent_server.py ./
COPY pdf2bpmn_scaledjob_worker.py ./
COPY a2a_server.py ./
COPY a2a_client.py ./

# Install Python dependencies using uv (much faster with lock file)
# Use --system to install to system Python instead of creating a venv
# Note: numpy<2.0 for older CPU compatibility (no X86_V2 requirement)
RUN uv pip install --system "numpy<2.0" && \
    uv pip install --system -e . && \
    uv pip install --system process-gpt-agent-sdk==0.4.13 supabase>=2.0.0 httpx>=0.25.0 sse-starlette>=2.0.0

# Create necessary directories
RUN mkdir -p /app/output /app/uploads

# Copy and set up entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Expose ports
# 8000: Agent Server (ProcessGPT SDK polling)
EXPOSE 8000

# Health check
# - This image runs in polling mode (no FastAPI required).
# - Check Neo4j connectivity (required for extraction pipeline).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.pdf2bpmn.graph.neo4j_client import Neo4jClient; c=Neo4jClient(); ok=c.verify_connection(); c.close(); import sys; sys.exit(0 if ok else 1)"

# Run both servers via entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
