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

# Copy project metadata + lock for reproducible install
COPY pyproject.toml uv.lock README.md ./

# Source package required because `-e .` is resolved by `uv sync`
COPY src/ ./src/

# Install dependencies strictly from uv.lock for a fully reproducible build.
# numpy<2 / langchain-core 1.2 / langgraph 1.0.5 pinning lives in pyproject + lock,
# so no extra ad-hoc `pip install` lines here.
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# Application entry-point scripts (not part of the installed package)
COPY run.py ./
COPY pdf2bpmn_agent_executor.py ./
COPY pdf2bpmn_agent_server.py ./
COPY pdf2bpmn_scaledjob_worker.py ./
COPY a2a_server.py ./
COPY a2a_client.py ./

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
# - Check Apache AGE connectivity (required for extraction pipeline).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.pdf2bpmn.graph.neo4j_client import Neo4jClient; c=Neo4jClient(); ok=c.verify_connection(); c.close(); import sys; sys.exit(0 if ok else 1)"

# Run both servers via entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
