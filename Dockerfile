# ProcessGPT BPMN Extractor Docker Image
# PDF to BPMN Converter with Agent Server
#
# Build: docker build --no-cache -t ghcr.io/uengine-oss/process-gpt-bpmn-extractor:dev .
# Run: docker run -p 8000:8000 -p 8001:8001 --env-file .env ghcr.io/uengine-oss/process-gpt-bpmn-extractor:dev

# ⚠ **파이썬 3.13 으로 올려 봤고, 되돌렸다** (2026-10-21 실측).
#
# 목적은 `protobuf` 한 건(CVE-2026-0994)이었다 — 잠금이 **3.13 이상에서만**
# 6.33.6 을 고르기 때문이다. 올리니 그 한 건은 사라졌는데 **다른 것이 여덟 개
# 올라왔다**: 3.13 에서는 langchain 계열이 다른 가지로 풀려 urllib3 2.7.0 2건 ·
# langsmith 2 · msgpack 2 · setuptools 70.3.0 1 · langchain-community 1 ·
# langchain-core 1 이 들어온다. **7 → 15 로 나빠진다.**
# 그래서 3.11 에 머문다. protobuf 한 건은 상류가 정리해야 풀린다.
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

# Install uv for faster dependency resolution.
#
# `setuptools` is upgraded in the same layer (2026-10-20 · robo-architect vuln
# patch): the copies **vendored inside setuptools** were the last two findings
# in the image — `jaraco-context 5.3.0` and `wheel 0.45.1`. They are not
# declared anywhere, so a lockfile bump cannot reach them; the packaging tool
# itself has to move.
RUN pip install --no-cache-dir uv 'setuptools>=84.0.0'

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
