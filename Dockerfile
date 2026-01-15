# PDF2BPMN Docker Image
# PDF to BPMN Converter with Agent Server
# 
# Build: docker build -t ghcr.io/uengine-oss/pdf2bpmn:v0.0.1 .
# Run: docker run -p 8000:8000 -p 8001:8001 --env-file agent.env ghcr.io/uengine-oss/pdf2bpmn:v0.0.1

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY pyproject.toml ./
COPY requirements-agent.txt ./

# Install Python dependencies
# 1. Install main package dependencies from pyproject.toml
RUN pip install --no-cache-dir \
    langgraph>=0.2.0 \
    langchain>=0.3.0 \
    langchain-openai>=0.2.0 \
    langchain-community>=0.3.0 \
    neo4j>=5.0.0 \
    openai>=1.0.0 \
    pypdf>=4.0.0 \
    pdfplumber>=0.11.0 \
    tiktoken>=0.7.0 \
    numpy>=1.26.0 \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0 \
    jinja2>=3.1.0 \
    uuid6>=2024.0.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    python-multipart>=0.0.12

# 2. Install agent-specific dependencies
RUN pip install --no-cache-dir -r requirements-agent.txt

# Copy application source
COPY src/ ./src/
COPY run.py ./
COPY pdf2bpmn_agent_executor.py ./
COPY pdf2bpmn_agent_server.py ./

# Create necessary directories
RUN mkdir -p /app/output /app/uploads

# Copy and set up entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Expose ports
# 8000: Agent Server (ProcessGPT SDK polling)
# 8001: FastAPI Server (PDF2BPMN API)
EXPOSE 8000 8001

# Health check - check both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Run both servers via entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
