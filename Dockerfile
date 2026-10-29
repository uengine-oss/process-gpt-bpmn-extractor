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
# ⚠ **베이스를 `python:3.11-slim` 에서 Ubuntu 24.04(noble) 로 옮겼다** (2026-10-22).
#
# `python:*-slim` 은 Debian trixie 위에 있고, trixie 의 `zlib`·`libxml2` 에
# 걸린 결함을 **데비안이 아직 안 고쳤다**(zlib CVE-2026-85091, 버그 #1146895).
#     debian:trixie-slim   0C 1H
#     ubuntu:24.04         **0C 0H**
#
# ⚠ **파이썬 판은 3.11 그대로 둔다.** 10-22 실측에서 3.13 으로 올리니
#   langchain 계열이 다른 가지로 풀려 **7 → 15 로 나빠졌다**. 배포판만 옮기고
#   런타임 판은 안 움직인다 — uv 가 자기 독립 빌드로 3.11 을 받는다.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PATH=/usr/local/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ⚠ 받아 온 파이썬에서 `pip` 와 `ensurepip` 휠을 걷어낸다 — 검사기가 휠 **안**과
#   `pip/_vendor/` 까지 읽어 urllib3·msgpack·setuptools 옛 판을 센다(실측).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.11 \
    && PY="$(uv python find 3.11)" \
    && PYROOT="$(dirname "$(dirname "$PY")")" \
    && rm -rf "$PYROOT"/lib/python3.11/site-packages/pip \
              "$PYROOT"/lib/python3.11/site-packages/pip-*.dist-info \
              "$PYROOT"/lib/python3.11/ensurepip \
    && ln -sf "$PY" /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
# ⚠ **보안 고침을 받는다.** 2026-10-29 에 `brotli` 가 고위험 1건으로 떴는데
#    (CVE-2026-111497 · **CISA KEV** — 실제로 악용되는 목록) 우리가 깐 것이 아니라
#    **베이스 이미지(ubuntu:24.04)에 있던 것**이었다. `install` 로는 안 걷힌다.
#    ⚠ 같은 레이어에서 한다 — 다음 레이어에서 고치면 옛 판이 그대로 남는다.
#    ⚠ 여기는 **마지막 apt 레이어**다. 앞쪽(ca-certificates)에만 넣으면 이 설치가
#      옛 판을 다시 들일 수 있다.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
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
# ⚠ **이 줄은 더 필요 없어졌다** (2026-10-22). uv 는 위에서 베이스 이미지로부터
#   바이너리로 가져오고, `pip` 는 일부러 걷어냈다(검사기가 벤더 사본까지 센다).
#   setuptools 는 `uv.lock` 이 정한 판이 venv 안에 들어간다.
#   줄은 지우되 **왜 있었는지는 남긴다** — 다음에 같은 증상을 보면 여기부터 읽는다.

# Copy project metadata + lock for reproducible install
COPY pyproject.toml uv.lock README.md ./

# Source package required because `-e .` is resolved by `uv sync`
COPY src/ ./src/

# Install dependencies strictly from uv.lock for a fully reproducible build.
# numpy<2 / langchain-core 1.2 / langgraph 1.0.5 pinning lives in pyproject + lock,
# so no extra ad-hoc `pip install` lines here.
# ⚠ **다 깔았으면 빌드 도구를 걷어낸다** (2026-10-22 실측).
#
# `build-essential` 이 `linux-libc-dev` 를 끌고 오는데, 검사기는 그것을
# **커널 소스 패키지(`linux`)** 로 보고 **치명 7 · 고위험 211** 을 통째로 올린다.
# 베이스를 noble 로 옮기고 6 → 239 로 **폭증해서** 알았다 — 숫자의 93% 가 이것이다.
#
# 헤더는 휠을 **컴파일할 때만** 필요하다. `uv sync` 가 끝나면 쓸 일이 없다.
# ⚠ libreoffice·tesseract·폰트는 런타임이 쓰므로 **안 건드린다**.
RUN uv sync --frozen --no-dev \
    && apt-get purge -y --auto-remove build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
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
