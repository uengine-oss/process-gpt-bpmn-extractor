# ProcessGPT BPMN Extractor (PDF2BPMN)

업무 편람/업무 정의서(PDF)에서 **프로세스 / 태스크 / 역할**을 추출하고, **BPMN XML**, **DMN 규칙**, **Agent Skill 문서**를 자동 생성하는 도구입니다.

## 🎯 주요 기능

- 📄 **PDF 문서 분석**: 업무편람, SOP, 정책/규정 문서에서 프로세스 요소 추출
- 🔄 **BPMN 생성**: 프로세스, 태스크, 역할, 게이트웨이를 BPMN 2.0 XML로 변환
- 📊 **DMN 생성**: 조건 분기와 의사결정 로직을 DMN 테이블로 변환
- 🤖 **Skill 문서 생성**: AI 에이전트용 Claude Skills 형식의 Markdown 문서 생성
- 📈 **Neo4j 지식그래프**: 모든 엔티티를 Neo4j에 저장하여 탐색/재사용 가능
- 🙋 **Human-in-the-loop**: 모호한 항목에 대해 사용자 확인 요청
- 🎨 **Vue.js 프론트엔드**: BPMN.io 기반 프로세스 시각화 및 출처 추적 기능

## 설치

### 사전 요구사항

- Python 3.11+
- Neo4j Desktop (또는 Neo4j AuraDB)
- OpenAI API Key

### uv를 사용한 설치

```bash
# uv 설치 (이미 설치되어 있다면 건너뛰기)
pip install uv

# 프로젝트 클론 및 이동
cd process-gpt-bpmn-extractor

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 OpenAI API Key와 Neo4j 연결 정보 입력
```

### Neo4j 설정

1. [Neo4j Desktop](https://neo4j.com/download/) 설치
2. 새 데이터베이스 생성
3. 비밀번호 설정 (기본값: `1234567bpmn`)
4. 데이터베이스 시작

## 사용법

### CLI 모드

```bash
# PDF 파일 변환
uv run python run.py convert your_document.pdf

# 여러 파일 변환
uv run python run.py convert doc1.pdf doc2.pdf doc3.pdf

# HITL 질문 건너뛰기
uv run python run.py convert your_document.pdf --skip-hitl

# Neo4j 스키마 초기화
uv run python run.py init
```

### Web UI 모드 (Vue.js - 권장)

```bash
# 백엔드 API 서버 시작
uv run python run.py api

# 새 터미널에서 프론트엔드 시작
cd frontend && npm install && npm run dev
```

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8001/docs

또는 한 번에 실행:

```bash
./start.sh
```

## Docker로 실행 (권장)

이 저장소는 Docker/Compose로 바로 실행할 수 있습니다.

### 1) 사전 준비

- `.env`: `OPENAI_API_KEY` 등 런타임 환경변수
- `agent.env`: ProcessGPT SDK/워크아이템 처리용 환경변수 (예: `AGENT_*`, Supabase 등)

예시 파일:

- `agent.env.example`
- `api.env copy.example`

### 2) Neo4j + 전체 스택(Agent + API) 실행

```bash
docker compose up --build
```

- Agent Server: `http://localhost:8000`
- API Server: `http://localhost:8001`
- API Docs: `http://localhost:8001/docs`
- Neo4j Browser: `http://localhost:7474`

### 3) API만 실행(로컬 변환/백엔드 테스트)

```bash
docker compose --profile api-only up --build
```

### 4) GHCR 이미지로 실행(빌드 없이)

GitHub Actions가 `ghcr.io/uengine-oss/process-gpt-bpmn-extractor`로 이미지를 퍼블리시합니다.

```bash
docker pull ghcr.io/uengine-oss/process-gpt-bpmn-extractor:main
docker run --rm -p 8000:8000 -p 8001:8001 --env-file agent.env ghcr.io/uengine-oss/process-gpt-bpmn-extractor:main
```

### Streamlit UI (레거시)

```bash
uv run python run.py ui
```

브라우저에서 `http://localhost:8501` 접속

## 출력 파일

변환 완료 후 `output/` 폴더에 다음 파일들이 생성됩니다:

- `process.bpmn` - BPMN 2.0 XML
- `decisions.dmn` - DMN 규칙 테이블
- `decisions.json` - DMN JSON 형식
- `*.skill.md` - 각 에이전트 태스크별 스킬 문서

## Neo4j 그래프 탐색

Neo4j Browser (`http://localhost:7474`)에서 다음 쿼리로 데이터 탐색:

```cypher
-- 모든 프로세스 조회
MATCH (p:Process) RETURN p

-- 프로세스와 태스크 관계 조회
MATCH (p:Process)-[:HAS_TASK]->(t:Task) RETURN p, t

-- 태스크와 역할 관계 조회
MATCH (t:Task)-[:PERFORMED_BY]->(r:Role) RETURN t, r

-- 전체 그래프 시각화
MATCH (n) RETURN n LIMIT 100
```

## 아키텍처

```
PDF 입력 → 텍스트 추출 → 엔티티 추출(LLM) → 정규화/중복제거 
        → HITL 질문 → 스킬/DMN/BPMN 생성 → Neo4j 저장 → 출력
```

### 기술 스택

- **LangGraph**: 워크플로우 오케스트레이션
- **OpenAI GPT-4.1**: 엔티티 추출 및 분석
- **Neo4j**: 지식그래프 저장
- **pdfplumber**: PDF 텍스트 추출
- **Streamlit**: Web UI
- **Jinja2**: 템플릿 기반 문서 생성

## 설정

`.env` 파일에서 설정:

```env
# OpenAI
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=1234567bpmn
```

## 라이선스

MIT License

