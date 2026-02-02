# PDF2BPMN 에이전트 서버

ProcessGPT SDK를 활용한 PDF to BPMN 변환 에이전트입니다.

## 개요

이 에이전트는 PDF 문서(업무 매뉴얼, SOP 등)를 분석하여 BPMN XML을 자동 생성하고, 
생성된 프로세스를 Supabase에 저장합니다.

## 시스템 아키텍처

```
[사용자] → [프론트엔드] → [메인채팅 에이전트 (MCP)]
                              ↓
                    create_pdf2bpmn_workitem (todolist 추가)
                              ↓
            [PDF2BPMN 에이전트] ← polling (agent_orch='pdf2bpmn')
                              ↓
                    [Events 테이블] → 실시간 진행 상황
                              ↓
                    [proc_def, configuration 저장]
                              ↓
            [프론트엔드] ← events watch (실시간 결과 표시)
```

## 동작 흐름

1. **PDF 업로드**: 사용자가 프론트엔드에서 PDF 파일 업로드 및 메시지 전송
2. **Todolist 추가**: 메인채팅 에이전트가 MCP 도구(`create_pdf2bpmn_workitem`)로 작업 추가
3. **작업 폴링**: PDF2BPMN 에이전트가 `agent_orch='pdf2bpmn'` 조건의 todo를 폴링
4. **BPMN 생성**: (에이전트 내부에서) PDF 분석 → 엔티티 추출 → BPMN XML 생성
5. **이벤트 전송**: 각 BPMN 생성 시 events 테이블에 실시간 이벤트 전송
6. **저장**: 생성된 BPMN을 proc_def 및 configuration(proc_map)에 저장
7. **프론트엔드 표시**: events watch로 실시간 진행 상황 표시

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements-agent.txt
```

또는 uv 사용:
```bash
uv pip install -r requirements-agent.txt
```

> 참고: `requirements-agent.txt`에는 `-e .`가 포함되어 있어, 프로젝트 메인 의존성(neo4j/langgraph/pdfplumber 등)도 함께 설치됩니다.

### 2. 환경 설정

`agent.env.example`을 `agent.env` 또는 `.env.local`로 복사하고 설정:

```bash
cp agent.env.example agent.env
```

환경 변수:
```env
# Supabase 설정
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SERVICE_ROLE_KEY=your-service-role-key

# 에이전트 설정
AGENT_ORCH=pdf2bpmn
POLLING_INTERVAL=5
TASK_TIMEOUT=300
```

### 3. 에이전트 서버 실행

```bash
uv run python pdf2bpmn_agent_server.py
```

> 주의: 시스템 Python(예: `C:\\Python313\\python.exe`)로 실행하면 패키지가 누락될 수 있습니다.  
> `uv run` 또는 `.venv` 활성화 후 실행하세요.

## 파일 구조

```
pdf2bpmn/
├── pdf2bpmn_agent_executor.py  # AgentExecutor 구현 (핵심 로직)
├── pdf2bpmn_agent_server.py    # 에이전트 서버 진입점
├── agent.env.example           # 환경 설정 예제
├── requirements-agent.txt      # 에이전트 의존성
└── src/pdf2bpmn/              # 기존 PDF2BPMN API
    ├── api/main.py            # FastAPI 서버
    ├── workflow/graph.py      # LangGraph 워크플로우
    ├── extractors/            # PDF/엔티티 추출기
    └── generators/            # BPMN/DMN 생성기
```

## 이벤트 형식

에이전트는 다음 형식의 이벤트를 전송합니다:

### 진행 상황 이벤트
```json
{
  "task_id": "uuid",
  "event_type": "pdf2bpmn_progress",
  "message": {
    "status": "processing",
    "progress": 50,
    "message": "BPMN 생성 중: 결재요청"
  }
}
```

### BPMN 생성 아티팩트
```json
{
  "task_id": "uuid",
  "artifact": {
    "parts": [{
      "type": "text",
      "text": {
        "type": "bpmn",
        "process_id": "approval_process",
        "process_name": "결재요청",
        "bpmn_xml": "<?xml version=\"1.0\"...>"
      }
    }]
  }
}
```

### 완료 이벤트
```json
{
  "task_id": "uuid",
  "event_type": "task_completed",
  "message": {
    "status": "completed",
    "process_count": 3,
    "saved_processes": [
      {"id": "proc_1", "name": "프로세스1"},
      {"id": "proc_2", "name": "프로세스2"}
    ]
  }
}
```

## 프론트엔드 연동

### WorkAssistantChatPanel.vue

PDF2BPMN 작업이 감지되면 자동으로 events 테이블을 watch하여:
- 실시간 진행 상황 표시
- 생성된 BPMN 목록 표시
- BPMN XML 미리보기 제공
- 완료 시 자동으로 정의 목록 새로고침

## 데이터베이스 스키마

### todolist 테이블
에이전트가 작업을 폴링하는 테이블

```sql
CREATE TABLE todolist (
  id UUID PRIMARY KEY,
  task_id UUID,
  tenant_id TEXT,
  user_uid UUID,
  agent_orch TEXT,  -- 'pdf2bpmn'
  query JSONB,      -- {pdf_url, pdf_file_name, ...}
  status TEXT,      -- 'pending', 'in_progress', 'completed', 'failed'
  ...
);
```

### events 테이블
실시간 이벤트 전송용 테이블

```sql
CREATE TABLE events (
  id UUID PRIMARY KEY,
  task_id UUID,
  event_type TEXT,
  message JSONB,
  artifact JSONB,
  created_at TIMESTAMPTZ
);
```

### proc_def 테이블
생성된 프로세스 정의 저장

```sql
CREATE TABLE proc_def (
  id TEXT PRIMARY KEY,
  name TEXT,
  definition JSONB,
  bpmn TEXT,
  tenant_id TEXT,
  ...
);
```

## 문제 해결

### 에이전트가 작업을 가져오지 않음
- `AGENT_ORCH` 환경 변수가 'pdf2bpmn'으로 설정되었는지 확인
- Supabase 연결 정보 확인
- todolist 테이블에 `agent_orch='pdf2bpmn'` 조건의 작업이 있는지 확인

### BPMN 생성 실패
- 에이전트 로그에서 오류 메시지 확인
- `OPENAI_API_KEY`, Neo4j 연결 정보(예: `NEO4J_URI`)가 올바른지 확인
- 문서 다운로드 URL(`pdf_url`) 접근 가능 여부 확인

### 프론트엔드에서 이벤트가 표시되지 않음
- Supabase Realtime이 활성화되어 있는지 확인
- events 테이블에 RLS 정책이 올바르게 설정되어 있는지 확인
- 브라우저 콘솔에서 WebSocket 연결 상태 확인
