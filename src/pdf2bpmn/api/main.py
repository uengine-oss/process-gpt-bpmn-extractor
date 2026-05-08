"""FastAPI backend for PDF2BPMN frontend."""
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..graph.neo4j_client import Neo4jClient
from ..workflow.graph import PDF2BPMNWorkflow
from ..converters.file_to_pdf import convert_to_pdf, FileToPdfError


# ---------------------------------------------------------------------------
# AGE multi-graph helpers
# ---------------------------------------------------------------------------
# 한 todo(=task_id) 의 모든 프로세스 데이터는 `g_<tenant>_<task_id>` 그래프
# 하나에 누적 저장된다. tenant_id + task_id 가 함께 주어지면 정확히 그 그래프를
# 식별할 수 있고, 없을 때만 catalog 토큰 매칭/순회로 fallback 한다.

def _list_age_graphs() -> List[str]:
    """List all AGE graphs in the database (best-effort)."""
    probe = Neo4jClient()
    try:
        conn = probe._connect()  # noqa: SLF001 - intentional fallback access
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM ag_catalog.ag_graph ORDER BY name")
            rows = cur.fetchall() or []
        return [
            str(r[0]).strip()
            for r in rows
            if isinstance(r, (list, tuple)) and r and str(r[0]).strip()
        ]
    except Exception:
        return []
    finally:
        try:
            probe.close()
        except Exception:
            pass


def _candidate_graph_names(
    *,
    tenant_id: str = "",
    task_id: str = "",
    run_id: str = "",
    explicit: str = "",
) -> List[str]:
    """
    Return AGE graph name candidates ordered by priority.
      1. explicit (graph_name 직접 지정)
      2. tenant_id + task_id 로 build_graph_name 결과 (가장 정확)
      3. catalog 에서 task_id / run_id 토큰을 포함하는 그래프 (legacy fallback)
      4. 기본 그래프 (Config.AGE_GRAPH_NAME)
      5. 그 외 발견된 모든 그래프
    """
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        n = (name or "").strip()
        if n and n not in seen:
            candidates.append(n)
            seen.add(n)

    if explicit:
        _add(explicit)

    if tenant_id and task_id:
        try:
            _add(Neo4jClient.build_graph_name(tenant_id=tenant_id, todo_id=task_id))
        except Exception:
            pass

    discovered = _list_age_graphs()

    def _token(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]", "_", str(value or "")).strip("_").lower()

    for hint in (task_id, run_id):
        token = _token(hint)
        if not token:
            continue
        for g in discovered:
            if token in g.lower():
                _add(g)

    try:
        _add(Neo4jClient().graph_name)
    except Exception:
        pass

    for g in discovered:
        _add(g)

    return candidates


def _try_with_graph(name: str, fn: Callable[[Neo4jClient], Any]) -> Optional[Any]:
    """Run callable against the given graph; absorb missing-graph errors."""
    if not name:
        return None
    client = Neo4jClient(graph_name=name)
    try:
        return fn(client)
    except Exception:
        return None
    finally:
        try:
            client.close()
        except Exception:
            pass


def _query_across_graphs(
    fn: Callable[[Neo4jClient], Any],
    *,
    tenant_id: str = "",
    task_id: str = "",
    run_id: str = "",
    explicit: str = "",
) -> tuple[Optional[Any], List[str], Optional[str]]:
    """Iterate candidate graphs; return (data, tried_graphs, hit_graph_name)."""
    candidates = _candidate_graph_names(
        tenant_id=tenant_id, task_id=task_id, run_id=run_id, explicit=explicit
    )
    for name in candidates:
        hit = _try_with_graph(name, fn)
        if hit:
            return hit, candidates, name
    return None, candidates, None

app = FastAPI(
    title="PDF2BPMN API",
    description="PDF to BPMN Converter API",
    version="0.1.0"
)

# CORS for Vue.js frontend - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Store for job progress
job_progress = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, error
    current_step: str
    progress: int  # 0-100
    steps: list[dict]
    detail_message: Optional[str] = None  # 상세 진행 상황 메시지
    chunk_info: Optional[dict] = None  # 청크 처리 정보 {"current": 1, "total": 5}
    error: Optional[str] = None
    result: Optional[dict] = None


class EntityResponse(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None
    evidence: Optional[dict] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    neo4j = Neo4jClient()
    neo4j_ok = neo4j.verify_connection()
    neo4j.close()
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j_ok else "disconnected"
    }


@app.get("/api/neo4j/status")
async def get_neo4j_status():
    """Check if Neo4j has existing data."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            # Count existing data
            result = session.run("""
                MATCH (p:Process) WITH count(p) as processes
                MATCH (t:Task) WITH processes, count(t) as tasks
                MATCH (r:Role) WITH processes, tasks, count(r) as roles
                RETURN processes, tasks, roles
            """)
            record = result.single()
            
            has_data = record["processes"] > 0 or record["tasks"] > 0 or record["roles"] > 0
            
            return {
                "has_data": has_data,
                "counts": {
                    "processes": record["processes"],
                    "tasks": record["tasks"],
                    "roles": record["roles"]
                }
            }
    finally:
        neo4j.close()


@app.post("/api/neo4j/clear")
async def clear_neo4j():
    """Clear all data from Neo4j."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return {"message": "Neo4j data cleared successfully"}
    finally:
        neo4j.close()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing.

    - Accepts any file type.
    - If not a PDF, converts to PDF (best-effort) so the same downstream pipeline can run.
    """
    
    # Save file
    Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = Config.UPLOAD_DIR / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Convert to PDF if needed
    pdf_path = str(file_path)
    if not file.filename.lower().endswith(".pdf"):
        try:
            pdf_path = convert_to_pdf(str(file_path), str(Config.UPLOAD_DIR))
        except FileToPdfError as e:
            raise HTTPException(400, f"파일을 PDF로 변환하지 못했습니다: {e}")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_progress[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "current_step": "uploaded",
        "progress": 0,
        "steps": [],
        # downstream always uses PDF path
        "file_path": str(pdf_path),
        "file_name": file.filename,
        "original_file_path": str(file_path),
    }
    
    return {
        "job_id": job_id,
        "file_name": file.filename,
        "pdf_file_path": str(pdf_path),
        "message": "File uploaded successfully"
    }


@app.post("/api/process/{job_id}")
async def start_processing(job_id: str, background_tasks: BackgroundTasks):
    """Start processing an uploaded file."""
    if job_id not in job_progress:
        raise HTTPException(404, "Job not found")
    
    job = job_progress[job_id]
    if job["status"] == "processing":
        raise HTTPException(400, "Job already processing")
    
    # Start background processing
    background_tasks.add_task(process_pdf_background, job_id)
    
    job["status"] = "processing"
    return {"message": "Processing started", "job_id": job_id}


async def process_pdf_background(job_id: str):
    """Background task for PDF processing with real-time SSE updates."""
    import concurrent.futures
    
    job = job_progress[job_id]
    file_path = job["file_path"]
    
    steps = [
        {"name": "ingest_pdf", "label": "PDF 파싱", "status": "pending"},
        {"name": "segment_sections", "label": "섹션 분석 및 임베딩", "status": "pending"},
        {"name": "extract_candidates", "label": "엔티티 추출", "status": "pending"},
        {"name": "normalize_entities", "label": "정규화 및 중복 제거", "status": "pending"},
        {"name": "create_relationships", "label": "관계 생성", "status": "pending"},
        {"name": "generate_bpmn", "label": "BPMN 생성", "status": "pending"},
        {"name": "generate_dmn", "label": "DMN 생성", "status": "pending"},
        {"name": "export", "label": "결과 저장", "status": "pending"},
    ]
    job["steps"] = steps
    
    # Create thread pool for running sync operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    
    try:
        graph_name = Neo4jClient.build_graph_name("api", job_id)
        job["graph_name"] = graph_name
        workflow = PDF2BPMNWorkflow(graph_name=graph_name)
        
        # Initialize Neo4j schema (run in thread)
        update_step(job, 0, "processing", 5, "Neo4j 스키마 초기화 중...")
        await asyncio.sleep(0)  # Allow SSE to send
        await loop.run_in_executor(executor, workflow.neo4j.init_schema)
        
        # Clear existing Process-related nodes before creating new ones
        update_step(job, 0, "processing", 7, "기존 프로세스 노드 정리 중...")
        await asyncio.sleep(0)
        clear_result = await loop.run_in_executor(executor, workflow.neo4j.clear_process_core_labels)
        deleted_count = clear_result.get("deleted_nodes", 0)
        if deleted_count > 0:
            update_step(job, 0, "processing", 9, f"기존 노드 {deleted_count}개 삭제 완료")
            await asyncio.sleep(0)
        
        update_step(job, 0, "processing", 10, "PDF 파일 로딩 중...")
        await asyncio.sleep(0)
        
        # Step 1: Ingest PDF
        state = {
            "pdf_paths": [file_path],
            "documents": [],
            "sections": [],
            "reference_chunks": [],
            "processes": [],
            "tasks": [],
            "roles": [],
            "gateways": [],
            "events": [],
            "skills": [],
            "dmn_decisions": [],
            "dmn_rules": [],
            "evidences": [],
            "confidence_threshold": 0.8,
            "current_step": "ingest_pdf",
            "error": None,
            "bpmn_xml": None,
            "bpmn_xmls": {},
            "bpmn_files": {},
            "skill_docs": {},
            "dmn_xml": None
        }
        result = await loop.run_in_executor(executor, workflow.ingest_pdf, state)
        state.update(result)
        
        chunk_count = len(state.get("reference_chunks", []))
        page_count = state.get("documents", [{}])[0].page_count if state.get("documents") else 0
        update_step(job, 0, "completed", 15, f"PDF 파싱 완료: {page_count}페이지, {chunk_count}개 청크")
        await asyncio.sleep(0)
        
        # Step 2: Segment sections
        update_step(job, 1, "processing", 20, "섹션 분석 및 임베딩 생성 중...")
        await asyncio.sleep(0)
        result = await loop.run_in_executor(executor, workflow.segment_sections, state)
        state.update(result)
        section_count = len(state.get("sections", []))
        update_step(job, 1, "completed", 30, f"섹션 분석 완료: {section_count}개 섹션")
        await asyncio.sleep(0)
        
        # Step 3: Extract candidates (with chunk progress)
        chunks = state.get("reference_chunks", [])
        total_chunks = len(chunks)
        update_step(job, 2, "processing", 35, f"엔티티 추출 시작: {total_chunks}개 청크", 
                   {"current": 0, "total": total_chunks})
        await asyncio.sleep(0)
        
        # Custom extraction with progress updates (using polling approach)
        # Since we can't easily callback from thread, we'll use a different approach
        def extract_with_logging():
            return workflow.extract_candidates_with_progress(state, 
                lambda current, total, msg: update_step(
                    job, 2, "processing", 
                    35 + int((current / max(total, 1)) * 15),
                    msg,
                    {"current": current, "total": total}
                )
            )
        
        result = await loop.run_in_executor(executor, extract_with_logging)
        state.update(result)
        
        process_count = len(state.get("processes", []))
        task_count = len(state.get("tasks", []))
        role_count = len(state.get("roles", []))
        update_step(job, 2, "completed", 50, 
                   f"추출 완료: 프로세스 {process_count}, 태스크 {task_count}, 역할 {role_count}")
        await asyncio.sleep(0)
        
        # Step 4: Normalize
        update_step(job, 3, "processing", 55, "엔티티 정규화 및 중복 제거 중...")
        await asyncio.sleep(0)
        result = await loop.run_in_executor(executor, workflow.normalize_entities, state)
        state.update(result)
        update_step(job, 3, "completed", 65, "정규화 완료")
        await asyncio.sleep(0)
        
        # Step 5: Relationships
        update_step(job, 4, "processing", 70, "Neo4j에 관계 생성 중...")
        await asyncio.sleep(0)
        # Relationships are created in normalize_entities
        update_step(job, 4, "completed", 75, "관계 생성 완료")
        await asyncio.sleep(0)
        
        # Step 6: Generate Skills
        update_step(job, 5, "processing", 78, "Agent Skill 문서 생성 중...")
        await asyncio.sleep(0)
        result = await loop.run_in_executor(executor, workflow.generate_skills, state)
        state.update(result)
        update_step(job, 5, "completed", 82, "Agent Skill 문서 생성 완료")
        await asyncio.sleep(0)
        
        # Note: BPMN is now generated on-demand when viewing processes, not during ingestion
        
        # Step 7: Generate DMN
        update_step(job, 6, "processing", 88, "DMN 의사결정 테이블 생성 중...")
        await asyncio.sleep(0)
        result = await loop.run_in_executor(executor, workflow.generate_dmn, state)
        state.update(result)
        update_step(job, 6, "completed", 92, "DMN 생성 완료")
        await asyncio.sleep(0)
        
        # Step 8: Export
        update_step(job, 7, "processing", 95, "결과물 저장 중...")
        await asyncio.sleep(0)
        result = await loop.run_in_executor(executor, workflow.export_artifacts, state)
        state.update(result)
        update_step(job, 7, "completed", 100, "완료!")
        
        # Store result summary
        job["detail_message"] = "모든 처리가 완료되었습니다!"
        
        # Collect BPMN file information
        bpmn_files = state.get("bpmn_files", {})
        bpmn_paths = list(bpmn_files.values()) if bpmn_files else [str(Config.OUTPUT_DIR / "process.bpmn")]

        # NOTE:
        # - 현재 파이프라인은 BPMN을 "온디맨드(조회 시 생성)"로 전환되어 있어 bpmn_files가 비어있을 수 있습니다.
        # - 하지만 agent(워커)가 "이번 job에서 생성된 프로세스만" 정확히 식별하려면 process_id 목록이 필요합니다.
        processes_state = state.get("processes", []) or []
        process_ids: list[str] = []
        process_names: dict[str, str] = {}
        for p in processes_state:
            pid = None
            pname = None
            if isinstance(p, dict):
                pid = p.get("proc_id") or p.get("process_id") or p.get("id")
                pname = p.get("name")
            else:
                pid = getattr(p, "proc_id", None) or getattr(p, "process_id", None) or getattr(p, "id", None)
                pname = getattr(p, "name", None)
            if pid:
                process_ids.append(str(pid))
                if pname:
                    process_names[str(pid)] = str(pname)
        
        job["result"] = {
            "processes": len(state.get("processes", [])),
            "tasks": len(state.get("tasks", [])),
            "roles": len(state.get("roles", [])),
            "gateways": len(state.get("gateways", [])),
            "decisions": len(state.get("dmn_decisions", [])),
            "process_ids": process_ids,
            "process_names": process_names,
            "bpmn_path": bpmn_paths[0] if bpmn_paths else str(Config.OUTPUT_DIR / "process.bpmn"),  # Backward compatibility
            "bpmn_paths": bpmn_paths,  # All BPMN file paths
            "bpmn_files": bpmn_files,  # process_id -> file_path mapping
            "dmn_path": str(Config.OUTPUT_DIR / "decisions.dmn")
        }
        
        job["status"] = "completed"
        
        workflow.neo4j.close()
        
    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["detail_message"] = f"오류 발생: {str(e)}"
        job["current_step"] = "error"
        print(f"Error in background processing: {e}")
        traceback.print_exc()
    finally:
        executor.shutdown(wait=False)


def update_step(job: dict, step_index: int, status: str, progress: int, 
                detail_message: str = None, chunk_info: dict = None):
    """Update step status and overall progress with detailed info."""
    if step_index < len(job["steps"]):
        job["steps"][step_index]["status"] = status
    job["progress"] = progress
    job["current_step"] = job["steps"][step_index]["name"] if step_index < len(job["steps"]) else "completed"
    
    # Add detail message for frontend display
    if detail_message:
        job["detail_message"] = detail_message
    
    # Add chunk processing info
    if chunk_info:
        job["chunk_info"] = chunk_info
    elif status == "completed":
        # Clear chunk info when step is completed
        job["chunk_info"] = None


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job processing status."""
    if job_id not in job_progress:
        raise HTTPException(404, "Job not found")
    return job_progress[job_id]


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Stream job status updates via SSE."""
    if job_id not in job_progress:
        raise HTTPException(404, "Job not found")
    
    async def event_generator():
        last_data = ""
        while True:
            job = job_progress.get(job_id)
            if not job:
                break
            
            # Send update if any data changed (not just progress)
            current_data = json.dumps(job, ensure_ascii=False)
            if current_data != last_data:
                last_data = current_data
                yield f"data: {current_data}\n\n"
            
            if job["status"] in ["completed", "error"]:
                break
            
            await asyncio.sleep(0.2)  # Check more frequently for real-time updates
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ==================== Entity APIs ====================

@app.get("/api/processes")
async def get_processes():
    """Get all processes."""
    neo4j = Neo4jClient()
    try:
        # Verify Neo4j connection first
        if not neo4j.verify_connection():
            raise HTTPException(503, "Neo4j connection failed")
        
        with neo4j.session() as session:
            # First check if there are any processes
            count_result = session.run("MATCH (p:Process) RETURN count(p) as count")
            count_record = count_result.single()
            process_count = count_record["count"] if count_record else 0
            
            print(f"[API] Found {process_count} processes in Neo4j")
            
            if process_count == 0:
                return {"processes": []}
            
            # Simplified query - avoid potential NULL issues
            result = session.run("""
                MATCH (p:Process)
                OPTIONAL MATCH (p)-[:HAS_TASK]->(t:Task)
                WITH p, count(DISTINCT t) as taskCount
                RETURN p.proc_id as proc_id,
                       COALESCE(p.name, '') as name,
                       COALESCE(p.purpose, '') as purpose,
                       COALESCE(p.description, '') as description,
                       CASE WHEN p.triggers IS NULL THEN [] ELSE p.triggers END as triggers,
                       CASE WHEN p.outcomes IS NULL THEN [] ELSE p.outcomes END as outcomes,
                       COALESCE(taskCount, 0) as taskCount
                ORDER BY p.name
            """)
            processes = []
            for record in result:
                try:
                    proc = {
                        "proc_id": str(record["proc_id"]) if record["proc_id"] else "",
                        "name": str(record["name"]) if record["name"] else "",
                        "purpose": str(record["purpose"]) if record["purpose"] else "",
                        "description": str(record["description"]) if record["description"] else "",
                        "triggers": list(record["triggers"]) if record["triggers"] else [],
                        "outcomes": list(record["outcomes"]) if record["outcomes"] else [],
                        "taskCount": int(record["taskCount"]) if record["taskCount"] is not None else 0,
                        "evidence": None
                    }
                    # Validate required fields
                    if proc["proc_id"] and proc["name"]:
                        processes.append(proc)
                    else:
                        print(f"[API] Skipping invalid process: {proc}")
                except Exception as e:
                    print(f"[API] Error processing process record: {e}, record: {dict(record)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"[API] Returning {len(processes)} processes")
            return {"processes": processes}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error fetching processes: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error fetching processes: {str(e)}")
    finally:
        try:
            neo4j.close()
        except:
            pass


@app.get("/api/processes/{proc_id}")
async def get_process_detail(proc_id: str):
    """Get process with all related entities."""
    neo4j = Neo4jClient()
    try:
        data = neo4j.get_process_with_details(proc_id)
        if not data:
            raise HTTPException(404, "Process not found")
        return data
    finally:
        neo4j.close()


@app.get("/api/processes/{proc_id}/graph")
async def get_process_graph(
    proc_id: str,
    include_integrated: bool = False,
    tenant_id: str = "",
    task_id: str = "",
    graph_name: str = "",
):
    """
    Get the *actual extracted* Neo4j subgraph for a process.

    한 todo 의 데이터는 `g_<tenant>_<task_id>` 그래프에 누적되며 같은 그래프
    안에서 proc_id 로 필터링된다. tenant_id + task_id 또는 graph_name 으로
    그래프를 명시하면 가장 정확하다.
    """

    def _fetch(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        data = client.get_process_graph_elements(proc_id)
        if not data:
            data = client.get_latest_request_process_graph_by_proc_id(proc_id)
        if not data:
            return None
        if include_integrated:
            try:
                data["integrated_graph"] = client.get_latest_integrated_graph_by_proc_id(proc_id)
            except Exception:
                data["integrated_graph"] = None
        return data

    data, tried, hit_name = _query_across_graphs(
        _fetch, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
    )
    if not data:
        raise HTTPException(
            404,
            f"Process graph not found (proc_id={proc_id}, "
            f"tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
        if tenant_id:
            data.setdefault("tenant_id", tenant_id)
        if task_id:
            data.setdefault("task_id", task_id)
    return data


@app.get("/api/processes/{proc_id}/graph/integrated")
async def get_integrated_graph_by_process(
    proc_id: str,
    tenant_id: str = "",
    task_id: str = "",
    graph_name: str = "",
):
    """Get latest request-level integrated graph associated with this process."""

    def _fetch(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        return client.get_latest_integrated_graph_by_proc_id(proc_id)

    data, tried, hit_name = _query_across_graphs(
        _fetch, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
    )
    if not data:
        raise HTTPException(
            404,
            f"Integrated graph not found for process "
            f"(proc_id={proc_id}, tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
    return data


@app.get("/api/graph/runs/{run_id}")
async def get_request_integrated_graph(
    run_id: str,
    tenant_id: str = "",
    task_id: str = "",
    graph_name: str = "",
):
    """Get request-level integrated graph snapshot."""

    def _fetch(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        return client.get_request_integrated_graph(run_id)

    data, tried, hit_name = _query_across_graphs(
        _fetch, tenant_id=tenant_id, task_id=task_id, run_id=run_id, explicit=graph_name
    )
    if not data:
        raise HTTPException(
            404,
            f"Graph run not found (run_id={run_id}, tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
    return data


@app.get("/api/graph/runs/{run_id}/processes/{proc_id}")
async def get_request_process_graph(
    run_id: str,
    proc_id: str,
    tenant_id: str = "",
    task_id: str = "",
    graph_name: str = "",
):
    """Get per-process graph snapshot for a specific request run."""

    def _fetch(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        return client.get_request_process_graph(run_id, proc_id)

    data, tried, hit_name = _query_across_graphs(
        _fetch, tenant_id=tenant_id, task_id=task_id, run_id=run_id, explicit=graph_name
    )
    if not data:
        raise HTTPException(
            404,
            f"Process graph snapshot not found "
            f"(run_id={run_id}, proc_id={proc_id}, tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
    return data


@app.get("/api/graph/requests/{task_id}")
async def get_request_integrated_graph_by_task(
    task_id: str,
    tenant_id: str = "",
    graph_name: str = "",
    max_nodes: int = 3000,
    allow_global_fallback: bool = True,
):
    """
    Get latest request-level integrated graph snapshot by task_id.

    tenant_id 가 함께 주어지면 build_graph_name(tenant, task_id) 로 정확한
    그래프를 짚어 조회한다.

    이전 버전 워커가 적재한 그래프(GraphRun/GraphSnapshot 노드 없음) 인 경우,
    같은 그래프의 process-core 전역 노드/엣지를 통합 그래프로 fallback 반환.
    """

    def _fetch(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        return client.get_latest_request_integrated_graph_by_task(task_id)

    data, tried, hit_name = _query_across_graphs(
        _fetch, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
    )

    if not data and allow_global_fallback:
        def _fetch_global(client: Neo4jClient) -> Optional[Dict[str, Any]]:
            payload = client.get_full_graph_elements(max_nodes=max_nodes)
            if not payload:
                return None
            counts = (payload.get("counts") or {}) if isinstance(payload, dict) else {}
            if isinstance(counts, dict) and not (counts.get("nodes") or counts.get("edges")):
                return None
            if isinstance(payload, dict):
                payload.setdefault("source", "global_fallback")
            return payload

        data, fallback_tried, hit_name = _query_across_graphs(
            _fetch_global, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
        )
        merged: List[str] = list(tried)
        for g in fallback_tried:
            if g not in merged:
                merged.append(g)
        tried = merged

    if not data:
        raise HTTPException(
            404,
            f"Request integrated graph not found "
            f"(task_id={task_id}, tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
        if tenant_id:
            data.setdefault("tenant_id", tenant_id)
        data.setdefault("task_id", task_id)
    return data


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks with relationships."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            result = session.run("""
                MATCH (t:Task)
                OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
                OPTIONAL MATCH (p:Process)-[:HAS_TASK]->(t)
                OPTIONAL MATCH (t)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                OPTIONAL MATCH (t)-[:NEXT]->(next:Task)
                OPTIONAL MATCH (prev:Task)-[:NEXT]->(t)
                WITH t, r.name as role_name, p.name as process_name, p.proc_id as process_id,
                     collect(DISTINCT {page: c.page, text: left(c.text, 200)})[0] as evidence,
                     collect(DISTINCT next.name) as next_tasks,
                     collect(DISTINCT prev.name) as prev_tasks
                RETURN t {.*} as task, role_name, process_name, process_id, evidence, next_tasks, prev_tasks
                ORDER BY coalesce(t.task_order, 0)
            """)
            tasks = []
            for record in result:
                task = record["task"]
                task["role_name"] = record["role_name"]
                task["process_name"] = record["process_name"]
                task["process_id"] = record["process_id"]
                task["evidence"] = record["evidence"]
                task["next_tasks"] = [n for n in record["next_tasks"] if n]
                task["prev_tasks"] = [p for p in record["prev_tasks"] if p]
                tasks.append(task)
            return {"tasks": tasks}
    finally:
        neo4j.close()


@app.get("/api/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """Get task detail with evidence/source information."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            # Try to find by task_id or by BPMN element ID (Activity_xxx)
            result = session.run("""
                MATCH (t:Task)
                WHERE t.task_id = $task_id OR t.name CONTAINS $task_id OR t.task_id CONTAINS $task_id
                OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
                OPTIONAL MATCH (p:Process)-[:HAS_TASK]->(t)
                OPTIONAL MATCH (t)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                OPTIONAL MATCH (t)-[:NEXT]->(next:Task)
                OPTIONAL MATCH (prev:Task)-[:NEXT]->(t)
                RETURN t {.*} as task,
                       r {.name, .role_id, .description} as role,
                       p {.name, .proc_id, .description} as process,
                       collect(DISTINCT {
                           chunk_id: c.chunk_id,
                           page: c.page, 
                           text: c.text,
                           span: c.span
                       }) as evidences,
                       collect(DISTINCT {name: next.name, task_id: next.task_id}) as next_tasks,
                       collect(DISTINCT {name: prev.name, task_id: prev.task_id}) as prev_tasks
                LIMIT 1
            """, {"task_id": task_id})
            
            record = result.single()
            if not record:
                raise HTTPException(404, "Task not found")
            
            task = record["task"]
            task["role"] = record["role"]
            task["process"] = record["process"]
            task["evidences"] = [e for e in record["evidences"] if e.get("chunk_id")]
            task["next_tasks"] = [n for n in record["next_tasks"] if n.get("name")]
            task["prev_tasks"] = [p for p in record["prev_tasks"] if p.get("name")]
            
            return task
    finally:
        neo4j.close()


@app.get("/api/bpmn/element/{element_id}")
async def get_bpmn_element(element_id: str):
    """Get element info by BPMN element ID (e.g., Activity_xxx, Gateway_xxx)."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            # Search across different entity types
            # Try Task first
            result = session.run("""
                MATCH (t:Task)
                WHERE t.name CONTAINS $search_term OR t.task_id CONTAINS $search_term
                OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
                OPTIONAL MATCH (p:Process)-[:HAS_TASK]->(t)
                OPTIONAL MATCH (t)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                RETURN 'Task' as element_type,
                       t {.*} as element,
                       r {.name, .role_id, .description} as related_role,
                       p {.name, .proc_id} as related_process,
                       collect(DISTINCT {
                           page: c.page, 
                           text: c.text
                       }) as evidences
                LIMIT 1
            """, {"search_term": element_id.replace("Activity_", "").replace("_", " ")})
            
            record = result.single()
            
            if not record:
                # Try Gateway
                result = session.run("""
                    MATCH (g:Gateway)
                    WHERE g.name CONTAINS $search_term OR g.gateway_id CONTAINS $search_term
                    OPTIONAL MATCH (p:Process)-[:HAS_GATEWAY]->(g)
                    OPTIONAL MATCH (g)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                    RETURN 'Gateway' as element_type,
                           g {.*} as element,
                           null as related_role,
                           p {.name, .proc_id} as related_process,
                           collect(DISTINCT {page: c.page, text: c.text}) as evidences
                    LIMIT 1
                """, {"search_term": element_id.replace("Gateway_", "").replace("_", " ")})
                record = result.single()
            
            if not record:
                # Try Event
                result = session.run("""
                    MATCH (e:Event)
                    WHERE e.name CONTAINS $search_term OR e.event_id CONTAINS $search_term
                    OPTIONAL MATCH (p:Process)-[:HAS_EVENT]->(e)
                    OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                    RETURN 'Event' as element_type,
                           e {.*} as element,
                           null as related_role,
                           p {.name, .proc_id} as related_process,
                           collect(DISTINCT {page: c.page, text: c.text}) as evidences
                    LIMIT 1
                """, {"search_term": element_id.replace("Event_", "").replace("StartEvent_", "").replace("EndEvent_", "").replace("_", " ")})
                record = result.single()
            
            if not record:
                return {"found": False, "element_id": element_id}
            
            return {
                "found": True,
                "element_type": record["element_type"],
                "element": record["element"],
                "role": record["related_role"],
                "process": record["related_process"],
                "evidences": [e for e in record["evidences"] if e.get("page")]
            }
    finally:
        neo4j.close()


@app.get("/api/roles")
async def get_roles():
    """Get all roles."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            result = session.run("""
                MATCH (r:Role)
                OPTIONAL MATCH (t:Task)-[:PERFORMED_BY]->(r)
                OPTIONAL MATCH (r)-[:MAKES_DECISION]->(d:DMNDecision)
                OPTIONAL MATCH (r)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                WITH r, count(DISTINCT t) as taskCount, count(DISTINCT d) as decisionCount, 
                     collect(DISTINCT {page: c.page, text: left(c.text, 200)})[0] as evidence
                RETURN r {.*, taskCount: taskCount, decisionCount: decisionCount} as role, evidence
                ORDER BY r.name
            """)
            roles = []
            for record in result:
                role = record["role"]
                role["evidence"] = record["evidence"]
                roles.append(role)
            return {"roles": roles}
    finally:
        neo4j.close()


@app.get("/api/decisions")
async def get_decisions():
    """Get all DMN decisions."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            result = session.run("""
                MATCH (d:DMNDecision)
                OPTIONAL MATCH (d)-[:HAS_RULE]->(rule:DMNRule)
                OPTIONAL MATCH (r:Role)-[:MAKES_DECISION]->(d)
                OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(c:ReferenceChunk)
                WITH d, count(DISTINCT rule) as ruleCount, collect(DISTINCT r.name) as roles,
                     collect(DISTINCT {page: c.page, text: left(c.text, 200)})[0] as evidence
                RETURN d {.*, ruleCount: ruleCount} as decision, roles, evidence
                ORDER BY d.name
            """)
            decisions = []
            for record in result:
                dec = record["decision"]
                dec["roles"] = [r for r in record["roles"] if r]
                dec["evidence"] = record["evidence"]
                decisions.append(dec)
            return {"decisions": decisions}
    finally:
        neo4j.close()


@app.get("/api/sequence-flows")
async def get_sequence_flows():
    """Get all sequence flows (NEXT relationships)."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            result = session.run("""
                MATCH (t1:Task)-[r:NEXT]->(t2:Task)
                OPTIONAL MATCH (p:Process)-[:HAS_TASK]->(t1)
                RETURN t1.name as from_task,
                       t1.task_id as from_task_id,
                       t2.name as to_task,
                       t2.task_id as to_task_id,
                       r.condition as condition,
                       p.name as process_name
                ORDER BY p.name, t1.order
            """)
            flows = []
            for record in result:
                flows.append({
                    "from_task": record["from_task"],
                    "from_task_id": record["from_task_id"],
                    "to_task": record["to_task"],
                    "to_task_id": record["to_task_id"],
                    "condition": record["condition"],
                    "process_name": record["process_name"]
                })
            return {"flows": flows}
    finally:
        neo4j.close()


@app.get("/api/graph-stats")
async def get_graph_stats():
    """Get graph statistics."""
    neo4j = Neo4jClient()
    try:
        with neo4j.session() as session:
            stats = {}
            
            queries = {
                "processes": "MATCH (n:Process) RETURN count(n) as count",
                "tasks": "MATCH (n:Task) RETURN count(n) as count",
                "roles": "MATCH (n:Role) RETURN count(n) as count",
                "gateways": "MATCH (n:Gateway) RETURN count(n) as count",
                "events": "MATCH (n:Event) RETURN count(n) as count",
                "decisions": "MATCH (n:DMNDecision) RETURN count(n) as count",
                "rules": "MATCH (n:DMNRule) RETURN count(n) as count",
                "skills": "MATCH (n:Skill) RETURN count(n) as count",
                "documents": "MATCH (n:Document) RETURN count(n) as count",
                "chunks": "MATCH (n:ReferenceChunk) RETURN count(n) as count",
            }
            
            for key, query in queries.items():
                result = session.run(query)
                stats[key] = result.single()["count"]
            
            # Relationship counts
            rel_queries = {
                "has_task": "MATCH ()-[r:HAS_TASK]->() RETURN count(r) as count",
                "performed_by": "MATCH ()-[r:PERFORMED_BY]->() RETURN count(r) as count",
                "next": "MATCH ()-[r:NEXT]->() RETURN count(r) as count",
                "supported_by": "MATCH ()-[r:SUPPORTED_BY]->() RETURN count(r) as count",
                "makes_decision": "MATCH ()-[r:MAKES_DECISION]->() RETURN count(r) as count",
            }
            
            stats["relationships"] = {}
            for key, query in rel_queries.items():
                result = session.run(query)
                stats["relationships"][key] = result.single()["count"]
            
            return stats
    finally:
        neo4j.close()


@app.get("/api/graph/full")
async def get_full_graph(
    tenant_id: str = "",
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
    proc_id: str = "",
    source: str = "integrated",
    max_nodes: int = 3000,
    graph_name: str = "",
):
    """
    Unified full graph endpoint.

    - tenant_id + task_id (권장): `g_<tenant>_<task_id>` 그래프를 정확히 식별.
    - graph_name: 그래프 이름 직접 지정.
    - proc_id 가 함께 들어오면 → 같은 그래프 안에서 해당 프로세스만 추출.

    source:
      - integrated (default): 요청 단위 통합 스냅샷 (run_id -> task_id -> latest)
      - global: 그래프의 process-core 전역 노드/엣지
    """
    src = (source or "integrated").strip().lower()
    task_id_val = task_id or ""
    run_id_val = run_id or ""

    # 1) proc_id 가 있으면 → 해당 프로세스 부분 그래프 (같은 그래프에서 필터)
    if proc_id:
        def _fetch_proc(client: Neo4jClient) -> Optional[Dict[str, Any]]:
            data = client.get_process_graph_elements(proc_id)
            if not data:
                data = client.get_latest_request_process_graph_by_proc_id(proc_id)
            return data or None

        data, tried, hit_name = _query_across_graphs(
            _fetch_proc,
            tenant_id=tenant_id,
            task_id=task_id_val,
            run_id=run_id_val,
            explicit=graph_name,
        )
        if not data:
            raise HTTPException(
                404,
                f"Process graph not found (proc_id={proc_id}, "
                f"tried graphs: {', '.join(tried) or '<none>'})",
            )
        if isinstance(data, dict) and hit_name:
            data.setdefault("graph_name", hit_name)
        return data

    # 2) source=global
    if src == "global":
        def _fetch_global(client: Neo4jClient) -> Optional[Dict[str, Any]]:
            data = client.get_full_graph_elements(max_nodes=max_nodes)
            if not data:
                return None
            if isinstance(data, dict):
                counts = data.get("counts") or {}
                if isinstance(counts, dict) and not (counts.get("nodes") or counts.get("edges")):
                    return None
            return data

        data, tried, hit_name = _query_across_graphs(
            _fetch_global,
            tenant_id=tenant_id,
            task_id=task_id_val,
            run_id=run_id_val,
            explicit=graph_name,
        )
        if not data:
            raise HTTPException(
                404,
                f"Global graph not found (tried graphs: {', '.join(tried) or '<none>'})",
            )
        if isinstance(data, dict) and hit_name:
            data.setdefault("graph_name", hit_name)
        return data

    # 3) source=integrated (default)
    def _fetch_integrated(client: Neo4jClient) -> Optional[Dict[str, Any]]:
        if run_id_val:
            return client.get_request_integrated_graph(run_id_val)
        if task_id_val:
            return client.get_latest_request_integrated_graph_by_task(task_id_val)
        return client.get_latest_request_integrated_graph()

    data, tried, hit_name = _query_across_graphs(
        _fetch_integrated,
        tenant_id=tenant_id,
        task_id=task_id_val,
        run_id=run_id_val,
        explicit=graph_name,
    )
    if not data:
        raise HTTPException(
            404,
            f"Integrated graph not found (tried graphs: {', '.join(tried) or '<none>'})",
        )
    if isinstance(data, dict) and hit_name:
        data.setdefault("graph_name", hit_name)
        if tenant_id:
            data.setdefault("tenant_id", tenant_id)
        if task_id_val:
            data.setdefault("task_id", task_id_val)
    return data


# ==================== File APIs ====================

@app.get("/api/files/bpmn")
async def get_bpmn_file(process_id: Optional[str] = None):
    """Get the generated BPMN file - dynamically generated from Neo4j.
    
    Args:
        process_id: Optional process ID to get a specific process BPMN.
                   If not provided, returns the first process BPMN.
    """
    from ..generators.bpmn_generator import BPMNGenerator
    from fastapi.responses import Response
    
    neo4j = Neo4jClient()
    try:
        # Get process ID if not provided
        if not process_id:
            all_processes = neo4j.get_all_processes()
            if not all_processes:
                raise HTTPException(404, "No processes found in database")
            process_id = all_processes[0]["proc_id"]
        
        # Get all entities for the process from Neo4j
        entities = neo4j.get_process_entities_for_bpmn(process_id)
        if not entities:
            raise HTTPException(404, f"Process {process_id} not found")
        
        # Get sequence flows
        sequence_flows = neo4j.get_sequence_flows(process_id)
        
        # Generate BPMN XML
        bpmn_generator = BPMNGenerator()
        bpmn_xml = bpmn_generator.generate(
            process=entities["process"],
            tasks=entities["tasks"],
            roles=entities["roles"],
            gateways=entities["gateways"],
            events=entities["events"],
            task_role_map=entities["task_role_map"],
            neo4j_sequence_flows=sequence_flows
        )
        
        # Create filename
        safe_name = entities["process"].name.replace(" ", "_").replace("/", "_")[:50]
        filename = f"process_{safe_name}_{process_id[:8]}.bpmn"
        
        return Response(
            content=bpmn_xml,
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    finally:
        neo4j.close()


@app.get("/api/files/bpmn/list")
async def list_bpmn_files():
    """List all processes that can generate BPMN (from Neo4j)."""
    neo4j = Neo4jClient()
    try:
        processes = neo4j.get_all_processes()
        files = []
        for proc in processes:
            safe_name = proc["name"].replace(" ", "_").replace("/", "_")[:50]
            filename = f"process_{safe_name}_{proc['proc_id'][:8]}.bpmn"
            files.append({
                "filename": filename,
                "process_id": proc["proc_id"],
                "process_name": proc["name"],
                "is_default": False
            })
        
        # Mark first as default
        if files:
            files[0]["is_default"] = True
        
        return {"files": files, "count": len(files)}
    finally:
        neo4j.close()


@app.get("/api/files/bpmn/content")
async def get_bpmn_content(process_id: Optional[str] = None):
    """Get BPMN file content as text.
    
    Args:
        process_id: Optional process ID to get a specific process BPMN.
                   If not provided, returns the default (first) BPMN content.
    """
    from ..generators.bpmn_generator import BPMNGenerator
    
    neo4j = Neo4jClient()
    try:
        # Get process ID if not provided
        if not process_id:
            all_processes = neo4j.get_all_processes()
            if not all_processes:
                raise HTTPException(404, "No processes found in database")
            process_id = all_processes[0]["proc_id"]
        
        # Get all entities for the process from Neo4j
        entities = neo4j.get_process_entities_for_bpmn(process_id)
        if not entities:
            raise HTTPException(404, f"Process {process_id} not found")
        
        # Get sequence flows
        sequence_flows = neo4j.get_sequence_flows(process_id)
        
        # Generate BPMN XML
        bpmn_generator = BPMNGenerator()
        bpmn_xml = bpmn_generator.generate(
            process=entities["process"],
            tasks=entities["tasks"],
            roles=entities["roles"],
            gateways=entities["gateways"],
            events=entities["events"],
            task_role_map=entities["task_role_map"],
            neo4j_sequence_flows=sequence_flows
        )
        
        return {
            "content": bpmn_xml,
            "process_id": process_id,
            "process_name": entities["process"].name
        }
    finally:
        neo4j.close()


@app.get("/api/files/bpmn/all")
async def get_all_bpmn_contents():
    """Get all BPMN file contents - dynamically generated from Neo4j."""
    from ..generators.bpmn_generator import BPMNGenerator
    
    neo4j = Neo4jClient()
    try:
        processes = neo4j.get_all_processes()
        results = {}
        bpmn_generator = BPMNGenerator()
        
        for proc in processes:
            proc_id = proc["proc_id"]
            
            # Get all entities for the process
            entities = neo4j.get_process_entities_for_bpmn(proc_id)
            if not entities:
                continue
            
            # Get sequence flows
            sequence_flows = neo4j.get_sequence_flows(proc_id)
            
            # Generate BPMN XML
            bpmn_xml = bpmn_generator.generate(
                process=entities["process"],
                tasks=entities["tasks"],
                roles=entities["roles"],
                gateways=entities["gateways"],
                events=entities["events"],
                task_role_map=entities["task_role_map"],
                neo4j_sequence_flows=sequence_flows
            )
            
            safe_name = proc["name"].replace(" ", "_").replace("/", "_")[:50]
            filename = f"process_{safe_name}_{proc_id[:8]}.bpmn"
            
            results[proc_id] = {
                "content": bpmn_xml,
                "filename": filename,
                "process_name": proc["name"]
            }
        
        return {"bpmn_files": results, "count": len(results)}
    finally:
        neo4j.close()


@app.get("/api/files/dmn")
async def get_dmn_file():
    """Get the generated DMN file."""
    dmn_path = Config.OUTPUT_DIR / "decisions.dmn"
    if not dmn_path.exists():
        raise HTTPException(404, "DMN file not found")
    return FileResponse(dmn_path, media_type="application/xml", filename="decisions.dmn")


@app.get("/api/files/pdf/{filename}")
async def get_pdf_file(filename: str):
    """Get uploaded PDF file."""
    pdf_path = Config.UPLOAD_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(404, "PDF file not found")
    return FileResponse(pdf_path, media_type="application/pdf")


def run_server():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_server()

