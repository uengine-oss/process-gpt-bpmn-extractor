#!/usr/bin/env python3
"""
PDF2BPMN Agent Server
ProcessGPT SDK를 사용한 PDF to BPMN 변환 에이전트 서버
"""

import asyncio
import os
import re
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

# Optional: embedded API server (Neo4j graph gateway for frontend)
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    import uvicorn  # type: ignore
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    env_file = current_dir / '.env'

    if env_file.exists():
        load_dotenv(env_file)
        print(f"[OK] Loaded env from: {env_file}")
    else:
        print("[WARN] .env file not found.")
except ImportError:
    print("[WARN] python-dotenv not installed. Using system env vars.")

# ProcessGPT SDK imports
try:
    from processgpt_agent_sdk import ProcessGPTAgentServer
    PROCESSGPT_SDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ProcessGPT SDK not available: {e}")
    print("pip install processgpt-agent-sdk 로 설치하세요.")
    PROCESSGPT_SDK_AVAILABLE = False

# 로컬 모듈 imports
from pdf2bpmn_agent_executor import PDF2BPMNAgentExecutor
try:
    from src.pdf2bpmn.graph.neo4j_client import Neo4jClient  # type: ignore
except Exception:
    Neo4jClient = None

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDF2BPMNServerConfig:
    """PDF2BPMN 에이전트 서버 설정"""
    
    def __init__(self):
        # ProcessGPT 설정
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.service_role_key = os.getenv("SERVICE_ROLE_KEY")
        self.polling_interval = int(os.getenv("POLLING_INTERVAL", "5"))
        self.agent_orch = os.getenv("AGENT_ORCH", "pdf2bpmn")
        
        # PDF2BPMN 서버 설정
        self.pdf2bpmn_url = os.getenv("PDF2BPMN_URL", "http://localhost:8001")
        self.task_timeout = int(os.getenv("TASK_TIMEOUT", "3600"))  # 1시간

        # Embedded Graph API (for frontend)
        self.graph_api_enabled = os.getenv("GRAPH_API_ENABLED", "true").lower() in ("1", "true", "yes", "y", "on")
        self.graph_api_host = os.getenv("GRAPH_API_HOST", "0.0.0.0")
        self.graph_api_port = int(os.getenv("GRAPH_API_PORT", "8012"))
        
        # 환경 검증
        self.validate()
    
    def validate(self):
        """설정 검증"""
        missing_vars = []
        
        if not self.supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not self.supabase_anon_key:
            missing_vars.append("SUPABASE_ANON_KEY")
            
        if missing_vars:
            raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return {
            "supabase_url": self.supabase_url,
            "polling_interval": self.polling_interval,
            "agent_orch": self.agent_orch,
            "pdf2bpmn_url": self.pdf2bpmn_url,
            "task_timeout": self.task_timeout,
            "graph_api_enabled": self.graph_api_enabled,
            "graph_api_host": self.graph_api_host,
            "graph_api_port": self.graph_api_port,
        }


class PDF2BPMNServerManager:
    """PDF2BPMN 에이전트 서버 관리자"""
    
    def __init__(self, config: PDF2BPMNServerConfig):
        self.config = config
        self.server: ProcessGPTAgentServer = None
        self.executor: PDF2BPMNAgentExecutor = None
        self.is_running = False
        self._graph_api_task: asyncio.Task | None = None
        self._graph_api_server: Any = None
        
        # 신호 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """신호 핸들러 - 우아한 종료"""
        logger.info(f"신호 {signum} 수신 - 서버 종료 중...")
        if self.server:
            self.server.stop()
        self.is_running = False
    
    async def initialize(self):
        """서버 초기화"""
        logger.info("PDF2BPMN 에이전트 서버 초기화 중...")
        
        # AgentExecutor 설정
        executor_config = {
            "pdf2bpmn_url": self.config.pdf2bpmn_url,
            "timeout": self.config.task_timeout,
            "supabase_url": self.config.supabase_url,
            "supabase_key": self.config.service_role_key
        }
        
        # AgentExecutor 생성
        self.executor = PDF2BPMNAgentExecutor(config=executor_config)
        logger.info(f"PDF2BPMNAgentExecutor 생성됨 (Server: {self.config.pdf2bpmn_url})")
        
        if not PROCESSGPT_SDK_AVAILABLE:
            logger.error("ProcessGPT SDK가 설치되지 않았습니다.")
            return False
        
        # ProcessGPT 서버 생성
        try:
            self.server = ProcessGPTAgentServer(
                agent_executor=self.executor,
                agent_type=self.config.agent_orch
            )
            self.server.polling_interval = self.config.polling_interval
            logger.info(f"ProcessGPT 서버 생성됨 - 에이전트: {self.config.agent_orch}")
            return True
            
        except Exception as e:
            logger.error(f"ProcessGPT 서버 생성 실패: {e}")
            return False

    @staticmethod
    def _list_age_graphs() -> List[str]:
        """List all AGE graphs in the database (best-effort)."""
        if Neo4jClient is None:
            return []
        probe = Neo4jClient()
        try:
            conn = probe._connect()  # noqa: SLF001 - embedded API fallback
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM ag_catalog.ag_graph ORDER BY name")
                rows = cur.fetchall() or []
            return [
                str(r[0]).strip()
                for r in rows
                if isinstance(r, (list, tuple)) and r and str(r[0]).strip()
            ]
        except Exception as e:
            logger.debug(f"[GRAPH] _list_age_graphs failed: {e}")
            return []
        finally:
            try:
                probe.close()
            except Exception:
                pass

    @classmethod
    def _candidate_graph_names(
        cls,
        *,
        tenant_id: str = "",
        task_id: str = "",
        explicit: str = "",
    ) -> List[str]:
        """
        Return candidate AGE graph names ordered by priority.

        한 todo(=task_id) 의 모든 프로세스 데이터는 `g_<tenant>_<task_id>`
        그래프 하나에 누적된다. 따라서 가장 정확한 식별 방법은 tenant_id +
        task_id 로 `Neo4jClient.build_graph_name()` 을 호출하는 것.

        우선순위:
          1. explicit (graph_name 직접 지정)
          2. tenant_id + task_id 로 build_graph_name 결과
          3. AGE catalog 에서 task_id 토큰을 포함하는 그래프 (legacy fallback)
          4. 기본 그래프 (Config.AGE_GRAPH_NAME) (legacy)
          5. 그 외 발견된 모든 그래프 (최후 수단)
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

        if Neo4jClient is not None and tenant_id and task_id:
            try:
                _add(Neo4jClient.build_graph_name(tenant_id=tenant_id, todo_id=task_id))
            except Exception:
                pass

        discovered = cls._list_age_graphs()

        if task_id:
            task_token = re.sub(r"[^0-9A-Za-z_]", "_", str(task_id)).strip("_").lower()
            if task_token:
                for g in discovered:
                    if task_token in g.lower():
                        _add(g)

        if Neo4jClient is not None:
            try:
                _add(Neo4jClient().graph_name)
            except Exception:
                pass

        for g in discovered:
            _add(g)

        return candidates

    @staticmethod
    def _try_with_graph(
        name: str,
        fn: Callable[["Neo4jClient"], Any],
    ) -> Optional[Any]:
        """Run callable against the given graph; absorb missing-graph errors."""
        if not name or Neo4jClient is None:
            return None
        client = Neo4jClient(graph_name=name)
        try:
            return fn(client)
        except Exception as e:
            logger.debug(f"[GRAPH] graph='{name}' query failed: {e}")
            return None
        finally:
            try:
                client.close()
            except Exception:
                pass

    @classmethod
    def _query_across_graphs(
        cls,
        fn: Callable[["Neo4jClient"], Any],
        *,
        tenant_id: str = "",
        task_id: str = "",
        explicit: str = "",
    ) -> tuple[Optional[Any], List[str], Optional[str]]:
        """Iterate candidate graphs; return (data, tried_graphs, hit_graph_name)."""
        candidates = cls._candidate_graph_names(
            tenant_id=tenant_id, task_id=task_id, explicit=explicit
        )
        for name in candidates:
            hit = cls._try_with_graph(name, fn)
            if hit:
                return hit, candidates, name
        return None, candidates, None

    def _build_graph_api_app(self) -> FastAPI:
        """
        Embedded Graph API.

        한 todo(=task_id) 의 모든 프로세스 데이터는 `g_<tenant>_<task_id>`
        그래프 하나에 누적 저장된다. 프론트는 다음 두 가지 호출만 알면 된다:

          - 전체 그래프(해당 todo 의 모든 프로세스 통합):
              GET /api/graph/full?tenant_id=...&task_id=...
            또는
              GET /api/graph/full?graph_name=g_...

          - 프로세스별 그래프(해당 todo 의 특정 프로세스만):
              GET /api/processes/{proc_id}/graph?tenant_id=...&task_id=...
            또는
              GET /api/processes/{proc_id}/graph?graph_name=g_...

        하위호환 엔드포인트도 유지:
          - GET /api/graph/requests/{task_id}?tenant_id=...
          - GET /api/graph/full?run_id=...&source=integrated|global
        """
        app = FastAPI(title="PDF2BPMN Embedded Graph API", version="0.2.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        @app.get("/api/health")
        async def health():
            if Neo4jClient is None:
                return {"status": "ok", "neo4j": "unavailable"}
            neo4j = Neo4jClient()
            try:
                ok = neo4j.verify_connection()
            finally:
                try:
                    neo4j.close()
                except Exception:
                    pass
            return {"status": "ok", "neo4j": "connected" if ok else "disconnected"}

        @app.get("/api")
        async def api_root():
            return {
                "status": "ok",
                "name": "PDF2BPMN Embedded Graph API",
                "endpoints": [
                    "/api/health",
                    "/api/processes/{proc_id}/graph",
                    "/api/graph/requests/{task_id}",
                    "/api/graph/full",
                ],
                "usage": {
                    "full_per_todo": "/api/graph/full?tenant_id=<tenant>&task_id=<todo>",
                    "process_per_todo": "/api/processes/{proc_id}/graph?tenant_id=<tenant>&task_id=<todo>",
                },
            }

        # -------------------------------------------------------------
        # 프로세스별 그래프 (해당 todo 안의 특정 proc_id)
        # -------------------------------------------------------------
        @app.get("/api/processes/{proc_id}/graph")
        async def process_graph(
            proc_id: str,
            tenant_id: str = "",
            task_id: str = "",
            graph_name: str = "",
        ):
            if Neo4jClient is None:
                raise HTTPException(500, "Neo4jClient is not available in this runtime")

            def _fetch(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                data = client.get_process_graph_elements(proc_id)
                if not data:
                    data = client.get_latest_request_process_graph_by_proc_id(proc_id)
                return data or None

            data, tried, hit_name = self._query_across_graphs(
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

        # -------------------------------------------------------------
        # 통합 그래프 (해당 todo 의 모든 프로세스 누적)
        # -------------------------------------------------------------
        @app.get("/api/graph/requests/{task_id}")
        async def request_integrated_graph(
            task_id: str,
            tenant_id: str = "",
            graph_name: str = "",
            max_nodes: int = 3000,
            allow_global_fallback: bool = True,
        ):
            if Neo4jClient is None:
                raise HTTPException(500, "Neo4jClient is not available in this runtime")

            def _fetch(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                return client.get_latest_request_integrated_graph_by_task(task_id)

            data, tried, hit_name = self._query_across_graphs(
                _fetch, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
            )

            # FALLBACK: GraphRun/GraphSnapshot 노드가 없는 (이전 버전 워커가
            # 적재한) 그래프인 경우, 같은 그래프의 process-core 전역 노드/엣지를
            # "통합 그래프" 로 간주해서 반환한다. 이렇게 하면 스냅샷 적재 전에
            # 만든 todo 의 그래프도 프론트에서 그대로 시각화 가능.
            if (
                not data
                and allow_global_fallback
                and Neo4jClient is not None
            ):
                def _fetch_global(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                    payload = client.get_full_graph_elements(max_nodes=max_nodes)
                    if not payload:
                        return None
                    counts = (payload.get("counts") or {}) if isinstance(payload, dict) else {}
                    if isinstance(counts, dict) and not (counts.get("nodes") or counts.get("edges")):
                        return None
                    if isinstance(payload, dict):
                        payload.setdefault("source", "global_fallback")
                    return payload

                data, fallback_tried, hit_name = self._query_across_graphs(
                    _fetch_global, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
                )
                # tried 목록 합치기 (중복 제거)
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

        # -------------------------------------------------------------
        # 통합 엔드포인트
        #   - source=integrated (default): 통합 스냅샷
        #   - source=global: 그래프의 process-core 전체 노드/엣지
        # -------------------------------------------------------------
        @app.get("/api/graph/full")
        async def full_graph(
            tenant_id: str = "",
            task_id: str = "",
            run_id: str = "",
            proc_id: str = "",
            source: str = "integrated",
            max_nodes: int = 3000,
            graph_name: str = "",
        ):
            if Neo4jClient is None:
                raise HTTPException(500, "Neo4jClient is not available in this runtime")

            src = (source or "integrated").strip().lower()

            # 1) proc_id 가 함께 들어오면 → 프로세스별 그래프로 위임 (같은 그래프에서 필터)
            if proc_id:
                def _fetch_proc(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                    data = client.get_process_graph_elements(proc_id)
                    if not data:
                        data = client.get_latest_request_process_graph_by_proc_id(proc_id)
                    return data or None

                data, tried, hit_name = self._query_across_graphs(
                    _fetch_proc, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
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

            # 2) source=global: 그래프 전역 (process-core 라벨)
            if src == "global":
                def _fetch_global(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                    data = client.get_full_graph_elements(max_nodes=max_nodes)
                    if not data:
                        return None
                    if isinstance(data, dict):
                        counts = data.get("counts") or {}
                        if isinstance(counts, dict) and not (
                            counts.get("nodes") or counts.get("edges")
                        ):
                            return None
                    return data

                data, tried, hit_name = self._query_across_graphs(
                    _fetch_global, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
                )
                if not data:
                    raise HTTPException(
                        404,
                        f"Global graph not found (tried graphs: {', '.join(tried) or '<none>'})",
                    )
                if isinstance(data, dict) and hit_name:
                    data.setdefault("graph_name", hit_name)
                return data

            # 3) source=integrated (default): 통합 그래프 스냅샷
            def _fetch_integrated(client: "Neo4jClient") -> Optional[Dict[str, Any]]:
                if run_id:
                    return client.get_request_integrated_graph(run_id)
                if task_id:
                    return client.get_latest_request_integrated_graph_by_task(task_id)
                return client.get_latest_request_integrated_graph()

            data, tried, hit_name = self._query_across_graphs(
                _fetch_integrated, tenant_id=tenant_id, task_id=task_id, explicit=graph_name
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
                if task_id:
                    data.setdefault("task_id", task_id)
            return data

        return app

    async def _start_embedded_graph_api(self):
        if not self.config.graph_api_enabled:
            return
        if not FASTAPI_AVAILABLE:
            logger.warning("[WARN] GRAPH_API_ENABLED=true but fastapi/uvicorn not available. Skipping embedded API.")
            return
        if Neo4jClient is None:
            logger.warning("[WARN] GRAPH_API_ENABLED=true but Neo4jClient import failed. Skipping embedded API.")
            return

        try:
            app = self._build_graph_api_app()
            cfg = uvicorn.Config(app, host=self.config.graph_api_host, port=self.config.graph_api_port, log_level="info")
            server = uvicorn.Server(cfg)
            self._graph_api_server = server
            logger.info(f"[OK] Embedded Graph API starting on http://{self.config.graph_api_host}:{self.config.graph_api_port}/api")
            await server.serve()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning(f"[WARN] Embedded Graph API failed to start: {e}")

    async def _stop_embedded_graph_api(self):
        try:
            if self._graph_api_server is not None:
                try:
                    self._graph_api_server.should_exit = True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._graph_api_task is not None and not self._graph_api_task.done():
                self._graph_api_task.cancel()
        except Exception:
            pass
    
    async def start(self):
        """서버 시작"""
        if not await self.initialize():
            logger.error("서버 초기화 실패")
            return False
        
        self.is_running = True
        
        print()
        print("=" * 70)
        print("[*] PDF2BPMN Agent Server")
        print("=" * 70)
        print(f"[>] Start Time: {datetime.now().isoformat()}")
        print(f"[>] Agent Type: {self.config.agent_orch}")
        print(f"[>] Polling Interval: {self.config.polling_interval}s")
        print(f"[>] PDF2BPMN Server: {self.config.pdf2bpmn_url}")
        print(f"[>] Task Timeout: {self.config.task_timeout}s")
        if self.config.graph_api_enabled:
            print(f"[>] Embedded Graph API: http://localhost:{self.config.graph_api_port}/api  (GRAPH_API_PORT)")
        print()
        print("[*] Supported Tasks:")
        print("  - PDF to BPMN conversion")
        print("  - Multi-process extraction from single PDF")
        print("  - Auto-save to proc_def and proc_map")
        print("  - Real-time progress events")
        print()
        print("[*] Query Example:")
        print('  \'{"pdf_url": "https://xxx.supabase.co/storage/.../file.pdf"}\'')
        print('  \'[InputData] pdf_file_url: https://xxx/file.pdf, pdf_file_name: manual.pdf\'')
        print()
        print("[!] Press Ctrl+C to stop the server")
        print("=" * 70)
        print()
        
        try:
            # Start embedded graph API in background (same process)
            if self.config.graph_api_enabled:
                self._graph_api_task = asyncio.create_task(self._start_embedded_graph_api())

            # ProcessGPT 서버 실행 (무한 폴링 루프)
            await self.server.run()
            
        except KeyboardInterrupt:
            logger.info("사용자가 서버 중지를 요청했습니다")
        except Exception as e:
            logger.error(f"서버 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
    
    async def stop(self):
        """서버 중지"""
        logger.info("서버 종료 중...")
        self.is_running = False

        await self._stop_embedded_graph_api()
        
        if self.server:
            try:
                self.server.stop()
                logger.info("ProcessGPT 서버 중지됨")
            except Exception as e:
                logger.warning(f"서버 중지 중 오류: {e}")
        
        print("\n[OK] PDF2BPMN Agent Server stopped gracefully")


def print_usage():
    """사용법 출력"""
    print()
    print("=" * 60)
    print("PDF2BPMN Agent Server - PDF to BPMN 변환 에이전트")
    print("=" * 60)
    print()
    print("필수 환경변수:")
    print("  SUPABASE_URL          - Supabase 프로젝트 URL")
    print("  SUPABASE_ANON_KEY     - Supabase 익명 키")
    print()
    print("선택적 환경변수:")
    print("  SERVICE_ROLE_KEY      - Supabase 서비스 역할 키 (DB 저장용)")
    print("  PDF2BPMN_URL=http://localhost:8001 - PDF2BPMN 서버 URL")
    print("  POLLING_INTERVAL=5    - 폴링 간격 (초)")
    print("  AGENT_ORCH=pdf2bpmn   - 에이전트 타입")
    print("  TASK_TIMEOUT=3600     - 태스크 타임아웃 (초, 기본 1시간)")
    print()
    print("실행 예시:")
    print("  export SUPABASE_URL='https://your-project.supabase.co'")
    print("  export SUPABASE_ANON_KEY='your-anon-key'")
    print("  python pdf2bpmn_agent_server.py")
    print()
    print("또는 .env 파일에 환경변수를 설정하세요.")
    print()


async def main():
    """메인 실행 함수"""
    try:
        # 설정 로드 및 검증
        config = PDF2BPMNServerConfig()
        
        # 서버 관리자 생성 및 시작
        server_manager = PDF2BPMNServerManager(config)
        await server_manager.start()
        
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        print_usage()
        sys.exit(1)
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Python 버전 체크
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher required")
        sys.exit(1)
    
    # 필수 패키지 체크
    missing_packages = []
    
    if not PROCESSGPT_SDK_AVAILABLE:
        missing_packages.append("processgpt-agent-sdk")
    
    try:
        import httpx
    except ImportError:
        missing_packages.append("httpx")
    
    try:
        from supabase import create_client
    except ImportError:
        missing_packages.append("supabase")
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        sys.exit(1)
    
    # 메인 실행
    asyncio.run(main())
