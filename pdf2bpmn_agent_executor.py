#!/usr/bin/env python3
"""
PDF2BPMN AgentExecutor for ProcessGPT SDK
ProcessGPT SDK의 AgentExecutor 인터페이스를 구현한 PDF2BPMN 에이전트
PDF를 분석하여 BPMN XML을 생성하고, 진행 상황을 실시간으로 이벤트로 전송
"""

import asyncio
import os
import logging
import uuid
import json
import re
import httpx
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import traceback
import xml.etree.ElementTree as ET

# Supabase imports
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase-py not available. Install with: pip install supabase")

# ProcessGPT SDK imports
try:
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.types import TaskStatusUpdateEvent, TaskState, TaskArtifactUpdateEvent
    from a2a.utils import new_agent_text_message, new_text_artifact
    PROCESSGPT_SDK_AVAILABLE = True
except ImportError:
    # Fallback classes for when SDK is not available
    class AgentExecutor:
        async def execute(self, context, event_queue): pass
        async def cancel(self, context, event_queue): pass
    
    class RequestContext:
        def get_user_input(self): return ""
        def get_context_data(self): return {}
    
    class EventQueue:
        def enqueue_event(self, event): pass
    
    class TaskStatusUpdateEvent:
        def __init__(self, **kwargs): pass
    
    class TaskState:
        working = "working"
        input_required = "input_required"
    
    class TaskArtifactUpdateEvent:
        def __init__(self, **kwargs): pass
    
    def new_agent_text_message(text, context_id, task_id): return text
    def new_text_artifact(name, description, text): return {"name": name, "description": description, "text": text}
    
    PROCESSGPT_SDK_AVAILABLE = False
    print("Warning: ProcessGPT SDK not available. Using fallback classes.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDF2BPMNAgentExecutor(AgentExecutor):
    """
    ProcessGPT SDK와 호환되는 PDF2BPMN AgentExecutor
    PDF 파일을 분석하여 BPMN XML을 생성하는 에이전트
    
    지원 기능:
    - PDF URL 다운로드 및 분석
    - 다중 프로세스 BPMN 생성
    - 실시간 진행 상황 이벤트 발송
    - proc_def, configuration(proc_map) 저장
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        PDF2BPMN AgentExecutor 초기화
        
        Args:
            config: 설정 딕셔너리
                - pdf2bpmn_url: PDF2BPMN 서버 URL (기본: http://localhost:8001)
                - timeout: API 호출 타임아웃 (초)
                - supabase_url: Supabase URL
                - supabase_key: Supabase 서비스 키
        """
        self.config = config or {}
        self.is_cancelled = False
        
        # PDF2BPMN 서버 설정
        self.pdf2bpmn_url = os.getenv('PDF2BPMN_URL', self.config.get('pdf2bpmn_url', 'http://localhost:8001'))
        self.timeout = self.config.get('timeout', 3600)  # 1시간 타임아웃
        
        # Supabase 설정
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SERVICE_ROLE_KEY')
        self.supabase_client: Optional[Client] = None
        
        # HTTP 클라이언트
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Supabase 초기화
        self._setup_supabase()
        
        logger.info(f"[OK] PDF2BPMNAgentExecutor initialized")
        logger.info(f"    - PDF2BPMN Server: {self.pdf2bpmn_url}")
        logger.info(f"    - Timeout: {self.timeout}s")

    def _setup_supabase(self):
        """Supabase 클라이언트 초기화"""
        if not SUPABASE_AVAILABLE:
            logger.warning("[WARN] Supabase library not installed.")
            return
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("[WARN] Supabase URL or key not configured.")
            return
        
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"[OK] Supabase client initialized")
        except Exception as e:
            logger.error(f"[ERROR] Supabase client init failed: {e}")
            self.supabase_client = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 반환 (lazy initialization)"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
        return self.http_client

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Query에서 PDF URL과 요청 정보를 파싱
        
        예시 입력:
        1. 순수 JSON: '{"pdf_url": "https://...", "description": "..."}'
        2. [InputData] JSON 형식:
           [InputData]
           {"path": "...", "fullPath": "http://...", "publicUrl": "http://...", "originalFileName": "..."}
        """
        result = {
            "pdf_url": "",
            "pdf_name": "",
            "description": "",
            "raw_query": query
        }
        
        # 1. 순수 JSON 형식 파싱 시도
        try:
            if query.strip().startswith('{'):
                data = json.loads(query)
                result["pdf_url"] = data.get("pdf_url", data.get("fileUrl", data.get("pdf_file_url", 
                                    data.get("fullPath", data.get("publicUrl", "")))))
                result["pdf_name"] = data.get("pdf_name", data.get("fileName", data.get("pdf_file_name",
                                    data.get("originalFileName", ""))))
                result["description"] = data.get("description", "")
                return result
        except json.JSONDecodeError:
            pass
        
        # 2. [InputData] 형식에서 JSON 추출
        if "[InputData]" in query:
            # [InputData] 다음의 JSON 객체 찾기
            input_data_match = re.search(r'\[InputData\]\s*(\{[^}]+\})', query, re.DOTALL)
            if input_data_match:
                try:
                    json_str = input_data_match.group(1)
                    data = json.loads(json_str)
                    # fullPath, publicUrl, path 순으로 URL 추출
                    result["pdf_url"] = data.get("fullPath", data.get("publicUrl", data.get("path", "")))
                    result["pdf_name"] = data.get("originalFileName", data.get("fileName", ""))
                    logger.info(f"[PARSE] Extracted from [InputData] JSON - URL: {result['pdf_url']}, Name: {result['pdf_name']}")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"[PARSE] Failed to parse [InputData] JSON: {e}")
            
            # key: value 형식 fallback
            url_match = re.search(r'pdf_file_url[:\s]+([^\s,]+)', query)
            if url_match:
                result["pdf_url"] = url_match.group(1).strip()
            
            name_match = re.search(r'pdf_file_name[:\s]+([^\s,]+)', query)
            if name_match:
                result["pdf_name"] = name_match.group(1).strip()
        
        # 3. URL 직접 추출 시도 (fallback)
        if not result["pdf_url"]:
            # .pdf로 끝나는 URL 또는 storage URL 찾기
            url_match = re.search(r'https?://[^\s<>"\'}\]]+(?:\.pdf|/storage/[^\s<>"\'}\]]+)', query, re.IGNORECASE)
            if url_match:
                result["pdf_url"] = url_match.group(0).rstrip('",')
                logger.info(f"[PARSE] Extracted URL via regex: {result['pdf_url']}")
        
        # 4. 파일명 추출 (URL에서)
        if result["pdf_url"] and not result["pdf_name"]:
            # URL에서 파일명 추출
            from urllib.parse import urlparse, unquote
            parsed = urlparse(result["pdf_url"])
            path_parts = parsed.path.split('/')
            if path_parts:
                result["pdf_name"] = unquote(path_parts[-1])
        
        return result

    async def _download_pdf(self, url: str, filename: str = None) -> str:
        """PDF 파일 다운로드 후 임시 파일 경로 반환"""
        client = await self._get_http_client()
        
        logger.info(f"[DOWNLOAD] Downloading PDF from: {url}")
        
        response = await client.get(url, follow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF: {response.status_code}")
        
        # 임시 파일 생성
        suffix = ".pdf"
        if filename:
            # 파일명에서 확장자 추출
            name_part = Path(filename).stem
            suffix = f"_{name_part}.pdf"
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"[DOWNLOAD] PDF saved to: {temp_file.name}")
        return temp_file.name

    async def _send_progress_event(
        self, 
        event_queue: EventQueue, 
        context_id: str, 
        task_id: str,
        job_id: str,
        message: str,
        status: str,
        progress: int = 0,
        extra_data: Dict = None
    ):
        """진행 상황 이벤트 발송"""
        event_data = {
            "message": message,
            "status": status,
            "progress": progress,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if extra_data:
            event_data.update(extra_data)
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(event_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=False,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": status,
                    "job_id": job_id,
                    "progress": progress
                }
            )
        )

    async def _send_bpmn_artifact(
        self,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        process_id: str,
        process_name: str,
        bpmn_xml: str,
        is_last: bool = False
    ):
        """BPMN XML 아티팩트 이벤트 발송"""
        artifact_data = {
            "type": "bpmn",
            "process_id": process_id,
            "process_name": process_name,
            "bpmn_xml": bpmn_xml,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                artifact=new_text_artifact(
                    name=f"BPMN: {process_name}",
                    description=f"Generated BPMN XML for process: {process_name}",
                    text=json.dumps(artifact_data, ensure_ascii=False),
                ),
                lastChunk=is_last,
                contextId=context_id,
                taskId=task_id,
            )
        )

    def _convert_xml_to_json(self, bpmn_xml: str) -> Dict[str, Any]:
        """
        BPMN XML을 ProcessGPT JSON 형식으로 변환
        ProcessDefinitionModule.vue의 convertXMLToJSON과 유사한 로직
        """
        try:
            root = ET.fromstring(bpmn_xml)
            
            # 네임스페이스 처리
            namespaces = {
                'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
                'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI'
            }
            
            result = {
                "processDefinitionId": "",
                "processDefinitionName": "",
                "version": "1.0",
                "shortDescription": "",
                "description": "",
                "data": [],
                "roles": [],
                "events": [],
                "activities": [],
                "gateways": [],
                "sequences": [],
                "subProcesses": []
            }
            
            # Process 정보 추출
            process = root.find('.//bpmn:process', namespaces)
            if process is None:
                # 네임스페이스 없는 경우
                process = root.find('.//process')
            
            if process is not None:
                result["processDefinitionId"] = process.get('id', '')
                result["processDefinitionName"] = process.get('name', '')
            
            # Participants에서 이름 추출 시도
            collaboration = root.find('.//bpmn:collaboration', namespaces)
            if collaboration is None:
                collaboration = root.find('.//collaboration')
            
            if collaboration is not None:
                participant = collaboration.find('.//bpmn:participant', namespaces)
                if participant is None:
                    participant = collaboration.find('.//participant')
                if participant is not None:
                    result["processDefinitionName"] = participant.get('name', result["processDefinitionName"])
            
            # Lanes (Roles) 추출
            lanes = root.findall('.//bpmn:lane', namespaces)
            if not lanes:
                lanes = root.findall('.//lane')
            
            for lane in lanes:
                role = {
                    "name": lane.get('name', ''),
                    "endpoint": "",
                    "resolutionRule": "",
                    "default": ""
                }
                result["roles"].append(role)
            
            # Tasks (Activities) 추출
            task_types = ['userTask', 'serviceTask', 'task', 'manualTask', 'scriptTask']
            for task_type in task_types:
                tasks = root.findall(f'.//bpmn:{task_type}', namespaces)
                if not tasks:
                    tasks = root.findall(f'.//{task_type}')
                
                for task in tasks:
                    activity = {
                        "id": task.get('id', ''),
                        "name": task.get('name', ''),
                        "type": task_type,
                        "description": "",
                        "instruction": "",
                        "role": "",
                        "tool": "formHandler:defaultform",
                        "duration": 5
                    }
                    result["activities"].append(activity)
            
            # Events 추출
            for event_type in ['startEvent', 'endEvent', 'intermediateThrowEvent', 'intermediateCatchEvent']:
                events = root.findall(f'.//bpmn:{event_type}', namespaces)
                if not events:
                    events = root.findall(f'.//{event_type}')
                
                for event in events:
                    evt = {
                        "id": event.get('id', ''),
                        "name": event.get('name', ''),
                        "type": event_type,
                        "role": "",
                        "process": result["processDefinitionId"]
                    }
                    result["events"].append(evt)
            
            # Gateways 추출
            for gateway_type in ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway']:
                gateways = root.findall(f'.//bpmn:{gateway_type}', namespaces)
                if not gateways:
                    gateways = root.findall(f'.//{gateway_type}')
                
                for gateway in gateways:
                    gw = {
                        "id": gateway.get('id', ''),
                        "name": gateway.get('name', ''),
                        "type": gateway_type,
                        "condition": ""
                    }
                    result["gateways"].append(gw)
            
            # Sequence Flows 추출
            sequences = root.findall('.//bpmn:sequenceFlow', namespaces)
            if not sequences:
                sequences = root.findall('.//sequenceFlow')
            
            for seq in sequences:
                sequence = {
                    "id": seq.get('id', ''),
                    "name": seq.get('name', ''),
                    "source": seq.get('sourceRef', ''),
                    "target": seq.get('targetRef', ''),
                    "condition": ""
                }
                result["sequences"].append(sequence)
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] XML to JSON conversion failed: {e}")
            return {
                "processDefinitionId": f"process_{uuid.uuid4().hex[:8]}",
                "processDefinitionName": "Converted Process",
                "data": [],
                "roles": [],
                "events": [],
                "activities": [],
                "gateways": [],
                "sequences": []
            }

    async def _save_proc_def(self, proc_def: Dict, tenant_id: str) -> bool:
        """프로세스 정의를 proc_def 테이블에 저장"""
        if not self.supabase_client:
            logger.error("[ERROR] Supabase client is None! Cannot save proc_def")
            return False
        
        try:
            logger.info(f"[DB-PROC_DEF] ========== START ==========")
            logger.info(f"[DB-PROC_DEF] id={proc_def['id']}, tenant_id={tenant_id}")
            logger.info(f"[DB-PROC_DEF] name={proc_def.get('name')}, bpmn_length={len(proc_def.get('bpmn', ''))}")
            logger.info(f"[DB-PROC_DEF] definition keys: {list(proc_def.get('definition', {}).keys()) if proc_def.get('definition') else 'None'}")
            
            # 기존 proc_def 확인
            logger.info(f"[DB-PROC_DEF] Checking existing...")
            existing = self.supabase_client.table('proc_def').select('id, uuid').eq('id', proc_def['id']).execute()
            logger.info(f"[DB-PROC_DEF] Existing result: {existing.data}")
            
            if existing.data and len(existing.data) > 0:
                existing_uuid = existing.data[0].get('uuid')
                logger.info(f"[DB-PROC_DEF] Updating existing uuid={existing_uuid}")
                result = self.supabase_client.table('proc_def').update({
                    'name': proc_def['name'],
                    'definition': proc_def['definition'],
                    'bpmn': proc_def['bpmn'],
                    'type': proc_def.get('type', 'bpmn'),
                    'isdeleted': False,
                    'tenant_id': tenant_id
                }).eq('uuid', existing_uuid).execute()
                logger.info(f"[DB-PROC_DEF] Update result: {result.data}")
            else:
                insert_data = {
                    'id': proc_def['id'],
                    'name': proc_def['name'],
                    'definition': proc_def['definition'],
                    'bpmn': proc_def['bpmn'],
                    'tenant_id': tenant_id,
                    'type': proc_def.get('type', 'bpmn'),
                    'isdeleted': False
                }
                
                logger.info(f"[DB-PROC_DEF] Inserting new record...")
                logger.info(f"[DB-PROC_DEF] Insert data keys: {list(insert_data.keys())}")
                result = self.supabase_client.table('proc_def').insert(insert_data).execute()
                logger.info(f"[DB-PROC_DEF] Insert result: {result.data}")
            
            logger.info(f"[DB-PROC_DEF] ========== SUCCESS ==========")
            return True
            
        except Exception as e:
            logger.error(f"[DB-PROC_DEF] ========== ERROR ==========")
            logger.error(f"[DB-PROC_DEF] Exception type: {type(e).__name__}")
            logger.error(f"[DB-PROC_DEF] Exception message: {e}")
            import traceback
            logger.error(f"[DB-PROC_DEF] Traceback:\n{traceback.format_exc()}")
            return False

    async def _update_proc_map(self, new_process: Dict, tenant_id: str) -> bool:
        """
        configuration 테이블의 proc_map 업데이트
        미분류 카테고리에 새 프로세스 추가
        """
        if not self.supabase_client:
            logger.warning("[WARN] Supabase client not available, skipping proc_map update")
            return False
        
        try:
            # 기존 proc_map 조회
            result = self.supabase_client.table('configuration').select('value').eq('key', 'proc_map').eq('tenant_id', tenant_id).execute()
            
            if result.data and len(result.data) > 0:
                proc_map = result.data[0].get('value', {})
            else:
                # proc_map이 없으면 새로 생성
                proc_map = {"mega_proc_list": []}
            
            if not isinstance(proc_map, dict):
                proc_map = {"mega_proc_list": []}
            
            mega_proc_list = proc_map.get('mega_proc_list', [])
            
            # 미분류 메가 프로세스 찾기
            unclassified_mega = None
            for mega in mega_proc_list:
                if mega.get('id') == 'unclassified' or mega.get('name') == '미분류':
                    unclassified_mega = mega
                    break
            
            if not unclassified_mega:
                # 미분류 메가 프로세스 생성
                unclassified_mega = {
                    "id": "unclassified",
                    "name": "미분류",
                    "major_proc_list": []
                }
                mega_proc_list.append(unclassified_mega)
            
            # 미분류 메이저 프로세스 찾기
            major_proc_list = unclassified_mega.get('major_proc_list', [])
            unclassified_major = None
            for major in major_proc_list:
                if major.get('id') == 'unclassified_major' or major.get('name') == '미분류':
                    unclassified_major = major
                    break
            
            if not unclassified_major:
                # 미분류 메이저 프로세스 생성
                unclassified_major = {
                    "id": "unclassified_major",
                    "name": "미분류",
                    "sub_proc_list": []
                }
                major_proc_list.append(unclassified_major)
                unclassified_mega['major_proc_list'] = major_proc_list
            
            # 서브 프로세스 목록에 추가 (중복 체크)
            sub_proc_list = unclassified_major.get('sub_proc_list', [])
            exists = any(p.get('id') == new_process['id'] for p in sub_proc_list)
            
            if not exists:
                sub_proc_list.append({
                    "id": new_process['id'],
                    "name": new_process['name'],
                    "path": new_process['id'],
                    "new": True
                })
                unclassified_major['sub_proc_list'] = sub_proc_list
            
            proc_map['mega_proc_list'] = mega_proc_list
            
            # configuration 테이블 업데이트
            if result.data and len(result.data) > 0:
                self.supabase_client.table('configuration').update({
                    'value': proc_map
                }).eq('key', 'proc_map').eq('tenant_id', tenant_id).execute()
            else:
                self.supabase_client.table('configuration').insert({
                    'key': 'proc_map',
                    'value': proc_map,
                    'tenant_id': tenant_id
                }).execute()
            
            logger.info(f"[DB] Updated proc_map with process: {new_process['id']}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update proc_map: {e}")
            return False

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        메인 실행 로직 - ProcessGPT SDK 인터페이스 구현
        
        Args:
            context: 요청 컨텍스트 (사용자 입력, 컨텍스트 데이터 포함)
            event_queue: 이벤트 큐 (진행 상황 및 결과 전송용)
        """
        # 1. 작업 정보 가져오기
        logger.info(f"[DEBUG] context: {context}")
        
        context_data = context.get_context_data()
        row = context_data.get("row", {})
        logger.info(f"[DEBUG] row: {row}")
        logger.info(f"[DEBUG] context_data keys: {context_data.keys()}")
        
        task_id = row.get("id")
        # context_id가 None이면 task_id를 사용 (adhoc task의 경우)
        context_id = row.get("root_proc_inst_id") or row.get("proc_inst_id") or task_id
        tenant_id = row.get("tenant_id", "uengine")
        
        # Query 가져오기 - 여러 소스에서 시도
        user_input = context.get_user_input()
        logger.info(f"[DEBUG] context.get_user_input(): '{user_input[:200] if user_input else 'None'}...'")
        
        # context_data에서 query 확인
        if not user_input and context_data.get('query'):
            user_input = context_data.get('query')
            logger.info(f"[INFO] Got user_input from context_data.query: '{user_input[:100]}...'")
        
        # row에서 query 확인
        if not user_input and row.get('query'):
            user_input = row.get('query')
            logger.info(f"[INFO] Got user_input from row.query: '{user_input[:100]}...'")
        
        # description fallback
        if not user_input and row.get('description'):
            user_input = row.get('description')
            logger.info(f"[INFO] Got user_input from description: '{user_input[:100]}...'")
        
        # Job ID 생성
        job_id = f"pdf2bpmn-{task_id}"
        
        logger.info(f"[START] PDF2BPMN task: {user_input[:100] if user_input else 'N/A'}... (job_id: {job_id})")
        
        temp_pdf_path = None
        
        try:
            # 2. 작업 시작 이벤트
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[START] PDF2BPMN 변환 작업을 시작합니다...",
                "task_started", 0
            )
            
            # 3. Query 파싱 (PDF 정보 추출)
            parsed = self._parse_query(user_input or "")
            pdf_name = parsed.get("pdf_name", "document.pdf")
            logger.info(f"[INFO] PDF Name: {pdf_name}")
            
            # # 4~7. PDF 다운로드/업로드/처리 (주석처리 - 프론트에서 이미 처리됨)
            # pdf_url = parsed.get("pdf_url", "")
            # if not pdf_url:
            #     raise Exception("PDF URL이 제공되지 않았습니다.")
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     f"[DOWNLOAD] PDF 파일 다운로드 중: {pdf_name}", "tool_usage_started", 5)
            # temp_pdf_path = await self._download_pdf(pdf_url, pdf_name)
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     "[UPLOAD] PDF 파일을 분석 서버에 업로드 중...", "tool_usage_started", 10)
            # client = await self._get_http_client()
            # with open(temp_pdf_path, 'rb') as f:
            #     files = {'file': (pdf_name, f, 'application/pdf')}
            #     upload_response = await client.post(f"{self.pdf2bpmn_url}/api/upload", files=files)
            # if upload_response.status_code != 200:
            #     raise Exception(f"PDF 업로드 실패: {upload_response.status_code}")
            # upload_result = upload_response.json()
            # processing_job_id = upload_result.get("job_id")
            # await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #     "[PROCESSING] PDF 분석 및 BPMN 변환을 시작합니다...", "tool_usage_started", 15)
            # process_response = await client.post(f"{self.pdf2bpmn_url}/api/process/{processing_job_id}")
            # if process_response.status_code != 200:
            #     raise Exception(f"처리 시작 실패: {process_response.status_code}")
            # max_retries = 600
            # retry_count = 0
            # last_progress = 15
            # while retry_count < max_retries:
            #     if self.is_cancelled:
            #         raise Exception("작업이 취소되었습니다.")
            #     status_response = await client.get(f"{self.pdf2bpmn_url}/api/jobs/{processing_job_id}")
            #     if status_response.status_code != 200:
            #         raise Exception(f"상태 조회 실패: {status_response.status_code}")
            #     job_status = status_response.json()
            #     current_status = job_status.get("status", "")
            #     current_progress = job_status.get("progress", 0)
            #     detail_message = job_status.get("detail_message", "")
            #     chunk_info = job_status.get("chunk_info")
            #     if retry_count % 5 == 0:
            #         logger.info(f"[POLL] status={current_status}, progress={current_progress}")
            #     if current_status == "completed":
            #         logger.info("[INFO] Processing completed")
            #         break
            #     elif current_status == "error":
            #         error_msg = job_status.get("error", "알 수 없는 오류")
            #         raise Exception(f"처리 중 오류 발생: {error_msg}")
            #     mapped_progress = 15 + int(current_progress * 0.7)
            #     if current_progress != last_progress:
            #         extra_data = {}
            #         if chunk_info:
            #             extra_data["chunk_info"] = chunk_info
            #         await self._send_progress_event(event_queue, context_id, task_id, job_id,
            #             f"[PROCESSING] {detail_message or f'진행 중... ({current_progress}%)'}", 
            #             "tool_usage_started", mapped_progress, extra_data)
            #         last_progress = current_progress
            #     await asyncio.sleep(1)
            #     retry_count += 1
            # if retry_count >= max_retries:
            #     raise Exception("처리 시간 초과")
            
            client = await self._get_http_client()
            
            # =================================================================
            # 4. PDF 처리 전: 기존 프로세스 목록 저장 (todo_id별 구분을 위해)
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[CHECK] 기존 프로세스 목록을 확인합니다...",
                "tool_usage_started", 5
            )
            
            existing_process_ids = set()
            try:
                list_response = await client.get(f"{self.pdf2bpmn_url}/api/files/bpmn/list")
                if list_response.status_code == 200:
                    list_result = list_response.json()
                    existing_process_ids = {p.get("process_id") for p in list_result.get("files", [])}
                    logger.info(f"[INFO] 기존 프로세스 {len(existing_process_ids)}개 확인: {existing_process_ids}")
            except Exception as e:
                logger.warning(f"[WARN] 기존 프로세스 목록 조회 실패: {e}")
            
            # =================================================================
            # 5. PDF URL로 처리 시작
            # =================================================================
            pdf_url = parsed.get("pdf_url", "")
            if not pdf_url:
                raise Exception("PDF URL이 제공되지 않았습니다. query에 pdf_url을 포함해주세요.")
            
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                f"[UPLOAD] PDF 파일을 분석 서버에 업로드 중: {pdf_name}",
                "tool_usage_started", 10
            )
            
            # PDF 다운로드
            temp_pdf_path = await self._download_pdf(pdf_url, pdf_name)
            
            # PDF2BPMN API에 업로드
            with open(temp_pdf_path, 'rb') as f:
                files = {'file': (pdf_name, f, 'application/pdf')}
                upload_response = await client.post(f"{self.pdf2bpmn_url}/api/upload", files=files)
            
            if upload_response.status_code != 200:
                raise Exception(f"PDF 업로드 실패: {upload_response.status_code} - {upload_response.text}")
            
            upload_result = upload_response.json()
            processing_job_id = upload_result.get("job_id")
            logger.info(f"[INFO] PDF 업로드 완료, job_id: {processing_job_id}")
            
            # =================================================================
            # 6. PDF 처리 시작 및 진행 상황 폴링
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[PROCESSING] PDF 분석 및 BPMN 변환을 시작합니다...",
                "tool_usage_started", 15
            )
            
            process_response = await client.post(f"{self.pdf2bpmn_url}/api/process/{processing_job_id}")
            if process_response.status_code != 200:
                raise Exception(f"처리 시작 실패: {process_response.status_code}")
            
            # 진행 상황 폴링 (self.timeout 사용 - config에서 전달받은 값, 기본 1시간)
            max_retries = self.timeout  # 1초 간격으로 폴링
            retry_count = 0
            last_progress = 15
            logger.info(f"[INFO] PDF 처리 폴링 시작 (timeout: {self.timeout}초)")
            
            while retry_count < max_retries:
                if self.is_cancelled:
                    raise Exception("작업이 취소되었습니다.")
                
                status_response = await client.get(f"{self.pdf2bpmn_url}/api/jobs/{processing_job_id}")
                if status_response.status_code != 200:
                    raise Exception(f"상태 조회 실패: {status_response.status_code}")
                
                job_status = status_response.json()
                current_status = job_status.get("status", "")
                current_progress = job_status.get("progress", 0)
                detail_message = job_status.get("detail_message", "")
                
                if retry_count % 10 == 0:  # 10초마다 로그
                    logger.info(f"[POLL] status={current_status}, progress={current_progress}")
                
                if current_status == "completed":
                    logger.info("[INFO] Processing completed")
                    break
                elif current_status == "error":
                    error_msg = job_status.get("error", "알 수 없는 오류")
                    raise Exception(f"처리 중 오류 발생: {error_msg}")
                
                # 진행률 이벤트 (변경 시에만)
                mapped_progress = 15 + int(current_progress * 0.7)  # 15% ~ 85%
                if current_progress != last_progress:
                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[PROCESSING] {detail_message or f'진행 중... ({current_progress}%)'}",
                        "tool_usage_started", mapped_progress
                    )
                    last_progress = current_progress
                
                await asyncio.sleep(1)
                retry_count += 1
            
            if retry_count >= max_retries:
                raise Exception("처리 시간 초과")
            
            # =================================================================
            # 7. 결과 가져오기 - 이 작업에서 생성된 BPMN만 필터링
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[GENERATING] BPMN XML 파일들을 생성합니다...",
                "tool_usage_started", 88
            )
            
            # 모든 BPMN 내용 조회
            all_bpmn_response = await client.get(f"{self.pdf2bpmn_url}/api/files/bpmn/all")
            if all_bpmn_response.status_code != 200:
                raise Exception(f"BPMN 파일 조회 실패: {all_bpmn_response.status_code}")
            
            all_bpmn_result = all_bpmn_response.json()
            all_bpmn_files = all_bpmn_result.get("bpmn_files", {})
            
            # **핵심: 기존에 없던 새 프로세스만 필터링**
            bpmn_files = {}
            for proc_id, bpmn_data in all_bpmn_files.items():
                if proc_id not in existing_process_ids:
                    bpmn_files[proc_id] = bpmn_data
                    logger.info(f"[NEW] 새 프로세스 발견: {proc_id}")
                else:
                    logger.info(f"[SKIP] 기존 프로세스 스킵: {proc_id}")
            
            bpmn_count = len(bpmn_files)
            logger.info(f"[INFO] 이 작업에서 생성된 BPMN: {bpmn_count}개 (전체: {len(all_bpmn_files)}개, 기존: {len(existing_process_ids)}개)")
            
            # 9. 각 BPMN에 대해 이벤트 발송 및 DB 저장
            saved_processes = []
            all_bpmn_xmls = {}  # proc_def_id -> bpmn_xml 매핑
            total_bpmn = len(bpmn_files)
            
            logger.info(f"[DEBUG] bpmn_files keys: {list(bpmn_files.keys())}")
            
            for idx, (proc_id, bpmn_data) in enumerate(bpmn_files.items()):
                bpmn_xml = bpmn_data.get("content", "")
                process_name = bpmn_data.get("process_name", f"Process {idx + 1}")
                
                logger.info(f"[DEBUG] Processing BPMN {idx+1}/{total_bpmn}: {process_name}")
                logger.info(f"[DEBUG] BPMN XML length: {len(bpmn_xml)} chars")
                
                if not bpmn_xml:
                    logger.warning(f"[WARN] Empty BPMN XML for process: {process_name}")
                    continue
                
                # XML을 JSON으로 변환
                proc_json = self._convert_xml_to_json(bpmn_xml)
                proc_json["processDefinitionName"] = process_name
                
                # proc_def_id 생성 (안전한 형식)
                safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', process_name.lower())[:50]
                proc_def_id = f"{safe_id}_{proc_id[:8]}"
                
                proc_json["processDefinitionId"] = proc_def_id
                
                # BPMN XML 저장
                all_bpmn_xmls[proc_def_id] = bpmn_xml
                
                # DB에 저장
                proc_def_data = {
                    "id": proc_def_id,
                    "name": process_name,
                    "definition": proc_json,
                    "bpmn": bpmn_xml,
                    "uuid": str(uuid.uuid4()),
                    "type": "bpmn",
                    "owner": None,
                    "prod_version": None
                }
                
                # proc_def 테이블 저장
                save_result = await self._save_proc_def(proc_def_data, tenant_id)
                logger.info(f"[DEBUG] proc_def save result: {save_result}")
                
                # proc_map 업데이트
                await self._update_proc_map({"id": proc_def_id, "name": process_name}, tenant_id)
                
                # saved_processes에 bpmn_xml 포함
                saved_processes.append({
                    "id": proc_def_id,
                    "name": process_name,
                    "bpmn_xml": bpmn_xml  # XML 내용 포함
                })
                
                # 진행 이벤트 (XML 포함)
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    f"[SAVED] 프로세스 저장 완료: {process_name}",
                    "tool_usage_finished", 90 + int(10 * (idx + 1) / total_bpmn),
                    {
                        "process_id": proc_def_id, 
                        "process_name": process_name,
                        "bpmn_xml": bpmn_xml  # 이벤트에도 XML 포함
                    }
                )
            
            # 10. 최종 결과 구성 (saved_processes에는 이미 bpmn_xml 포함됨)
            actual_count = len(saved_processes)
            logger.info(f"[DEBUG] Actual saved process count: {actual_count}")
            
            final_result = {
                "message": f"[COMPLETED] PDF2BPMN 변환 완료: {actual_count}개의 프로세스가 생성되었습니다.",
                "status": "completed",
                "job_id": job_id,
                "pdf_name": pdf_name,
                "process_count": actual_count,
                "saved_processes": saved_processes,  # bpmn_xml 포함
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # 11. 최종 결과 아티팩트 이벤트 (browser_use와 동일한 패턴)
            # 이 이벤트가 프론트엔드에서 최종 결과로 사용됨
            # saved_processes에서 요약 정보만 추출 (draft 크기 제한 고려)
            saved_processes_summary = [
                {"id": p["id"], "name": p["name"]} for p in saved_processes
            ]
            
            final_artifact_data = {
                "type": "pdf2bpmn_result",
                "pdf_name": pdf_name,
                "process_count": actual_count,
                "saved_processes": saved_processes_summary,  # 요약만
                "bpmn_xmls": all_bpmn_xmls,  # 모든 XML 내용
                "success": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "task_type": "pdf2bpmn"
            }
            
            event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="PDF2BPMN Result",
                        description=f"PDF2BPMN 변환 결과: {bpmn_count}개 프로세스 생성",
                        text=json.dumps(final_artifact_data, ensure_ascii=False),
                    ),
                    lastChunk=True,  # 최종 결과 표시
                    contextId=context_id,
                    taskId=task_id,
                )
            )
            
            # 12. 완료 상태 이벤트
            event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status={
                        "state": TaskState.working,
                        "message": new_agent_text_message(
                            json.dumps(final_result, ensure_ascii=False),
                            context_id, task_id
                        ),
                    },
                    final=True,
                    contextId=context_id,
                    taskId=task_id,
                    metadata={
                        "crew_type": "pdf2bpmn",
                        "event_type": "task_completed",
                        "job_id": job_id,
                        "process_count": bpmn_count
                    }
                )
            )
            
            logger.info(f"[DONE] Task completed: {job_id} ({bpmn_count} processes)")
            
        except httpx.ConnectError as e:
            logger.error(f"[ERROR] Cannot connect to PDF2BPMN server: {e}")
            error_msg = f"PDF2BPMN 서버에 연결할 수 없습니다: {self.pdf2bpmn_url}. 서버가 실행 중인지 확인하세요."
            await self._send_error_event(event_queue, context_id, task_id, job_id, error_msg, "connection_error")
            
        except Exception as e:
            logger.error(f"[ERROR] Task execution error: {e}")
            logger.error(traceback.format_exc())
            await self._send_error_event(event_queue, context_id, task_id, job_id, str(e), type(e).__name__)
        
        finally:
            # 임시 파일 정리
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                    logger.info(f"[CLEANUP] Removed temp file: {temp_pdf_path}")
                except Exception as e:
                    logger.warning(f"[WARN] Failed to remove temp file: {e}")
            
            # HTTP 클라이언트 정리
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None

    async def _send_error_event(
        self, 
        event_queue: EventQueue, 
        context_id: str, 
        task_id: str, 
        job_id: str, 
        error_msg: str, 
        error_type: str
    ):
        """에러 이벤트 발송"""
        error_data = {
            "message": f"[ERROR] PDF2BPMN 작업 실패: {error_msg}",
            "error": error_msg,
            "error_type": error_type,
            "status": "failed",
            "job_id": job_id,
            "pdf2bpmn_url": self.pdf2bpmn_url
        }
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(error_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=True,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": "error",
                    "job_id": job_id
                }
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 처리"""
        self.is_cancelled = True
        
        row = context.get_context_data().get("row", {})
        context_id = row.get("root_proc_inst_id") or row.get("proc_inst_id")
        task_id = row.get("id")
        
        cancel_data = {
            "message": "[CANCELLED] PDF2BPMN 작업이 취소되었습니다.",
            "status": "cancelled"
        }
        
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status={
                    "state": TaskState.working,
                    "message": new_agent_text_message(
                        json.dumps(cancel_data, ensure_ascii=False),
                        context_id, task_id
                    ),
                },
                final=True,
                contextId=context_id,
                taskId=task_id,
                metadata={
                    "crew_type": "pdf2bpmn",
                    "event_type": "task_cancelled"
                }
            )
        )
        
        # HTTP 클라이언트 정리
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        logger.info("[CANCELLED] PDF2BPMN task cancelled")
