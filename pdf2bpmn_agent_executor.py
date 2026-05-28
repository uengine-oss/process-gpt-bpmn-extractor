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
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Set, Tuple
import traceback
import xml.etree.ElementTree as ET
import sys
from html.parser import HTMLParser
from urllib.parse import quote, urlparse, urlunparse
import io
import zipfile

from src.pdf2bpmn.processgpt.bpmn_xml_generator import ProcessGPTBPMNXmlGenerator
from src.pdf2bpmn.processgpt.process_definition_prompt import build_system_prompt_processgpt
from src.pdf2bpmn.processgpt.process_consulting_prompt import get_process_consulting_system_prompt
from src.pdf2bpmn.processgpt.process_generation_messages import build_process_definition_messages
from src.pdf2bpmn.config import Config
from src.pdf2bpmn.models.entities import Document as PdfDocument, Section, ReferenceChunk
from src.pdf2bpmn.process_post_processor import ProcessPostProcessor
from src.pdf2bpmn.skill_enricher import (
    SkillEnricher,
    build_activity_index,
    render_skill_markdown,
)
from src.pdf2bpmn.hitl import (
    HitlPauseException,
    build_question_payload,
    clear_hitl_checkpoint,
    mark_hitl_process_resolved,
    stable_hitl_question_id,
    custom_text as hitl_custom_text,
    emit_human_feedback_received,
    emit_waiting_for_user,
    is_skipped as hitl_is_skipped,
    make_question_id,
    pause_for_hitl,
    read_batch_responses,
    selected_ids as hitl_selected_ids,
)

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

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

# ---------------------------------------------------------------------------
# Form generation prompt data (ported from process-gpt-vue3)
# - Source: process-gpt-vue3/src/components/ai/FormDesignGeneratorPromptSnipptsData.js
# - NOTE: examples are intentionally omitted to reduce token/cost; rules + component specs are kept.
# ---------------------------------------------------------------------------

FORM_CONTAINER_SPACE_SETS: List[List[int]] = [
    [12],
    [6, 6],
    [4, 8],
    [8, 4],
    [4, 4, 4],
    [3, 6, 3],
    [3, 3, 3, 3],
]

FORM_COMPONENT_INFOS: List[Dict[str, str]] = [
    {
        "tagName": "boolean-field",
        "tag": "<boolean-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></boolean-field>",
        "purpose": "To select either 'true' or 'false'",
        "limit": "",
    },
    {
        "tagName": "user-select-field",
        "tag": "<user-select-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></user-select-field>",
        "purpose": "To select users from the system",
        "limit": "",
    },
    {
        "tagName": "select-field",
        "tag": (
            "<select-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></select-field>"
        ),
        "purpose": "To select one option from multiple choices",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "checkbox-field",
        "tag": (
            "<checkbox-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></checkbox-field>"
        ),
        "purpose": "To select multiple options from a list of choices",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "radio-field",
        "tag": (
            "<radio-field name='<unique_identifier>' alias='<display_label>' is_dynamic_load='<fixed|urlBinding>' "
            "items='<options_list_when_is_dynamic_load_is_false>' "
            "dynamic_load_url='<JSON_data_load_URL_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_key_json_path='<JSON_PATH_for_key_array_when_is_dynamic_load_is_urlBinding>' "
            "dynamic_load_value_json_path='<JSON_PATH_for_value_array_when_is_dynamic_load_is_urlBinding>' "
            "disabled='<true|false>' readonly='<true|false>'></radio-field>"
        ),
        "purpose": "To select one option from multiple listed choices (displayed as radio buttons)",
        "limit": (
            "When is_dynamic_load is fixed, items is required and must be formatted as "
            """'[{"key1": "label1"}, {"key2": "label2"}]'. """
            "When is_dynamic_load is urlBinding, dynamic_load_url, dynamic_load_key_json_path, and "
            "dynamic_load_value_json_path are all required."
        ),
    },
    {
        "tagName": "file-field",
        "tag": "<file-field name='<unique_identifier>' alias='<display_label>' disabled='<true|false>' readonly='<true|false>'></file-field>",
        "purpose": "To upload files",
        "limit": "",
    },
    {
        "tagName": "label-field",
        "tag": "<label-field label='<label_text>'></label-field>",
        "purpose": "To provide descriptive text for components",
        "limit": "Not needed for components that already have name and alias attributes (which automatically generate labels)",
    },
    {
        "tagName": "report-field",
        "tag": "<report-field name='<unique_identifier>' alias='<display_label>'></report-field>",
        "purpose": "To collect markdown input",
        "limit": "Write markdown body only; use '---' as section separators when needed.",
    },
    {
        "tagName": "slide-field",
        "tag": "<slide-field name='<unique_identifier>' alias='<display_label>'></slide-field>",
        "purpose": "To collect slide input",
        "limit": "Write markdown body only; use '---' as section separators when needed.",
    },
    {
        "tagName": "bpmn-uengine-field",
        "tag": "<bpmn-uengine-field name='<unique_identifier>' alias='<display_label>'></bpmn-uengine-field>",
        "purpose": "To collect BPMN process definitions as XML",
        "limit": "Use this field when the user explicitly asks for a BPMN process editor or diagram input.",
    },
    {
        "tagName": "text-field",
        "tag": "<text-field name='<unique_identifier>' alias='<display_label>' type='<text|number|email|url|date|datetime-local|month|week|time|password|tel|color>' disabled='<true|false>' readonly='<true|false>'></text-field>",
        "purpose": "To collect various types of text input",
        "limit": "For selections with many options (like years), use text-field instead of select-field",
    },
    {
        "tagName": "textarea-field",
        "tag": "<textarea-field name='<unique_identifier>' alias='<display_label>' rows='<number_of_rows>' disabled='<true|false>' readonly='<true|false>'></textarea-field>",
        "purpose": "To collect multi-line text input",
        "limit": "",
    },
]


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

        # Docker 컨테이너 내부에서 localhost 해석 차이 보정
        # - 일부 환경에서 localhost가 IPv6(::1) 우선으로 해석되며 0.0.0.0 바인딩 서버에 접속이 실패할 수 있어
        #   PDF2BPMN_URL의 host가 localhost면 127.0.0.1로 고정합니다.
        if self._is_running_in_docker():
            self.pdf2bpmn_url = self._rewrite_localhost_url(self.pdf2bpmn_url, localhost_target="127.0.0.1")

        # Claude Skills 서비스(제품 UI가 사용하는 스킬 스토리지)
        self.claude_skills_base_url = os.getenv(
            "CLAUDE_SKILLS_BASE_URL",
            self.config.get("claude_skills_base_url", "http://localhost:8088/claude-skills"),
        )
        if self._is_running_in_docker():
            self.claude_skills_base_url = self._rewrite_localhost_url(
                self.claude_skills_base_url, localhost_target="127.0.0.1"
            )
        
        # Supabase 설정
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SERVICE_ROLE_KEY')
        self.supabase_client: Optional[Client] = None

        # OpenAI-compatible client (for form generation)
        self.openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.openai_base_url = (os.getenv("LLM_BASE_URL") or "").strip()
        # Keep for logging/debug compatibility only.
        self.openai_timeout_sec = 0.0
        # Default to gpt-4.1 for longer, more stable structured outputs.
        self.openai_model = (os.getenv("LLM_MODEL") or "gpt-4.1").strip()
        # Separate models (requested: user/agent creation vs process creation)
        self.user_mapping_model = os.getenv("USER_MAPPING_MODEL", self.openai_model)
        self.process_definition_model = os.getenv("PROCESS_DEF_MODEL", self.openai_model)
        self.openai_client: Optional[OpenAI] = None

        # ---- 프로세스 실행 검증(validation) 설정 --------------------------------
        # 생성 완료 후 process-gpt-completion 실행 엔진(실제 /initiate·/complete)으로
        # start→end 실행 테스트를 돌려, 결함이 있으면 자동 개선한다.
        # - PDF2BPMN_VALIDATION_ENABLED: 검증 단계 on/off (기본 on)
        # - COMPLETION_ENGINE_URL: process-gpt-completion API 서버 base URL
        # - PDF2BPMN_VALIDATION_MAX_ITERS: 검증-개선 루프 최대 반복 횟수
        # - PDF2BPMN_VALIDATION_CLEANUP: 검증용 테스트 인스턴스 정리 여부 (기본 on)
        self.validation_enabled = (
            os.getenv("PDF2BPMN_VALIDATION_ENABLED", "true").strip().lower() == "true"
        )
        self.completion_engine_url = (
            os.getenv("COMPLETION_ENGINE_URL", "http://localhost:8000").strip()
        )
        if self._is_running_in_docker():
            self.completion_engine_url = self._rewrite_localhost_url(
                self.completion_engine_url, localhost_target="127.0.0.1"
            )
        try:
            self.validation_max_iters = max(
                1, int(os.getenv("PDF2BPMN_VALIDATION_MAX_ITERS", "100"))
            )
        except Exception:
            self.validation_max_iters = 100
        # 검증은 실제 프로세스 인스턴스를 만든다. 기본값은 '보존'(false) — 검증이 실제
        # 엔진으로 어떤 흐름을 탔는지 DB(todolist/bpm_proc_inst)에서 직접 확인할 수 있게
        # 남겨둔다. 누적이 부담되면 PDF2BPMN_VALIDATION_CLEANUP=true 로 삭제 가능.
        self.validation_cleanup = (
            os.getenv("PDF2BPMN_VALIDATION_CLEANUP", "false").strip().lower() == "true"
        )
        # 제출(/complete) 후 폴링 서비스가 다음 단계로 진행(SUBMITTED→다음 활동 또는
        # 프로세스 완료)할 때까지 기다리는 최대 시간(초). 마지막 태스크의 완료 처리가
        # 폴링 주기상 늦어질 수 있어 넉넉히 둔다(기본 70초).
        try:
            self.validation_advance_timeout = max(
                5.0, float(os.getenv("PDF2BPMN_VALIDATION_ADVANCE_TIMEOUT", "70"))
            )
        except Exception:
            self.validation_advance_timeout = 70.0
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                client_kwargs = {"api_key": self.openai_api_key}
                if self.openai_base_url:
                    client_kwargs["base_url"] = self.openai_base_url
                self.openai_client = OpenAI(**client_kwargs)
            except Exception as e:
                logger.warning(f"[WARN] OpenAI client init failed: {e}")
                self.openai_client = None
        
        # HTTP 클라이언트
        self.http_client: Optional[httpx.AsyncClient] = None

        # Org/user/agent cache (lazy)
        self._org_loaded: bool = False
        self._org_config_uuid: Optional[str] = None
        self._org_value: Optional[Dict[str, Any]] = None  # configuration.value (may include chart + extras)
        self._org_chart: Optional[Dict[str, Any]] = None
        self._org_teams_by_name: Dict[str, str] = {}  # normalized team name -> team(node) id
        self._org_team_name_by_id: Dict[str, str] = {}  # team(node) id -> display name
        self._org_members_by_team_id: Dict[str, List[str]] = {}  # team(node) id -> [user_id...]

        # users table cache
        self._users: List[Dict[str, Any]] = []   # all users (agents + humans)
        self._agents: List[Dict[str, Any]] = []  # users where is_agent=true

        # ProcessGPT flow toggle:
        # - When enabled, DO NOT use PDF2BPMN-generated BPMN XML as the source of truth.
        # - Instead: Neo4j extracted info -> (user mapping LLM) -> (process definition LLM) -> (ProcessGPTBPMNXmlGenerator.create_bpmn_xml) -> save.
        # 요구사항: 기존 XML 생성/활용 경로는 사용하지 않고, 이 흐름만 사용합니다.
        self._enable_processgpt_flow = True
        self._processgpt_bpmn_xml_generator = ProcessGPTBPMNXmlGenerator()

        # LLM-based assignment controls
        # - ENABLE_LLM_ROLE_MAPPING: allow LLM to suggest best assignee (existing user/agent/team)
        self._enable_llm_role_mapping: bool = os.getenv("ENABLE_LLM_ROLE_MAPPING", "true").lower() == "true"
        # Temporary runtime switch:
        # - false: skip skill document generation/upload/sync paths
        self._enable_skill_generation: bool = os.getenv("ENABLE_SKILL_GENERATION", "false").lower() == "true"
        # Post-process policy: task instruction 기반 스킬 추출 + lane-role 기준 agent 생성
        self._skill_extraction_min_ratio: float = float(
            os.getenv("SKILL_EXTRACTION_MIN_RATIO", str(Config.SKILL_EXTRACTION_MIN_RATIO))
        )
        self._skill_extraction_min_count: int = int(
            os.getenv("SKILL_EXTRACTION_MIN_COUNT", str(Config.SKILL_EXTRACTION_MIN_COUNT))
        )
        self._agent_creation_min_tasks_per_skill_per_lane: int = int(
            os.getenv(
                "AGENT_CREATION_MIN_TASKS_PER_SKILL_PER_LANE",
                str(Config.AGENT_CREATION_MIN_TASKS_PER_SKILL_PER_LANE),
            )
        )
        self._agent_creation_require_automation: bool = (
            os.getenv(
                "AGENT_CREATION_REQUIRE_AUTOMATION",
                "true" if Config.AGENT_CREATION_REQUIRE_AUTOMATION else "false",
            ).lower()
            == "true"
        )
        # Process definition generation mode:
        # - true: use LLM generation first, then fallback/normalize
        # - false: skip LLM and build deterministic definition from extracted data
        self._use_llm_procdef_enrich: bool = os.getenv("USE_LLM_PROCDEF_ENRICH", "true").lower() == "true"
        self._llm_assignment_min_conf: float = float(os.getenv("LLM_ASSIGNMENT_MIN_CONFIDENCE", "0.72"))

        # In-run cache to avoid repeated LLM calls per role name
        self._role_assignment_cache: Dict[str, Dict[str, Any]] = {}
        
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

    # -----------------------------------------------------------------------
    # Skill upload + Supabase sync (UI-compatible)
    # -----------------------------------------------------------------------

    def _normalize_skill_key(self, name: str) -> str:
        """Claude Skills 서비스 경로에 쓸 안전한 스킬 키."""
        s = " ".join(str(name or "").split()).strip()
        s = s.replace("\\", "_").replace("/", "_").replace("..", "_")
        return s[:160] if len(s) > 160 else s

    def _extract_skill_name_from_markdown(self, markdown: str) -> str:
        """`# Skill: ...` 헤더에서 스킬명을 추출."""
        if not markdown:
            return ""
        # preferred: YAML frontmatter name
        m = re.search(r"^---\s*\n[\s\S]*?^name:\s*(.+?)\s*$[\s\S]*?^---\s*$", markdown, flags=re.MULTILINE)
        if m:
            return " ".join(m.group(1).strip().strip("\"'").split()).strip()
        m = re.search(r"^#\s*Skill:\s*(.+)\s*$", markdown, flags=re.MULTILINE)
        if m:
            return " ".join(m.group(1).split()).strip()
        # fallback: 첫 줄이 '# ' 로 시작하면 그 제목을 사용
        first = (markdown.splitlines() or [""])[0].strip()
        if first.startswith("#"):
            return " ".join(first.lstrip("#").split()).strip()
        return ""

    def _match_skill_score_for_activity(self, activity: Dict[str, Any], skill_meta: Dict[str, Any]) -> float:
        """Activity와 skill 메타 간의 단순 키워드 매칭 점수(0~1)."""
        a_text = " ".join(
            str(x or "")
            for x in (
                activity.get("name"),
                activity.get("role"),
                activity.get("instruction"),
                activity.get("description"),
                activity.get("tool"),
            )
        )
        s_text = " ".join(
            str(x or "")
            for x in (
                skill_meta.get("name"),
                skill_meta.get("summary"),
                skill_meta.get("purpose"),
                skill_meta.get("procedure_text"),
            )
        )
        a_norm = self._normalize_text_key(a_text)
        s_norm = self._normalize_text_key(s_text)
        if not a_norm or not s_norm:
            return 0.0

        # direct containment gives strong signal
        if a_norm in s_norm or s_norm in a_norm:
            return 0.95

        # token overlap score
        a_tokens = {t for t in a_norm.split() if len(t) >= 2}
        s_tokens = {t for t in s_norm.split() if len(t) >= 2}
        if not a_tokens or not s_tokens:
            return 0.0
        overlap = a_tokens & s_tokens
        if not overlap:
            return 0.0
        denom = max(len(a_tokens), 1)
        return min(0.9, len(overlap) / denom)

    def _match_skill_score_for_agent(
        self,
        *,
        agent_profile: Dict[str, Any],
        skill_meta: Dict[str, Any],
        activity_text: str = "",
        role_hints: Optional[Set[str]] = None,
    ) -> float:
        """Agent profile + assigned activities 기반 스킬 적합도 점수(0~1)."""
        role_hints = role_hints or set()
        profile_text = " ".join(
            str(x or "")
            for x in (
                agent_profile.get("username"),
                agent_profile.get("role"),
                agent_profile.get("goal"),
                agent_profile.get("persona"),
                agent_profile.get("tools"),
                " ".join(sorted(role_hints)),
                activity_text or "",
            )
        )
        skill_text = " ".join(
            str(x or "")
            for x in (
                skill_meta.get("name"),
                skill_meta.get("summary"),
                skill_meta.get("purpose"),
                skill_meta.get("procedure_text"),
            )
        )

        p_norm = self._normalize_text_key(profile_text)
        s_norm = self._normalize_text_key(skill_text)
        if not p_norm or not s_norm:
            return 0.0

        if p_norm in s_norm or s_norm in p_norm:
            return 0.95

        p_tokens = {t for t in p_norm.split() if len(t) >= 2}
        s_tokens = {t for t in s_norm.split() if len(t) >= 2}
        if not p_tokens or not s_tokens:
            return 0.0
        overlap = p_tokens & s_tokens
        if not overlap:
            return 0.0

        # weighted overlap: profile relevance + activity evidence
        jaccard = len(overlap) / max(len(p_tokens | s_tokens), 1)
        coverage = len(overlap) / max(len(s_tokens), 1)
        return min(0.98, (jaccard * 0.55) + (coverage * 0.45))

    def _build_skill_zip_bytes(self, *, skill_name: str, file_name: str, content: str) -> bytes:
        """`/skills/upload`용 zip 바이트 생성(내부에 반드시 SKILL.md 포함)."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # backend parser requirement: archive must contain SKILL.md
            arcname = f"{skill_name}/SKILL.md"
            zf.writestr(arcname, content or "")
        return bio.getvalue()

    async def _upload_skill_to_claude_skills(
        self,
        *,
        tenant_id: str,
        skill_name: str,
        file_name: str,
        content: str,
    ) -> bool:
        """
        Claude Skills 서비스에 스킬 파일을 업로드합니다.
        - POST /skills/upload (zip + tenant_id) 를 기본 경로로 사용
        - 서버는 zip 내부에 SKILL.md가 있어야 스킬로 인식합니다.
        """
        base = (self.claude_skills_base_url or "").rstrip("/")
        if not base:
            return False

        client = await self._get_http_client()
        safe_skill = self._normalize_skill_key(skill_name)
        safe_file = "SKILL.md"

        # upload zip (UI의 uploadSkills와 동일한 엔드포인트)
        try:
            zip_bytes = self._build_skill_zip_bytes(skill_name=safe_skill, file_name=safe_file, content=content)
            files = {"file": (f"{safe_skill}.zip", zip_bytes, "application/zip")}
            data = {"tenant_id": tenant_id}
            upload_url = f"{base}/skills/upload"
            resp = await client.post(upload_url, data=data, files=files)
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(
                f"[WARN] claude-skills upload failed: status={resp.status_code} body={(resp.text or '')[:300]}"
            )
        except Exception as e:
            logger.warning(f"[WARN] claude-skills upload exception: skill={safe_skill!r} err={e}")

        return False

    async def _sync_skills_to_supabase(
        self,
        *,
        tenant_id: str,
        skill_names: List[str],
        agent_user_ids: Set[str],
    ) -> None:
        """Supabase에 skills/agent_skills를 동기화(스키마 차이를 고려해 best-effort)."""
        if not self.supabase_client:
            return

        normalized_skills = [self._normalize_skill_key(s) for s in (skill_names or []) if str(s or "").strip()]
        normalized_skills = [s for s in normalized_skills if s]
        if not normalized_skills:
            return

        # A) users.skills 업데이트 + agent_skills upsert
        for uid in sorted({str(x).strip() for x in (agent_user_ids or set()) if str(x).strip()}):
            try:
                # agent 여부 확인(아니면 스킵)
                ures = (
                    self.supabase_client.table("users")
                    .select("id, tenant_id, is_agent, skills")
                    .eq("id", uid)
                    .eq("tenant_id", tenant_id)
                    .execute()
                )
                row = (ures.data[0] if getattr(ures, "data", None) else None) if ures else None
                if not isinstance(row, dict) or row.get("is_agent") is not True:
                    continue

                current = str(row.get("skills") or "")
                existing = [s.strip() for s in current.split(",") if s.strip()]
                merged = list(dict.fromkeys(existing + normalized_skills))  # preserve order, unique
                merged_str = ",".join(merged)

                self.supabase_client.table("users").update({"skills": merged_str}).eq("id", uid).eq(
                    "tenant_id", tenant_id
                ).execute()

                # agent_skills: best-effort insert (ignore duplicates)
                for s in normalized_skills:
                    try:
                        self.supabase_client.table("agent_skills").insert(
                            {"user_id": uid, "tenant_id": tenant_id, "skill_name": s}
                        ).execute()
                    except Exception:
                        # pk 충돌/스키마 차이 등은 무시
                        pass
            except Exception as e:
                logger.warning(f"[WARN] sync users/agent_skills failed: user_id={uid} err={e}")

        # B) tenants.skills 동기화(테넌트 스키마가 환경마다 다를 수 있어 best-effort)
        try:
            tres = self.supabase_client.table("tenants").select("*").eq("id", tenant_id).execute()
            trow = (tres.data[0] if getattr(tres, "data", None) else None) if tres else None
            if isinstance(trow, dict) and ("skills" in trow):
                current = str(trow.get("skills") or "")
                existing = [s.strip() for s in current.split(",") if s.strip()]
                merged = list(dict.fromkeys(existing + normalized_skills))
                self.supabase_client.table("tenants").update({"skills": ",".join(merged)}).eq("id", tenant_id).execute()
        except Exception:
            # tenants.skills 미존재 등은 무시
            pass

    # -----------------------------------------------------------------------
    # Form generation + saving (B안: proc_def 저장 후 폼 생성/저장)
    # -----------------------------------------------------------------------

    def _build_form_generator_base_messages(self) -> List[Dict[str, Any]]:
        """FormDesignGenerator.js의 시스템/가이드 프롬프트를 python용으로 구성합니다."""
        container_space_sets_prompt_str = ", ".join("{" + ", ".join(map(str, s)) + "}" for s in FORM_CONTAINER_SPACE_SETS)

        component_infos_prompt_str = "\n".join(
            [
                "#### {tagName}\n"
                "1. Tag Syntax\n"
                "`{tag}`\n\n"
                "2. Purpose\n"
                "{purpose}{limit_part}\n".format(
                    tagName=c["tagName"],
                    tag=c["tag"],
                    purpose=c.get("purpose", ""),
                    limit_part=("\n\n3. Limitation\n" + c["limit"]) if c.get("limit") else "",
                )
                for c in FORM_COMPONENT_INFOS
            ]
        )

        # NOTE:
        # - datasourcePrompt/datasourceURL은 워커 환경에서 보통 없음 → null로 두고 사용 금지 가이드만 둠
        datasource_prompt = "null"
        datasource_url = "null"

        system = {
            "role": "system",
            "content": (
                "# Role\n"
                "You are an HTML form creator assistant for process management systems, designed to generate and modify structured forms with precision and adherence to specific component guidelines.\n\n"
                "## Expertise\n"
                "- Expert in creating semantically structured HTML forms for business process management\n"
                "- Proficient in implementing grid-based layouts with proper containment hierarchies\n"
                "- Skilled at translating user requirements into functional forms\n"
                "- Specialized in component organization and responsive column distribution\n\n"
                "## Behavior Guidelines\n"
                "- Generate forms that strictly adhere to the provided component specifications\n"
                "- Maintain consistency in naming patterns and attribute formats\n"
                "- Produce clean, well-structured HTML that follows established patterns\n"
                "- Verify uniqueness of all name attributes across the entire form\n\n"
                "## Output Standards\n"
                "- Provide only valid HTML that conforms to the specified tag structure\n"
                "- Return responses in the exact JSON format specified in the guidelines\n\n"
                "# Instruction for DataSource Use\n"
                "You may be given a set of available dataSources before generating fields.\n"
                "If there is no datasource or datasourceURL is null, do not use dataSources.\n"
            ),
        }

        user_guideline = {
            "role": "user",
            "content": (
                "# Task Guidelines\n"
                "## About Task\n"
                "You create forms based on user instructions.\n"
                "You must only use the tags specified in the provided documentation.\n\n"
                "## Creating a Form from Scratch\n"
                "### Layout Structure\n"
                "First, create a layout to contain components.\n\n"
                "Layout example:\n"
                "```html\n"
                "<section>\n"
                "  <div class='row' name='<unique_layout_name>' alias='<layout_display_name>' is_multidata_mode='<true|false>'>\n"
                "      <div class='col-sm-6'>\n"
                "      </div>\n"
                "      <div class='col-sm-6'>\n"
                "      </div>\n"
                "  </div>\n"
                "</section>\n"
                "```\n\n"
                "- A section must contain exactly one div with class='row'.\n"
                "- Inside a div with class='row', you must include divs with class='col-sm-{number}'.\n"
                "- The sum of all {number} values in a row must equal 12.\n"
                f"- You must use one of these column combinations: [{container_space_sets_prompt_str}]\n"
                "- Layouts can be nested by placing a new section inside a col-sm div.\n\n"
                "### Adding Components\n"
                "- All components must be placed inside a div with class='col-sm-{number}'.\n"
                "- Every name attribute (including in div.row) must be unique.\n"
                "- For non-array string attributes, only use Korean characters, numbers, English letters, spaces, underscores(_), hyphens(-), and periods(.)\n"
                "- When creating a form, if there is no suitable result to create (insufficient task information), a text area with a default label of \"Free Input\" should be created. The form must exist.\n\n"
                "### How to infer fields from task information (flexible)\n"
                "- Use the task name/description/instruction to infer the minimum necessary inputs.\n"
                "- Prefer concrete business fields (dates, amounts, identifiers, decision/result, comment, attachments) when the text suggests them.\n"
                "- If the task clearly involves a human decision (e.g., approval/reject/hold), include fields for decision and rationale.\n"
                "- If the task involves money/payment/deposit, include date/amount/payer/proof fields.\n"
                "- If the task involves review/verification, include result and comment fields.\n"
                "- If the task involves contract/signature, include contract id/date/sign method fields.\n"
                "- These are suggestions: do NOT invent details that contradict the document; when uncertain, fall back to Free Input.\n\n"
                "### Available components\n"
                f"{component_infos_prompt_str}\n\n"
                f"{datasource_prompt}\n"
                "# Datasource URL\n"
                f"{datasource_url}\n\n"
                "### Output Format\n"
                "When responding, provide only the JSON response in markdown format, wrapped in triple backticks:\n"
                "```json\n"
                "{\n"
                '  "htmlOutput": "Generated form HTML code"\n'
                "}\n"
                "```\n"
            ),
        }

        assistant_ack = {"role": "assistant", "content": "Approved."}
        return [system, user_guideline, assistant_ack]

    def _make_fallback_form_html(self) -> str:
        # 프롬프트 가이드(폼은 비어있으면 안 됨)에 맞춘 안전한 최소 폼
        return (
            "<section>"
            "  <div class='row' name='free_input_layout' alias='Free Input' is_multidata_mode='false'>"
            "    <div class='col-sm-12'>"
            "      <textarea-field name='free_input' alias='Free Input' rows='5' disabled='false' readonly='false'></textarea-field>"
            "    </div>"
            "  </div>"
            "</section>"
        )

    async def _call_openai_for_form_html(self, request_text: str) -> str:
        """LLM 호출로 폼 HTML 생성. 실패 시 예외를 던집니다(상위에서 폴백 처리)."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client is not configured (missing OPENAI_API_KEY or openai package).")

        messages = self._build_form_generator_base_messages()
        # FormDesignGenerator의 noteMessage와 유사: alias는 한국어, name은 영어 권장
        note = "Please write values such as alias and label of the form being created in Korean. However, make sure all name attributes are written in English only."
        user_message = (
            "# Request Type\n"
            "Create\n\n"
            "# Request\n"
            f"{request_text}\n\n"
            "# Note\n"
            f"{note}\n"
        )
        messages.append({"role": "user", "content": user_message})

        def _run():
            return self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=float(os.getenv("FORM_LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("FORM_LLM_MAX_TOKENS", "2500")),
            )

        # 원복: 헤지/하드타임아웃 없이 응답 완료까지 대기
        resp = await asyncio.to_thread(_run)
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty LLM response.")

        # 응답이 ```json ... ``` 형태일 수 있음 → code fence 제거
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
        if fence_match:
            content = fence_match.group(1).strip()

        try:
            obj = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM JSON: {e}. raw={content[:300]}...")

        html = (obj.get("htmlOutput") or "").strip()
        if not html:
            raise RuntimeError("LLM JSON did not include htmlOutput.")
        return html

    def _extract_fields_json_from_form_html(self, html: str) -> List[Dict[str, Any]]:
        """프론트 `extractFields()` 로직을 python으로 포팅."""

        field_tags = {
            "text-field",
            "select-field",
            "checkbox-field",
            "radio-field",
            "file-field",
            "label-field",
            "boolean-field",
            "textarea-field",
            "user-select-field",
            "report-field",
            "slide-field",
            "bpmn-uengine-field",
        }

        class _FieldParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.fields: List[Dict[str, Any]] = []

            def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]):
                t = (tag or "").lower()
                if t not in field_tags:
                    return
                attr = {k.lower(): v for (k, v) in attrs if k}

                alias = attr.get("alias") or ""
                name_attr = attr.get("name") or ""
                v_model = attr.get("v-model") or ""

                # v-model 바인딩에서 bracket 표기법 키 우선 추출, 없으면 name 사용
                key = name_attr
                m = re.search(r"\[['\"](.+?)['\"]\]", v_model)
                if m and m.group(1):
                    key = m.group(1)

                field_type = attr.get("type") or t.replace("-field", "")
                disabled = attr.get("disabled") if "disabled" in attr else False
                readonly = attr.get("readonly") if "readonly" in attr else False

                self.fields.append(
                    {
                        "text": alias,
                        "key": key,
                        "type": field_type,
                        "disabled": disabled,
                        "readonly": readonly,
                    }
                )

        parser = _FieldParser()
        parser.feed(html or "")
        return parser.fields

    async def _save_form_def(self, *, form_def: Dict[str, Any], tenant_id: str) -> bool:
        """form_def 테이블에 저장 (프론트 putRawDefinition(type=form)과 호환되는 컬럼 사용)."""
        if not self.supabase_client:
            logger.error("[ERROR] Supabase client is None! Cannot save form_def")
            return False

        try:
            proc_def_id = form_def.get("proc_def_id")
            activity_id = form_def.get("activity_id")
            form_id = form_def.get("id")

            if not proc_def_id or not activity_id or not form_id:
                raise ValueError("form_def requires id/proc_def_id/activity_id")

            # 기존 row 탐색(프론트와 동일 기준: tenant_id + proc_def_id + activity_id)
            existing = (
                self.supabase_client.table("form_def")
                .select("uuid,id")
                .eq("tenant_id", tenant_id)
                .eq("proc_def_id", proc_def_id)
                .eq("activity_id", activity_id)
                .execute()
            )

            if existing.data and len(existing.data) > 0:
                existing_uuid = existing.data[0].get("uuid")
                # uuid가 있으면 uuid 기준 업데이트(레거시 호환)
                if existing_uuid:
                    self.supabase_client.table("form_def").update(
                        {
                            "id": form_id,
                            "html": form_def.get("html"),
                            "proc_def_id": proc_def_id,
                            "activity_id": activity_id,
                            "fields_json": form_def.get("fields_json") or [],
                            "tenant_id": tenant_id,
                        }
                    ).eq("uuid", existing_uuid).execute()
                else:
                    # uuid가 없으면 id 기준으로 업데이트 시도
                    self.supabase_client.table("form_def").update(
                        {
                            "html": form_def.get("html"),
                            "fields_json": form_def.get("fields_json") or [],
                        }
                    ).eq("id", form_id).execute()
            else:
                self.supabase_client.table("form_def").insert(
                    {
                        "id": form_id,
                        "html": form_def.get("html"),
                        "proc_def_id": proc_def_id,
                        "activity_id": activity_id,
                        "fields_json": form_def.get("fields_json") or [],
                        "tenant_id": tenant_id,
                    }
                ).execute()

            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to save form_def: {e}")
            logger.error(traceback.format_exc())
            return False

    def _compute_form_def_id(self, *, proc_def_id: str, activity: Dict[str, Any]) -> str:
        """프론트와 동일한 form id 결정 규칙."""
        tool = (activity.get("tool") or "").strip()
        activity_id = (activity.get("id") or "").strip()

        form_id = ""
        if tool.startswith("formHandler:"):
            form_id = tool.replace("formHandler:", "", 1).strip()
        if not form_id:
            form_id = f"{proc_def_id}_{activity_id}_form"

        # 프론트는 '/'를 '#'로 치환
        form_id = form_id.replace("/", "#")
        if not form_id or form_id == "defaultform":
            form_id = f"{proc_def_id}_{activity_id.lower()}_form"
        return form_id

    async def _ensure_forms_for_process(
        self,
        *,
        proc_def_id: str,
        process_name: str,
        proc_json: Dict[str, Any],
        tenant_id: str,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        proc_def 저장 후, activity별 폼 생성+저장을 완료합니다(프론트가 없어도 수행).

        Returns:
          {
            "forms_saved": int,
            "activities": int,
            "forms": {
              "<activity_id>": {
                "form_id": "<form_def.id>",
                "fields_json": [ {text,key,type,...}, ... ]   # from _extract_fields_json_from_form_html
              }
            }
          }
        """
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list) or not activities:
            return {"forms_saved": 0, "activities": 0, "forms": {}}

        forms_saved = 0
        total = len(activities)
        forms_by_activity_id: Dict[str, Dict[str, Any]] = {}
        max_forms = int(os.getenv("FORM_MAX_PER_PROCESS", "200"))
        form_llm_concurrency = max(1, int(os.getenv("FORM_LLM_CONCURRENCY", "4")))
        if total > max_forms:
            activities = activities[:max_forms]
            total = len(activities)
        sem = asyncio.Semaphore(form_llm_concurrency)

        async def _process_one_form(idx: int, a: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
            if not isinstance(a, dict):
                return False, "", {}

            activity_id = str(a.get("id") or f"Activity_{idx+1}")
            activity_name = str(a.get("name") or f"활동 {idx+1}")
            role_name = str(a.get("role") or "")
            instruction = str(a.get("instruction") or "")
            description = str(a.get("description") or "")
            input_data = a.get("inputData") or []
            output_data = a.get("outputData") or []

            form_def_id = self._compute_form_def_id(proc_def_id=proc_def_id, activity=a)
            # IMPORTANT:
            # - form_def는 우리가 생성/저장하므로, 프로세스 정의(activity.tool)도 동일 id를 참조해야 프론트가 기본폼(defaultform) 대신 생성된 폼을 사용합니다.
            # - proc_def는 이미 저장되었더라도, 상위에서 definition 업데이트를 다시 수행합니다.
            a["tool"] = f"formHandler:{form_def_id}"

            await self._send_progress_event(
                event_queue,
                context_id,
                task_id,
                job_id,
                f"[FORM] 폼 생성 시작 ({idx+1}/{total}): {process_name} / {activity_name}",
                "tool_usage_started",
                92,
                {"proc_def_id": proc_def_id, "activity_id": activity_id, "form_def_id": form_def_id},
            )

            request_text = (
                f"다음 BPM 프로세스의 사용자 태스크에 필요한 입력 폼을 생성하세요.\n\n"
                f"- 프로세스명: {process_name}\n"
                f"- 프로세스ID(proc_def_id): {proc_def_id}\n"
                f"- 태스크ID(activity_id): {activity_id}\n"
                f"- 태스크명: {activity_name}\n"
                f"- 담당 역할: {role_name}\n\n"
                f"태스크 설명:\n{description}\n\n"
                f"태스크 지시사항(instruction):\n{instruction}\n\n"
                f"입력 데이터 후보(inputData): {json.dumps(input_data, ensure_ascii=False)}\n"
                f"출력 데이터 후보(outputData): {json.dumps(output_data, ensure_ascii=False)}\n\n"
                f"요구사항:\n"
                f"- 태스크 수행에 필요한 최소 입력 필드를 포함하세요.\n"
                f"- 필드 alias는 한국어로, name은 영어로 작성하세요.\n"
                f"- 태스크 정보가 충분하지 않다면, 자유입력(Free Input) 중심의 폼이 생성되어도 괜찮습니다.\n"
            )

            async with sem:
                html = ""
                # 1) LLM 시도
                try:
                    html = await self._call_openai_for_form_html(request_text)
                except Exception as e:
                    # 운영상 폼은 반드시 존재해야 하므로 폴백 폼으로 진행
                    logger.warning(f"[WARN] form LLM failed. process={proc_def_id} activity={activity_id} err={e}")
                    html = self._make_fallback_form_html()

                # 2) fields_json 추출
                try:
                    fields_json = self._extract_fields_json_from_form_html(html)
                except Exception as e:
                    logger.warning(f"[WARN] fields_json extract failed. fallback empty. err={e}")
                    fields_json = []

                # 3) 저장
                ok = await self._save_form_def(
                    form_def={
                        "id": form_def_id,
                        "html": html,
                        "proc_def_id": proc_def_id,
                        "activity_id": activity_id,
                        "fields_json": fields_json,
                    },
                    tenant_id=tenant_id,
                )

            await self._send_progress_event(
                event_queue,
                context_id,
                task_id,
                job_id,
                f"[FORM] 폼 저장 {'성공' if ok else '실패'}: {activity_name} (form_id={form_def_id})",
                "tool_usage_finished",
                95,
                {"proc_def_id": proc_def_id, "activity_id": activity_id, "form_def_id": form_def_id, "saved": ok},
            )

            return ok, activity_id, {"form_id": form_def_id, "fields_json": fields_json}

        form_tasks: List[asyncio.Task] = []
        for idx, a in enumerate(activities):
            if not isinstance(a, dict):
                continue
            form_tasks.append(asyncio.create_task(_process_one_form(idx, a)))

        if form_tasks:
            results = await asyncio.gather(*form_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"[WARN] form task failed unexpectedly: {res}")
                    continue
                ok, activity_id, form_meta = res
                if ok:
                    forms_saved += 1
                if activity_id:
                    forms_by_activity_id[activity_id] = form_meta

        return {"forms_saved": forms_saved, "activities": total, "forms": forms_by_activity_id}

    # -----------------------------------------------------------------------
    # Post-process expansion: inputData wiring after forms exist
    # -----------------------------------------------------------------------

    def _extract_form_field_refs(self, form_id: str, fields_json: Any) -> List[Dict[str, str]]:
        """
        Convert fields_json (from _extract_fields_json_from_form_html) into a list of candidates:
          [{"ref": "<form_id>.<field_key>", "label": "...", "type": "..."}, ...]
        """
        out: List[Dict[str, str]] = []
        if not form_id:
            return out
        if not isinstance(fields_json, list):
            return out
        seen: Set[str] = set()
        for f in fields_json:
            if not isinstance(f, dict):
                continue
            key = str(f.get("key") or "").strip()
            if not key:
                continue
            ref = f"{form_id}.{key}"
            if ref in seen:
                continue
            seen.add(ref)
            out.append(
                {
                    "ref": ref,
                    "label": str(f.get("text") or ""),
                    "type": str(f.get("type") or ""),
                }
            )
        return out

    def _build_predecessor_activity_map_for_gateways(self, proc_json: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        gateway_id -> 그 게이트웨이로 도달 가능한 모든 선행 activity_id 리스트.
        - sequences 의 source->target 그래프를 역방향으로 탐색하여 activity 노드만 수집.
        - sequences 가 없거나 비어있으면 fallback 으로 activities 의 list order 를 그대로 사용 (모든 활동을 선행으로 간주).
        """
        gateways = proc_json.get("gateways") or []
        if not isinstance(gateways, list) or not gateways:
            return {}
        gateway_ids = [str(g.get("id")) for g in gateways if isinstance(g, dict) and g.get("id")]
        if not gateway_ids:
            return {}

        activities = proc_json.get("activities") or []
        activity_ids = [str(a.get("id")) for a in activities if isinstance(a, dict) and a.get("id")]
        activity_id_set = set(activity_ids)

        sequences = proc_json.get("sequences") or []
        if not isinstance(sequences, list) or not sequences:
            return {gid: list(activity_ids) for gid in gateway_ids}

        rev: Dict[str, List[str]] = {}
        edge_count = 0
        for s in sequences:
            if not isinstance(s, dict):
                continue
            src = str(s.get("source") or "").strip()
            tgt = str(s.get("target") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            rev.setdefault(tgt, []).append(src)
            edge_count += 1

        if edge_count == 0:
            return {gid: list(activity_ids) for gid in gateway_ids}

        out: Dict[str, List[str]] = {}
        for gid in gateway_ids:
            seen_nodes: Set[str] = set()
            preds: List[str] = []
            q: List[str] = list(rev.get(gid) or [])
            while q:
                cur = q.pop(0)
                if cur in seen_nodes:
                    continue
                seen_nodes.add(cur)
                if cur in activity_id_set:
                    preds.append(cur)
                for p in rev.get(cur) or []:
                    if p not in seen_nodes:
                        q.append(p)
                if len(seen_nodes) > 5000:
                    break
            preds_sorted = [x for x in activity_ids if x in set(preds)]
            out[gid] = preds_sorted
        return out

    def _build_predecessor_activity_map(self, proc_json: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build a mapping: activity_id -> list of predecessor activity_ids (reachable via sequences).
        - Uses runtime proc_def.definition shape: activities + sequences (+ events/gateways).
        - If sequences are missing/invalid, falls back to "list order" (all previous activities).
        """
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list):
            return {}
        activity_ids = [str(a.get("id")) for a in activities if isinstance(a, dict) and a.get("id")]
        activity_id_set = set(activity_ids)

        sequences = proc_json.get("sequences") or []
        if not isinstance(sequences, list) or not sequences:
            # fallback: list order
            out2: Dict[str, List[str]] = {}
            prev: List[str] = []
            for aid in activity_ids:
                out2[aid] = list(prev)
                prev.append(aid)
            return out2

        # Build reverse adjacency for all nodes (events/gateways/activities)
        rev: Dict[str, List[str]] = {}
        edge_count = 0
        for s in sequences:
            if not isinstance(s, dict):
                continue
            src = str(s.get("source") or "").strip()
            tgt = str(s.get("target") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            rev.setdefault(tgt, []).append(src)
            edge_count += 1

        if edge_count == 0:
            out2 = {}
            prev = []
            for aid in activity_ids:
                out2[aid] = list(prev)
                prev.append(aid)
            return out2

        # For each activity, walk backwards through rev graph and collect activity nodes.
        out: Dict[str, List[str]] = {}
        for aid in activity_ids:
            seen_nodes: Set[str] = set()
            preds: List[str] = []
            q: List[str] = list(rev.get(aid) or [])
            while q:
                cur = q.pop(0)
                if cur in seen_nodes:
                    continue
                seen_nodes.add(cur)
                if cur in activity_id_set and cur != aid:
                    preds.append(cur)
                for p in rev.get(cur) or []:
                    if p not in seen_nodes:
                        q.append(p)
                # safety guard to avoid pathological loops
                if len(seen_nodes) > 5000:
                    break
            # Stable order: keep by activities list order (older first)
            preds_sorted = [x for x in activity_ids if x in set(preds)]
            out[aid] = preds_sorted
        return out

    async def _llm_choose_inputdata_for_process(
        self,
        *,
        process_name: str,
        proc_def_id: str,
        proc_json: Dict[str, Any],
        candidates_by_activity_id: Dict[str, List[Dict[str, str]]],
    ) -> Optional[Dict[str, List[str]]]:
        """
        Decide inputData for activities, using ONLY provided candidates.
        Returns: { activity_id: [ "form_id.field_key", ... ] }
        """
        if not self.openai_client:
            return None
        if os.getenv("ENABLE_LLM_INPUTDATA_MAPPING", "true").lower() != "true":
            return None

        activities = proc_json.get("activities") or []
        if not isinstance(activities, list):
            return None

        tasks_payload: List[Dict[str, Any]] = []
        for a in activities:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if not aid:
                continue
            cands = candidates_by_activity_id.get(aid) or []
            # Keep prompt compact
            tasks_payload.append(
                {
                    "task_id": aid,
                    "name": str(a.get("name") or ""),
                    "role": str(a.get("role") or ""),
                    "description": str(a.get("description") or ""),
                    "instruction": str(a.get("instruction") or ""),
                    "candidates": cands[:120],  # cap
                }
            )

        system_prompt = (
            "당신은 BPM 프로세스의 각 태스크(UserTask)에 대해 inputData(참조 데이터)를 설계하는 전문가입니다.\n"
            "규칙:\n"
            "- inputData에는 반드시 제공된 candidates.ref 값만 넣을 수 있습니다.\n"
            "- inputData는 '이 태스크를 수행할 때 참고하면 좋은 이전 태스크의 입력값'이어야 합니다.\n"
            "- 불필요한 참조는 넣지 마세요. 꼭 필요한 것만 선택하세요.\n"
            "- 출력은 JSON ONLY 입니다.\n"
        )

        user_prompt = (
            f"프로세스명: {process_name}\n"
            f"proc_def_id: {proc_def_id}\n\n"
            "각 태스크별 후보(candidates) 중에서 inputData로 적절한 것들을 골라주세요.\n"
            "반환 형식:\n"
            "{\n"
            '  "mappings": [\n'
            '    {"task_id": "...", "inputData": ["form_id.field_key", "..."]}\n'
            "  ]\n"
            "}\n\n"
            f"tasks:\n{json.dumps(tasks_payload, ensure_ascii=False)}\n"
        )

        obj = await self._call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(os.getenv("LLM_INPUTDATA_MAX_TOKENS", "1400")),
            model=os.getenv("INPUTDATA_MAPPING_MODEL", self.process_definition_model),
            temperature=float(os.getenv("LLM_INPUTDATA_TEMPERATURE", "0.0")),
        )
        if not isinstance(obj, dict):
            return None
        mappings = obj.get("mappings")
        if not isinstance(mappings, list):
            return None

        out: Dict[str, List[str]] = {}
        for m in mappings:
            if not isinstance(m, dict):
                continue
            tid = str(m.get("task_id") or "").strip()
            if not tid:
                continue
            arr = m.get("inputData") or []
            if not isinstance(arr, list):
                continue
            cleaned: List[str] = []
            seen: Set[str] = set()
            allowed = {c.get("ref") for c in (candidates_by_activity_id.get(tid) or []) if isinstance(c, dict) and c.get("ref")}
            for x in arr:
                ref = str(x or "").strip()
                if not ref or ref in seen:
                    continue
                if allowed and ref not in allowed:
                    continue
                seen.add(ref)
                cleaned.append(ref)
            out[tid] = cleaned
        return out

    async def _llm_choose_conditiondata_for_gateways(
        self,
        *,
        process_name: str,
        proc_def_id: str,
        proc_json: Dict[str, Any],
        candidates_by_gateway_id: Dict[str, List[Dict[str, str]]],
    ) -> Optional[Dict[str, List[str]]]:
        """
        각 ExclusiveGateway 의 conditionData(분기 판단용 참조 필드)를 LLM 으로 선택.
        - inputData 와 동일한 형식: ["form_id.field_key", ...]
        - 반드시 candidates.ref 안에서만 선택
        - 분기 시퀀스(outgoing)의 name/condition 을 함께 제공하여 어떤 필드가 분기 판단에 필요한지 추론하게 함
        """
        if not self.openai_client:
            return None
        if os.getenv("ENABLE_LLM_CONDITIONDATA_MAPPING", "true").lower() != "true":
            return None

        gateways = proc_json.get("gateways") or []
        if not isinstance(gateways, list) or not gateways:
            return None

        sequences = proc_json.get("sequences") or []
        out_seqs_by_gateway: Dict[str, List[Dict[str, str]]] = {}
        if isinstance(sequences, list):
            for s in sequences:
                if not isinstance(s, dict):
                    continue
                src = str(s.get("source") or "").strip()
                if not src:
                    continue
                out_seqs_by_gateway.setdefault(src, []).append(
                    {
                        "id": str(s.get("id") or ""),
                        "name": str(s.get("name") or ""),
                        "condition": str(s.get("condition") or ""),
                        "target": str(s.get("target") or ""),
                    }
                )

        gateways_payload: List[Dict[str, Any]] = []
        for g in gateways:
            if not isinstance(g, dict):
                continue
            gid = str(g.get("id") or "").strip()
            if not gid:
                continue
            gtype = str(g.get("type") or "").lower()
            # 분기 판단이 의미있는 게이트웨이만 (exclusive/inclusive). parallel 은 조건 없이 모두 진행.
            if "parallel" in gtype:
                continue
            cands = candidates_by_gateway_id.get(gid) or []
            gateways_payload.append(
                {
                    "gateway_id": gid,
                    "name": str(g.get("name") or ""),
                    "type": g.get("type") or "ExclusiveGateway",
                    "description": str(g.get("description") or ""),
                    "branches": out_seqs_by_gateway.get(gid) or [],
                    "candidates": cands[:120],
                }
            )

        if not gateways_payload:
            return None

        system_prompt = (
            "당신은 BPM 프로세스의 분기 게이트웨이(ExclusiveGateway/InclusiveGateway)에 대해 "
            "conditionData(분기 판단에 필요한 참조 데이터)를 설계하는 전문가입니다.\n"
            "규칙:\n"
            "- conditionData 에는 반드시 제공된 candidates.ref 값만 넣을 수 있습니다.\n"
            "- conditionData 는 '이 게이트웨이의 분기 조건을 평가할 때 참조해야 하는 이전 태스크 폼의 필드' 여야 합니다.\n"
            "- branches[].name 과 branches[].condition 을 읽고, 분기 판단을 가능하게 하는 핵심 필드만 선택하세요.\n"
            "- 보통 게이트웨이 직전 태스크의 결과 필드(분기 결과/판정값/체크박스 등)가 핵심이며, 너무 많이 넣지 마세요.\n"
            "- 출력은 JSON ONLY 입니다.\n"
        )

        user_prompt = (
            f"프로세스명: {process_name}\n"
            f"proc_def_id: {proc_def_id}\n\n"
            "각 게이트웨이별 후보(candidates) 중에서 conditionData 로 적절한 것들을 골라주세요.\n"
            "반환 형식:\n"
            "{\n"
            '  "mappings": [\n'
            '    {"gateway_id": "...", "conditionData": ["form_id.field_key", "..."]}\n'
            "  ]\n"
            "}\n\n"
            f"gateways:\n{json.dumps(gateways_payload, ensure_ascii=False)}\n"
        )

        obj = await self._call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(os.getenv("LLM_CONDITIONDATA_MAX_TOKENS", "1200")),
            model=os.getenv("CONDITIONDATA_MAPPING_MODEL", self.process_definition_model),
            temperature=float(os.getenv("LLM_CONDITIONDATA_TEMPERATURE", "0.0")),
        )
        if not isinstance(obj, dict):
            return None
        mappings = obj.get("mappings")
        if not isinstance(mappings, list):
            return None

        out: Dict[str, List[str]] = {}
        for m in mappings:
            if not isinstance(m, dict):
                continue
            gid = str(m.get("gateway_id") or "").strip()
            if not gid:
                continue
            arr = m.get("conditionData") or []
            if not isinstance(arr, list):
                continue
            cleaned: List[str] = []
            seen: Set[str] = set()
            allowed = {c.get("ref") for c in (candidates_by_gateway_id.get(gid) or []) if isinstance(c, dict) and c.get("ref")}
            for x in arr:
                ref = str(x or "").strip()
                if not ref or ref in seen:
                    continue
                if allowed and ref not in allowed:
                    continue
                seen.add(ref)
                cleaned.append(ref)
            out[gid] = cleaned
        return out

    async def _postprocess_skills_and_tasks(
        self,
        *,
        proc_json: Dict[str, Any],
        process_name: str,
    ) -> List[Dict[str, Any]]:
        """
        activity 지침에서 반복 패턴을 스킬로 추출 → LLM 으로 풍부한 스킬 카드(SOP) 생성
        → activity.skills 에 부착 → process-scope 메타로 채움.

        - LLM 호출이 비활성화/실패하면 캐노니컬 문장 기반 폴백 카드를 그대로 사용한다.
        - 반환되는 metas 에는 후속 단계(_apply_assignment_..., SKILL.md 직렬화)가 그대로
          쓸 수 있도록 풍부한 필드 (description/when_to_use/procedure/...) 가 포함된다.
        """
        try:
            processor = ProcessPostProcessor(
                min_ratio=self._skill_extraction_min_ratio,
                min_count=self._skill_extraction_min_count,
                lane_skill_min_tasks=self._agent_creation_min_tasks_per_skill_per_lane,
                require_automation=self._agent_creation_require_automation,
            )

            # 1) 클러스터링만 수행
            cluster_result = processor.build_skill_clusters(proc_json)
            reusable = cluster_result.get("reusable") or []
            threshold = cluster_result.get("threshold")

            if not reusable:
                logger.info(
                    "[SKILL][POST] process=%r skills=0 threshold=%s (no reusable cluster)",
                    process_name, threshold,
                )
                processor.apply_enriched_skills(proc_json, [])
                return []

            # 2) LLM 으로 enrich (실패/비활성 시 자동 폴백)
            enricher = SkillEnricher()
            activity_by_id = build_activity_index(proc_json)
            try:
                cards = await enricher.enrich_clusters(
                    clusters=reusable,
                    process_name=process_name,
                    activity_by_id=activity_by_id,
                    post_processor=processor,
                )
            except Exception as e:
                logger.warning(
                    "[SKILL][LLM] enrich_clusters failed (%s) → fallback for %d clusters",
                    e, len(reusable),
                )
                cards = [
                    processor.fallback_skill_card(c, idx)
                    for idx, c in enumerate(reusable, start=1)
                ]

            # 3) proc_json 에 부착
            processor.apply_enriched_skills(proc_json, cards)

            logger.info(
                "[SKILL][POST] process=%r skills=%d threshold=%s",
                process_name, len(cards), threshold,
            )

            # 4) downstream 메타 (이미 LLM 카드 형식이므로 그대로 통과)
            metas: List[Dict[str, Any]] = []
            for s in cards:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name") or "").strip()
                safe = str(s.get("safe_name") or "").strip()
                if not name or not safe:
                    continue
                meta = dict(s)
                # 필수 호환 키 보강
                meta["id"] = safe
                meta["safe_name"] = safe
                meta["purpose"] = meta.get("description") or meta.get("summary") or ""
                meta.setdefault("procedure_text", "")
                metas.append(meta)
            return metas
        except Exception as e:
            logger.warning(f"[WARN] postprocess_skills_and_tasks failed: {e}")
            proc_json["skills"] = []
            return []

    def _pick_existing_agent_for_lane_skill(
        self,
        *,
        role_name: str,
        skill_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """역할 + 스킬 키워드로 기존 agent 재사용 후보를 찾음."""
        role_key = self._normalize_text_key(role_name)
        skill_keys = [self._normalize_text_key(s) for s in skill_names if str(s or "").strip()]
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for a in (self._agents or []):
            if not isinstance(a, dict) or not a.get("id"):
                continue
            text = " ".join(
                str(x or "")
                for x in (
                    a.get("username"),
                    a.get("role"),
                    a.get("alias"),
                    a.get("description"),
                    a.get("goal"),
                    a.get("persona"),
                    " ".join(a.get("skills") or []) if isinstance(a.get("skills"), list) else "",
                )
            )
            tkey = self._normalize_text_key(text)
            if not tkey:
                continue
            score = 0.0
            if role_key and role_key in tkey:
                score += 1.0
            score += sum(0.5 for sk in skill_keys if sk and sk in tkey)
            if score > best_score:
                best_score = score
                best = a
        if best_score <= 0:
            return None
        return best

    async def _assign_or_create_agents_by_lane_skill(
        self,
        *,
        proc_json: Dict[str, Any],
        tenant_id: str,
        process_name: str,
    ) -> Dict[str, Set[str]]:
        """
        lane(role)-skill 집계로 agent를 생성/재사용하고 activities/roles를 업데이트.
        반환: {agent_id: {skill_name...}}
        """
        await self._load_org_and_agents(tenant_id)
        activities = proc_json.get("activities") or []
        roles = proc_json.get("roles") or []
        if not isinstance(activities, list):
            activities = []
        if not isinstance(roles, list):
            roles = []
        proc_skills = proc_json.get("skills") or []
        skill_name_by_id = {
            str(s.get("id") or "").strip(): str(s.get("name") or "").strip()
            for s in proc_skills
            if isinstance(s, dict)
        }

        processor = ProcessPostProcessor(
            min_ratio=self._skill_extraction_min_ratio,
            min_count=self._skill_extraction_min_count,
            lane_skill_min_tasks=self._agent_creation_min_tasks_per_skill_per_lane,
            require_automation=self._agent_creation_require_automation,
        )
        candidates = processor.collect_lane_skill_candidates(proc_json)
        logger.info("[ASSIGN][LANE] process=%r candidates=%d", process_name, len(candidates))

        activity_by_id = {
            str(a.get("id") or "").strip(): a
            for a in activities
            if isinstance(a, dict) and str(a.get("id") or "").strip()
        }
        role_agent_by_name: Dict[str, str] = {}
        agent_skill_names: Dict[str, Set[str]] = {}

        for cand in candidates:
            role_name = str(cand.get("role") or "").strip()
            skill_id = str(cand.get("skill_id") or "").strip()
            activity_ids = [str(x).strip() for x in (cand.get("activity_ids") or []) if str(x).strip()]
            if not role_name or not skill_id or not activity_ids:
                continue

            skill_name = skill_name_by_id.get(skill_id) or skill_id
            agent_id = role_agent_by_name.get(role_name)

            if not agent_id:
                existing = self._pick_existing_agent_for_lane_skill(
                    role_name=role_name,
                    skill_names=[skill_name],
                )
                if existing and existing.get("id"):
                    agent_id = str(existing.get("id"))

            if not agent_id:
                team_id = self._org_teams_by_name.get(self._normalize_text_key(role_name)) or ""
                team_name = self._org_team_name_by_id.get(team_id) or role_name or "미분류"
                snippets: List[str] = []
                for aid in activity_ids[:6]:
                    a = activity_by_id.get(aid, {})
                    snippets.append(
                        " ".join(
                            str(x or "")
                            for x in (
                                a.get("name"),
                                a.get("instruction"),
                                a.get("description"),
                            )
                        )
                    )
                user_input = (
                    f"프로세스 '{process_name}'의 역할 '{role_name}'에 대해 다음 공통 스킬을 수행할 에이전트를 설계하세요.\n"
                    f"- 공통 스킬: {skill_name}\n"
                    f"- 대표 태스크 맥락: {' | '.join(snippets)}\n"
                    "불필요한 일반 업무는 제외하고 자동화 가능한 작업 중심으로 설계하세요."
                )
                mcp_tools = self._safe_json_loads(os.getenv("MCP_TOOLS_JSON", "")) or {}
                profile = await self._llm_generate_agent_profile(
                    team_name=team_name,
                    user_input=user_input,
                    mcp_tools=mcp_tools,
                )
                if profile:
                    created = await self._insert_agent_user(
                        tenant_id=tenant_id,
                        agent_profile=profile,
                        agent_type="agent",
                    )
                    if created and created.get("id"):
                        agent_id = str(created.get("id"))
                        if team_id:
                            await self._update_org_chart_add_member(
                                tenant_id=tenant_id,
                                team_id=team_id,
                                member_user=created,
                            )

            if not agent_id:
                continue

            role_agent_by_name[role_name] = agent_id
            agent_skill_names.setdefault(agent_id, set()).add(skill_name)

            for aid in activity_ids:
                a = activity_by_id.get(aid)
                if not isinstance(a, dict):
                    continue
                a["agent"] = agent_id
                # fixed policy for skill-assigned tasks
                a["agentMode"] = "complete"
                a["orchestration"] = "deepagents"

        # Candidates가 아닌 activity는 명시적으로 none 처리
        for a in activities:
            if not isinstance(a, dict):
                continue
            if isinstance(a.get("skills"), list) and a.get("skills"):
                a["agentMode"] = "complete"
                a["orchestration"] = "deepagents"
                continue
            if str(a.get("agent") or "").strip():
                continue
            a["agent"] = None
            a["agentMode"] = "none"
            a["orchestration"] = None

        # roles.endpoint 갱신 (lane/role에 agent 배치)
        for r in roles:
            if not isinstance(r, dict):
                continue
            rname = str(r.get("name") or "").strip()
            if not rname:
                continue
            agent_id = role_agent_by_name.get(rname)
            if agent_id:
                r["endpoint"] = agent_id
                r["origin"] = "used"

        proc_json["activities"] = activities
        proc_json["roles"] = roles
        return agent_skill_names

    async def _expand_process_after_forms(
        self,
        *,
        proc_def_id: str,
        process_name: str,
        proc_json: Dict[str, Any],
        forms_result: Dict[str, Any],
        extracted: Optional[Dict[str, Any]] = None,
        tenant_id: str,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Post-processing step AFTER forms exist:
        - Set inputData using real form_id + fields_json from earlier tasks
        """
        # 1) Build predecessors based on sequences
        pred_map = self._build_predecessor_activity_map(proc_json)

        # 2) Build candidate form-field refs per activity from predecessor activities only
        forms_by_activity_id = (forms_result.get("forms") or {}) if isinstance(forms_result, dict) else {}
        candidates_by_activity_id: Dict[str, List[Dict[str, str]]] = {}
        for aid, preds in (pred_map or {}).items():
            cand: List[Dict[str, str]] = []
            seen: Set[str] = set()
            for pid in preds:
                info = forms_by_activity_id.get(pid) if isinstance(forms_by_activity_id, dict) else None
                if not isinstance(info, dict):
                    continue
                form_id = str(info.get("form_id") or "").strip()
                fields_json = info.get("fields_json")
                for c in self._extract_form_field_refs(form_id, fields_json):
                    ref = c.get("ref") or ""
                    if ref and ref not in seen:
                        seen.add(ref)
                        cand.append(c)
            candidates_by_activity_id[aid] = cand

        # 3) Ask LLM to choose relevant inputData, otherwise fallback to "all candidates"
        await self._send_progress_event(
            event_queue,
            context_id,
            task_id,
            job_id,
            f"[EXPAND] inputData(참조 필드) 자동 설정을 시작합니다: {process_name}",
            "tool_usage_started",
            97,
            {"proc_def_id": proc_def_id},
        )

        chosen = await self._llm_choose_inputdata_for_process(
            process_name=process_name,
            proc_def_id=proc_def_id,
            proc_json=proc_json,
            candidates_by_activity_id=candidates_by_activity_id,
        )

        max_inputs = int(os.getenv("INPUTDATA_MAX_PER_TASK", "60"))
        activities = proc_json.get("activities") or []
        if isinstance(activities, list):
            for a in activities:
                if not isinstance(a, dict):
                    continue
                aid = str(a.get("id") or "").strip()
                if not aid:
                    continue

                # normalize agent fields (final)
                has_skills = isinstance(a.get("skills"), list) and len(a.get("skills") or []) > 0
                agent_id = str(a.get("agent") or "").strip()
                if has_skills:
                    # fixed policy for skill-assigned tasks
                    a["agentMode"] = "complete"
                    a["orchestration"] = "deepagents"
                elif agent_id:
                    a["agentMode"] = "draft"
                    a["orchestration"] = "crewai-action"
                else:
                    a["agentMode"] = "none"
                    a["orchestration"] = None

                # inputData:
                # - MUST be limited to predecessor candidates only (prevents referencing future/non-existent forms)
                allowed = {
                    str(c.get("ref"))
                    for c in (candidates_by_activity_id.get(aid) or [])
                    if isinstance(c, dict) and c.get("ref")
                }

                # 1) If LLM provided mapping for this task, it is already filtered by `allowed` upstream.
                if isinstance(chosen, dict) and aid in chosen:
                    new_inputs = chosen.get(aid) or []
                    if isinstance(new_inputs, list):
                        a["inputData"] = [str(x).strip() for x in new_inputs if str(x or "").strip()][:max_inputs]
                        continue

                # 2) Otherwise, sanitize any existing inputData to allowed-only.
                existing = a.get("inputData") or []
                sanitized: List[str] = []
                seen2: Set[str] = set()
                if isinstance(existing, list) and allowed:
                    for x in existing:
                        ref = str(x or "").strip()
                        if not ref or ref in seen2:
                            continue
                        if ref not in allowed:
                            continue
                        seen2.add(ref)
                        sanitized.append(ref)
                        if len(sanitized) >= max_inputs:
                            break

                # 3) If nothing left, fallback to "all candidates" (dedup) up to max_inputs.
                if not sanitized:
                    refs = [
                        str(c.get("ref") or "").strip()
                        for c in (candidates_by_activity_id.get(aid) or [])
                        if isinstance(c, dict)
                    ]
                    for r in refs:
                        if not r or r in seen2:
                            continue
                        seen2.add(r)
                        sanitized.append(r)
                        if len(sanitized) >= max_inputs:
                            break

                a["inputData"] = sanitized

        # 4) Gateways: conditionData(분기 판단용 참조 필드) 자동 설정
        #    - 게이트웨이의 선행 액티비티들의 폼 필드를 후보로 모아서, LLM 으로 분기 조건에 필요한 필드 선택.
        #    - LLM 미사용/실패 시 fallback: 가장 가까운 선행 액티비티의 모든 폼 필드를 conditionData 로 설정.
        gateway_pred_map = self._build_predecessor_activity_map_for_gateways(proc_json)
        gateways = proc_json.get("gateways") or []

        candidates_by_gateway_id: Dict[str, List[Dict[str, str]]] = {}
        nearest_pred_by_gateway_id: Dict[str, Optional[str]] = {}
        if isinstance(gateways, list) and gateways:
            for gid, preds in (gateway_pred_map or {}).items():
                cand: List[Dict[str, str]] = []
                seen3: Set[str] = set()
                for pid in preds:
                    info = forms_by_activity_id.get(pid) if isinstance(forms_by_activity_id, dict) else None
                    if not isinstance(info, dict):
                        continue
                    form_id = str(info.get("form_id") or "").strip()
                    fields_json = info.get("fields_json")
                    for c in self._extract_form_field_refs(form_id, fields_json):
                        ref = c.get("ref") or ""
                        if ref and ref not in seen3:
                            seen3.add(ref)
                            cand.append(c)
                candidates_by_gateway_id[gid] = cand
                nearest_pred_by_gateway_id[gid] = preds[-1] if preds else None

        chosen_gateways: Optional[Dict[str, List[str]]] = None
        if isinstance(gateways, list) and gateways:
            chosen_gateways = await self._llm_choose_conditiondata_for_gateways(
                process_name=process_name,
                proc_def_id=proc_def_id,
                proc_json=proc_json,
                candidates_by_gateway_id=candidates_by_gateway_id,
            )

        if isinstance(gateways, list):
            max_cond = int(os.getenv("CONDITIONDATA_MAX_PER_GATEWAY", "30"))
            for g in gateways:
                if not isinstance(g, dict):
                    continue
                gid = str(g.get("id") or "").strip()
                if not gid:
                    continue
                gtype = str(g.get("type") or "").lower()
                # parallel 게이트웨이는 조건 평가 자체가 없으므로 conditionData 미세팅
                if "parallel" in gtype:
                    if not isinstance(g.get("conditionData"), list):
                        g["conditionData"] = []
                    continue

                allowed = {
                    str(c.get("ref"))
                    for c in (candidates_by_gateway_id.get(gid) or [])
                    if isinstance(c, dict) and c.get("ref")
                }

                # 1) LLM 선택값 적용 (이미 allowed 필터링됨)
                if isinstance(chosen_gateways, dict) and gid in chosen_gateways:
                    new_inputs = chosen_gateways.get(gid) or []
                    if isinstance(new_inputs, list) and new_inputs:
                        g["conditionData"] = [str(x).strip() for x in new_inputs if str(x or "").strip()][:max_cond]
                        continue

                # 2) 기존값 정리 (allowed 만 통과)
                existing = g.get("conditionData") or []
                sanitized: List[str] = []
                seen4: Set[str] = set()
                if isinstance(existing, list) and allowed:
                    for x in existing:
                        ref = str(x or "").strip()
                        if not ref or ref in seen4:
                            continue
                        if ref not in allowed:
                            continue
                        seen4.add(ref)
                        sanitized.append(ref)
                        if len(sanitized) >= max_cond:
                            break

                # 3) Fallback: 가장 가까운 선행 액티비티의 모든 폼 필드를 그대로 conditionData 로
                if not sanitized:
                    nearest_pid = nearest_pred_by_gateway_id.get(gid)
                    if nearest_pid:
                        info = forms_by_activity_id.get(nearest_pid) if isinstance(forms_by_activity_id, dict) else None
                        if isinstance(info, dict):
                            form_id = str(info.get("form_id") or "").strip()
                            fields_json = info.get("fields_json")
                            for c in self._extract_form_field_refs(form_id, fields_json):
                                ref = str(c.get("ref") or "").strip()
                                if not ref or ref in seen4:
                                    continue
                                seen4.add(ref)
                                sanitized.append(ref)
                                if len(sanitized) >= max_cond:
                                    break

                g["conditionData"] = sanitized

        await self._send_progress_event(
            event_queue,
            context_id,
            task_id,
            job_id,
            f"[EXPAND] inputData(참조 필드) 자동 설정 완료: {process_name}",
            "tool_usage_finished",
            98,
            {"proc_def_id": proc_def_id},
        )

        return {
            "candidates_count": {k: len(v) for k, v in candidates_by_activity_id.items()},
            "gateway_candidates_count": {k: len(v) for k, v in candidates_by_gateway_id.items()},
            "llm_used": bool(isinstance(chosen, dict)),
            "llm_used_gateways": bool(isinstance(chosen_gateways, dict)),
        }

    async def _strip_dmn_from_saved_proc_defs(
        self,
        *,
        proc_def_ids: List[str],
        tenant_id: str,
    ) -> int:
        """사용자가 통합 elicit 에서 'DMN 적용 안 함' 을 선택한 경우 저장된 proc_def 들에서
        dmn 관련 키(decisions/rules/dmnXml/businessRuleTask 참조 등) 를 제거하고 update.

        반환: 처리된 row 수.
        """
        if not self.supabase_client or not proc_def_ids:
            return 0
        DMN_KEYS = ("dmn", "dmnXml", "dmn_xml", "decisions", "decision_rules", "dmn_decisions", "dmn_rules")
        count = 0
        for pid in proc_def_ids:
            pid_str = str(pid or "").strip()
            if not pid_str:
                continue
            try:
                res = self.supabase_client.table("proc_def").select("definition").eq("id", pid_str).limit(1).execute()
                rows = getattr(res, "data", None) or []
                if not rows:
                    continue
                definition = rows[0].get("definition") or {}
                if isinstance(definition, str):
                    try:
                        definition = json.loads(definition)
                    except Exception:
                        definition = {}
                if not isinstance(definition, dict):
                    continue
                changed = False
                for k in DMN_KEYS:
                    if k in definition:
                        definition.pop(k, None)
                        changed = True
                # activity 단의 businessRule 참조도 정리 (있을 경우)
                acts = definition.get("activities")
                if isinstance(acts, list):
                    for a in acts:
                        if not isinstance(a, dict):
                            continue
                        if a.get("type") == "businessRuleTask" or a.get("decisionRef"):
                            a.pop("decisionRef", None)
                            changed = True
                if changed:
                    self.supabase_client.table("proc_def").update(
                        {"definition": definition, "tenant_id": tenant_id, "isdeleted": False}
                    ).eq("id", pid_str).execute()
                    count += 1
            except Exception as e:
                logger.warning(f"[DMN-STRIP] proc_def {pid_str} 처리 실패: {e}")
        logger.info(f"[DMN-STRIP] DMN 제거 완료: {count}/{len(proc_def_ids)} proc_def")
        return count

    async def _update_proc_def_definition_only(self, *, proc_def_id: str, tenant_id: str, definition: Dict[str, Any]) -> bool:
        """proc_def.definition만 업데이트(폼 id 연결을 위해)."""
        if not self.supabase_client:
            return False
        try:
            # id는 tenant별 유니크라고 가정. (프론트도 id로 조회)
            self.supabase_client.table("proc_def").update(
                {
                    "definition": definition,
                    "tenant_id": tenant_id,
                    "isdeleted": False,
                }
            ).eq("id", proc_def_id).execute()
            return True
        except Exception as e:
            logger.warning(f"[WARN] proc_def.definition update failed: id={proc_def_id} err={e}")
            return False

    async def _update_proc_def_bpmn_only(self, *, proc_def_id: str, tenant_id: str, bpmn_xml: str) -> bool:
        """proc_def.bpmn만 업데이트(확장 단계 이후 최종 XML 반영용)."""
        if not self.supabase_client:
            return False
        try:
            self.supabase_client.table("proc_def").update(
                {
                    "bpmn": bpmn_xml,
                    "tenant_id": tenant_id,
                    "isdeleted": False,
                }
            ).eq("id", proc_def_id).execute()
            return True
        except Exception as e:
            logger.warning(f"[WARN] proc_def.bpmn update failed: id={proc_def_id} err={e}")
            return False

    def _apply_runtime_definition_to_elements_model(
        self,
        *,
        elements_model: Dict[str, Any],
        runtime_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        런타임 정의(proc_def.definition; activities/events/gateways/sequences 기반)에서
        XML 생성에 필요한 필드(tool/inputData/outputData/checkpoints/agent 등)를
        elements 모델(jsonModel.elements 기반)에 반영합니다.

        NOTE:
        - ProcessGPTBPMNXmlGenerator는 elements_model을 기준으로 uengine:json을 만듭니다.
        - 따라서 폼/참조정보 확장 이후의 최종 값을 XML에 반영하려면, 생성 직전에 sync가 필요합니다.
        """
        em = dict(elements_model or {})
        rd = dict(runtime_def or {})

        # top-level fields
        for k in ("processDefinitionId", "processDefinitionName", "megaProcessId", "majorProcessId", "description", "isHorizontal", "data"):
            if k in rd and rd.get(k) is not None:
                em[k] = rd.get(k)

        # roles: lane endpoint/resolutionRule는 roles에서 읽는다
        if isinstance(rd.get("roles"), list):
            em["roles"] = rd.get("roles") or []

        # build activity lookup by id
        acts_by_id: Dict[str, Dict[str, Any]] = {}
        for a in (rd.get("activities") or []):
            if isinstance(a, dict) and a.get("id"):
                acts_by_id[str(a.get("id"))] = a

        # build gateway lookup by id (conditionData sync 용)
        gws_by_id: Dict[str, Dict[str, Any]] = {}
        for g in (rd.get("gateways") or []):
            if isinstance(g, dict) and g.get("id"):
                gws_by_id[str(g.get("id"))] = g

        elems = em.get("elements")
        if not isinstance(elems, list):
            # The generator can accept dict-shaped elements too, but this backend path uses list.
            return em

        for e in elems:
            if not isinstance(e, dict):
                continue
            etype = e.get("elementType")

            if etype == "Activity":
                aid = str(e.get("id") or "").strip()
                if not aid or aid not in acts_by_id:
                    continue

                a = acts_by_id[aid]

                # keep canonical fields in sync
                if a.get("name"):
                    e["name"] = a.get("name")
                if a.get("description") is not None:
                    e["description"] = a.get("description") or ""
                if a.get("role") is not None:
                    e["role"] = a.get("role") or ""
                if isinstance(a.get("inputData"), list):
                    e["inputData"] = a.get("inputData") or []
                if isinstance(a.get("outputData"), list):
                    e["outputData"] = a.get("outputData") or []
                if isinstance(a.get("checkpoints"), list):
                    e["checkpoints"] = a.get("checkpoints") or []
                if isinstance(a.get("skills"), list):
                    e["skills"] = a.get("skills") or []

                # properties are serialized into uengine:json for tasks
                props = e.get("properties") if isinstance(e.get("properties"), dict) else {}
                props = dict(props)
                props.update(
                    {
                        "role": a.get("role"),
                        "duration": a.get("duration", 5),
                        "instruction": a.get("instruction") or "",
                        "tool": a.get("tool") or "",
                        "agent": a.get("agent", None),
                        "agentMode": a.get("agentMode") or "none",
                        "orchestration": a.get("orchestration", None),
                        "attachments": a.get("attachments") or [],
                        "skills": a.get("skills") or [],
                        "customProperties": a.get("customProperties") or [],
                    }
                )
                e["properties"] = props

            elif etype == "Gateway":
                # 게이트웨이의 conditionData(분기 판단용 참조 필드)도 elements_model 에 sync.
                # → BPMN XML 생성기가 properties(uengine:json)로 직렬화하여 프론트에서 읽을 수 있게 함.
                gid = str(e.get("id") or "").strip()
                if not gid or gid not in gws_by_id:
                    continue
                g = gws_by_id[gid]
                if g.get("name"):
                    e["name"] = g.get("name")
                if g.get("description") is not None:
                    e["description"] = g.get("description") or ""
                if g.get("role") is not None:
                    e["role"] = g.get("role") or ""
                if isinstance(g.get("conditionData"), list):
                    e["conditionData"] = g.get("conditionData") or []

        em["elements"] = elems
        return em

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
            "pdf_urls": [],
            "pdf_names": [],
            "input_files": [],
            "description": "",
            "room_id": "",
            "tenant_id": "",
            # 사용자 [도구 설정] 다이얼로그에서 선택한 처리 강도.
            # 메인 에이전트가 [InputData].tool_settings 로 넣어 주면 여기에 그대로 보존되어
            # workflow.set_dedup_level() 호출에 사용된다.
            # 예: {"pdf2bpmnLevel": "concise" | "standard" | "detailed"}
            "tool_settings": {},
            # 컨설팅 기반 생성 모드: [InputData].input_mode == "consulting" 이면
            # 파일 대신 컨설팅 내용(user_request/consulting_outline/user_answer/image_analysis)을
            # 담는다. 이 값이 채워지면 execute() 가 메멘토/섹션/그래프 추출을 건너뛴다.
            "consulting_payload": None,
            "raw_query": query
        }

        def _extract_json_object_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
            idx = text.find(marker)
            if idx < 0:
                return None
            start = text.find("{", idx + len(marker))
            if start < 0:
                return None

            depth = 0
            in_str = False
            escaped = False
            end = -1
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_str = False
                    continue

                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

            if end < 0:
                return None
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None

        def _normalize_files_from_input(data: Dict[str, Any]) -> List[Dict[str, str]]:
            files: List[Dict[str, str]] = []
            default_room_id = str(data.get("room_id") or "").strip()
            default_tenant_id = str(data.get("tenant_id") or "").strip()

            def _pick_url(item: Dict[str, Any]) -> str:
                return str(
                    item.get("fullPath")
                    or item.get("publicUrl")
                    or item.get("fileUrl")
                    or item.get("url")
                    or item.get("path")
                    or item.get("pdf_url")
                    or item.get("pdf_file_url")
                    or ""
                ).strip()

            def _pick_name(item: Dict[str, Any]) -> str:
                return str(
                    item.get("originalFileName")
                    or item.get("fileName")
                    or item.get("name")
                    or item.get("pdf_name")
                    or item.get("pdf_file_name")
                    or ""
                ).strip()

            def _append_candidate(candidate: Any):
                if isinstance(candidate, dict):
                    u = _pick_url(candidate)
                    if not u:
                        return
                    files.append(
                        {
                            "url": u,
                            "name": _pick_name(candidate),
                            "path": str(candidate.get("path") or "").strip().rstrip("?"),
                            "room_id": str(candidate.get("room_id") or default_room_id).strip(),
                            "tenant_id": str(candidate.get("tenant_id") or default_tenant_id).strip(),
                        }
                    )
                    return
                if isinstance(candidate, str) and candidate.strip():
                    files.append(
                        {
                            "url": candidate.strip(),
                            "name": "",
                            "path": "",
                            "room_id": default_room_id,
                            "tenant_id": default_tenant_id,
                        }
                    )

            # 단일 파일 호환 키
            for key in ("file", "pdf_file", "attachment"):
                if key in data:
                    _append_candidate(data.get(key))

            # 다중 파일 키
            for key in ("files", "pdf_files", "attachments", "uploaded_files"):
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        _append_candidate(item)

            # 최상위 키 자체가 파일 정보를 담는 경우
            top_url = _pick_url(data)
            if top_url:
                files.append({"url": top_url, "name": _pick_name(data)})

            # dedupe by URL
            deduped: List[Dict[str, str]] = []
            seen: Set[str] = set()
            for f in files:
                u = str(f.get("url") or "").strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                deduped.append(
                    {
                        "url": u,
                        "name": str(f.get("name") or "").strip(),
                        "path": str(f.get("path") or "").strip().rstrip("?"),
                        "room_id": str(f.get("room_id") or default_room_id).strip(),
                        "tenant_id": str(f.get("tenant_id") or default_tenant_id).strip(),
                    }
                )
            return deduped

        def _apply_parsed_data(data: Dict[str, Any]):
            result["room_id"] = str(data.get("room_id") or result.get("room_id") or "").strip()
            result["tenant_id"] = str(data.get("tenant_id") or result.get("tenant_id") or "").strip()
            # 컨설팅 기반 생성 모드 감지: 업로드 문서 없이 컨설팅 내용으로 프로세스를 만든다.
            if str(data.get("input_mode") or "").strip().lower() == "consulting":
                result["consulting_payload"] = {
                    "user_request": str(data.get("user_request") or "").strip(),
                    "consulting_outline": str(data.get("consulting_outline") or "").strip(),
                    "user_answer": str(data.get("user_answer") or "").strip(),
                    "image_analysis": str(data.get("image_analysis") or "").strip(),
                }
            # tool_settings 화이트리스트 검증 후 보존.
            ts = data.get("tool_settings")
            if isinstance(ts, dict) and ts:
                allowed_levels = {"concise", "standard", "detailed"}
                lv = str(ts.get("pdf2bpmnLevel") or "").strip().lower()
                cleaned: Dict[str, Any] = {}
                if lv in allowed_levels:
                    cleaned["pdf2bpmnLevel"] = lv
                if cleaned:
                    result["tool_settings"] = cleaned
            normalized_files = _normalize_files_from_input(data)
            if normalized_files:
                result["input_files"] = normalized_files
                result["pdf_urls"] = [f.get("url", "") for f in normalized_files if f.get("url")]
                result["pdf_names"] = [f.get("name", "") for f in normalized_files]
                result["pdf_url"] = result["pdf_urls"][0] if result["pdf_urls"] else ""
                result["pdf_name"] = result["pdf_names"][0] if result["pdf_names"] else ""
            else:
                result["pdf_url"] = data.get("pdf_url", data.get("fileUrl", data.get("pdf_file_url",
                                    data.get("fullPath", data.get("publicUrl", "")))))
                result["pdf_name"] = data.get("pdf_name", data.get("fileName", data.get("pdf_file_name",
                                    data.get("originalFileName", ""))))
                if result["pdf_url"]:
                    result["pdf_urls"] = [result["pdf_url"]]
                    result["pdf_names"] = [result["pdf_name"]]
                    result["input_files"] = [{"url": result["pdf_url"], "name": result["pdf_name"]}]
            result["description"] = data.get("description", result.get("description", ""))
        
        # 1. 순수 JSON 형식 파싱 시도
        try:
            if query.strip().startswith('{'):
                data = json.loads(query)
                _apply_parsed_data(data)
                return result
        except json.JSONDecodeError:
            pass
        
        # 2. [InputData] 형식에서 JSON 추출
        if "[InputData]" in query:
            data = _extract_json_object_after_marker(query, "[InputData]")
            if isinstance(data, dict):
                _apply_parsed_data(data)
                logger.info(
                    "[PARSE] Extracted from [InputData] JSON - "
                    f"files={len(result.get('input_files') or [])}, "
                    f"first={result.get('pdf_name') or result.get('pdf_url')}"
                )
                return result
            
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

        if result["pdf_url"] and not result["pdf_urls"]:
            result["pdf_urls"] = [result["pdf_url"]]
        if result["pdf_name"] and not result["pdf_names"]:
            result["pdf_names"] = [result["pdf_name"]]
        if result["pdf_urls"] and not result["input_files"]:
            result["input_files"] = [
                {"url": u, "name": result["pdf_names"][i] if i < len(result["pdf_names"]) else ""}
                for i, u in enumerate(result["pdf_urls"])
            ]

        # Docker 환경에서 로컬 Supabase(Storage) URL이 localhost/127.0.0.1로 들어오는 경우 보정
        # - 컨테이너 내부에서 127.0.0.1은 컨테이너 자신이므로, 호스트의 Supabase에 접근하려면 host.docker.internal로 바꿔야 함
        if self._is_running_in_docker():
            rewritten_files: List[Dict[str, str]] = []
            for item in result.get("input_files", []) or []:
                old_url = str(item.get("url") or "")
                new_url = self._rewrite_localhost_url(old_url, localhost_target="host.docker.internal")
                if old_url and new_url != old_url:
                    logger.info(f"[PARSE] Rewrote file URL for Docker: {old_url} -> {new_url}")
                rewritten_files.append(
                    {
                        "url": new_url or old_url,
                        "name": str(item.get("name") or ""),
                        "path": str(item.get("path") or "").strip().rstrip("?"),
                        "room_id": str(item.get("room_id") or result.get("room_id") or "").strip(),
                        "tenant_id": str(item.get("tenant_id") or result.get("tenant_id") or "").strip(),
                    }
                )
            if rewritten_files:
                result["input_files"] = rewritten_files
                result["pdf_urls"] = [f.get("url", "") for f in rewritten_files if f.get("url")]
                result["pdf_names"] = [f.get("name", "") for f in rewritten_files]
                result["pdf_url"] = result["pdf_urls"][0] if result["pdf_urls"] else result.get("pdf_url", "")
                result["pdf_name"] = result["pdf_names"][0] if result["pdf_names"] else result.get("pdf_name", "")
        
        return result

    @staticmethod
    def _is_running_in_docker() -> bool:
        """컨테이너 내부 실행 여부를 최대한 안전하게 판별"""
        try:
            return os.getenv("RUNNING_IN_DOCKER", "").lower() == "true" or Path("/.dockerenv").exists()
        except Exception:
            return False

    @staticmethod
    def _rewrite_localhost_url(url: str, localhost_target: str) -> str:
        """
        URL의 host가 localhost/127.0.0.1이면 localhost_target으로 치환합니다.
        예) http://127.0.0.1:54321/... -> http://host.docker.internal:54321/...
        """
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if host not in {"localhost", "127.0.0.1"}:
                return url

            netloc = localhost_target
            if p.port:
                netloc = f"{localhost_target}:{p.port}"
            return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
        except Exception:
            return url

    # NOTE: _download_file / _extract_storage_file_path 는 메멘토 재사용 흐름 도입으로 제거됨.
    # 파일 다운로드/PDF 변환은 모두 메멘토(`save-to-storage`)가 수행하며, pdf2bpmn 은
    # `_fetch_memento_chunks` 를 통해 사전 처리된 청크/임베딩만 가져온다.

    def _normalize_text_key(self, s: str) -> str:
        return re.sub(r"\s+", "", (s or "").strip().lower())

    # NOTE: _parse_with_synap / _extract_text_from_hwp_or_hwpx / _build_state_from_page_texts 는
    # 메멘토 재사용 흐름 도입으로 모두 제거되었다. 동일 기능은 메멘토가 수행하며,
    # 결과는 GET /documents/chunks-with-embeddings 로 그대로 가져와 사용한다.

    # =========================================================================
    # Memento integration helpers
    # -------------------------------------------------------------------------
    # 사용 흐름은 메인채팅(프론트) → 메멘토(`save-to-storage`) → 메인채팅 에이전트
    # → pdf2bpmn 으로 고정되어 있다.
    # 즉, pdf2bpmn 시점에는 이미 메멘토가 다음 작업을 마친 상태이다:
    #   - Supabase Storage 업로드 (PDF가 아니면 PDF로 변환된 페이지 텍스트를 보유)
    #   - 페이지/문서 텍스트 추출 후 chunking
    #   - 임베딩 (Chroma + Supabase documents)
    # 따라서 pdf2bpmn은 더 이상 "다운로드 → 변환 → ingest_pdf → embed" 를 직접 하지 않고
    # 메멘토에서 청크/임베딩을 가져와 그대로 활용한다.
    # =========================================================================

    def _memento_base_url(self) -> str:
        base = (os.getenv("MEMENTO_BASE_URL") or self.config.get("memento_base_url") or "").strip()
        if not base:
            base = (Config.MEMENTO_BASE_URL or "http://localhost:8005").strip()
        base = base.rstrip("/")
        if self._is_running_in_docker():
            base = self._rewrite_localhost_url(base, localhost_target="host.docker.internal")
        return base

    async def _fetch_memento_chunks(
        self,
        *,
        tenant_id: str,
        file_path: str = "",
        file_name: str = "",
        room_id: str = "",
        include_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        메멘토에서 사전 처리된 청크/임베딩을 가져온다.
        - tenant_id + (file_path 또는 file_name) 기준 조회
        - 반환: [{page_content, metadata, embedding?}, ...]
        - 실패 시 빈 리스트 반환 (호출 측에서 명시적 에러를 만들도록 함)
        """
        if not tenant_id:
            return []
        if not file_path and not file_name:
            return []

        base_url = self._memento_base_url()
        if not base_url:
            logger.warning("[MEMENTO] MEMENTO_BASE_URL이 비어있어 청크 조회를 건너뜁니다.")
            return []

        params: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "include_embeddings": "true" if include_embeddings else "false",
        }
        if file_path:
            params["file_path"] = file_path
        if file_name:
            params["file_name"] = file_name
        if room_id:
            params["room_id"] = room_id

        url = f"{base_url}/documents/chunks-with-embeddings"
        client = await self._get_http_client()
        try:
            resp = await client.get(
                url,
                params=params,
                timeout=float(Config.MEMENTO_TIMEOUT_SEC or 60.0),
            )
        except httpx.ConnectError as e:
            raise Exception(f"메멘토 연결 실패: {url} ({e})")

        if resp.status_code != 200:
            logger.warning(
                "[MEMENTO] chunks-with-embeddings 응답 비정상: status=%s body=%s",
                resp.status_code, resp.text[:300],
            )
            return []

        body = resp.json() or {}
        chunks = body.get("chunks") or []
        logger.info(
            "[MEMENTO] chunks fetched: file_path=%r file_name=%r room=%r tenant=%r "
            "→ chunks=%d embeddings=%d",
            file_path, file_name, room_id, tenant_id,
            len(chunks), int(body.get("embedding_count") or 0),
        )
        return chunks

    def _build_state_from_memento_chunks(
        self,
        *,
        display_name: str,
        source: str,
        memento_chunks: List[Dict[str, Any]],
        pdf2bpmn_level: str = "standard",
    ) -> Tuple[List[PdfDocument], List[Section], List[ReferenceChunk]]:
        """
        메멘토 청크를 그대로 활용해 (Document, [Section], [ReferenceChunk]) 를 구성한다.

        설계 원칙:
        - 청크는 "메멘토가 만든 그대로" ReferenceChunk 로 변환한다 (재청킹 X).
        - 임베딩은 메멘토에서 받은 값을 chunk.embedding 에 직접 세팅한다.
          → 이후 workflow.segment_sections 의 batch_embed_chunks 가
            embedding 이 이미 있는 청크는 자동 스킵하므로 재임베딩이 일어나지 않는다.
        - 섹션은 메멘토 청크의 page_number 별로 묶어 page_texts 를 재구성한 뒤
          기존 PDFExtractor 의 heading/SOP 분리 로직을 그대로 사용한다.
        """
        if not memento_chunks:
            return [], [], []

        from src.pdf2bpmn.extractors.pdf_extractor import PDFExtractor  # type: ignore
        import hashlib as _hashlib

        # 1) 청크들을 page_number 기준으로 정리 (page_number(1-based) 우선,
        #    없으면 0-based page +1, 둘 다 없으면 1로 폴백)
        def _resolve_page(meta: Dict[str, Any]) -> int:
            pno = meta.get("page_number")
            if pno is None and meta.get("page") is not None:
                try:
                    pno = int(meta.get("page")) + 1
                except (TypeError, ValueError):
                    pno = None
            try:
                return max(1, int(pno)) if pno is not None else 1
            except (TypeError, ValueError):
                return 1

        page_to_texts: Dict[int, List[str]] = {}
        normalized_chunks: List[Tuple[Dict[str, Any], str, int]] = []
        for ch in memento_chunks:
            text = str(ch.get("page_content") or "").strip()
            if not text:
                continue
            meta = ch.get("metadata") or {}
            page = _resolve_page(meta)
            page_to_texts.setdefault(page, []).append(text)
            normalized_chunks.append((ch, text, page))

        if not page_to_texts:
            return [], [], []

        page_texts: Dict[int, str] = {
            p: "\n\n".join(parts) for p, parts in page_to_texts.items() if parts
        }
        sorted_pages = sorted(page_texts.keys())

        # 2) Document
        doc = PdfDocument(
            title=Path(display_name).stem or display_name or "document",
            source=source,
            page_count=max(sorted_pages),
        )

        # 3) Section: 기존 heading/SOP 분리 로직 재사용
        extractor = PDFExtractor()
        # 사용자 [도구 설정] 의 pdf2bpmnLevel 을 SOP 분할 단계에도 적용한다.
        # SOP 개수 자체는 LLM 이 문서 내용을 보고 판단하며 (하드코딩 N 개 강제 X),
        # level 은 "분할에 얼마나 적극적인가" 의 임계만 조절한다.
        try:
            extractor.set_segmentation_level(pdf2bpmn_level)
        except Exception:
            pass
        all_text = [(p, page_texts[p]) for p in sorted_pages]
        sop_sections: List[Section] = []
        heading_sections: List[Section] = extractor._extract_sections(doc.doc_id, all_text)
        if Config.ENABLE_SOP_SEGMENTATION and Config.OPENAI_API_KEY:
            try:
                sop_sections = extractor._extract_sop_sections(doc.doc_id, page_texts)
            except Exception as e:
                logger.warning(f"[SECTION] SOP 분할 실패, heading 분할로 폴백: {e}")
                sop_sections = []

        if extractor._should_use_heading_sections(
            sop_sections=sop_sections,
            heading_sections=heading_sections,
            page_count=doc.page_count,
        ):
            sections = heading_sections
        else:
            sections = sop_sections or heading_sections

        if Config.FORCE_SINGLE_SECTION:
            sections = extractor._force_single_section(doc.doc_id, page_texts)

        # 4) ReferenceChunk: 메멘토 청크 그대로 변환 + 임베딩 설정
        ref_chunks: List[ReferenceChunk] = []
        reused_embedding_count = 0
        for ch, text, page in normalized_chunks:
            meta = ch.get("metadata") or {}
            try:
                chunk_index = int(meta.get("chunk_index") or 0)
            except (TypeError, ValueError):
                chunk_index = 0
            # span 은 정확한 원본 위치가 아니어도 downstream에서는 식별용으로만 쓰여
            # chunk_index 기준의 가짜 span 으로 충분하다.
            start = chunk_index * 1000
            text_hash = _hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

            embedding = ch.get("embedding")
            if isinstance(embedding, list) and len(embedding) == Config.EMBEDDING_DIMENSIONS:
                reused_embedding_count += 1
            else:
                # 차원 불일치/None 인 경우 None 으로 두면 segment_sections 의
                # batch_embed_chunks 가 해당 청크만 다시 임베딩한다.
                embedding = None

            ref_chunks.append(
                ReferenceChunk(
                    doc_id=doc.doc_id,
                    page=page,
                    span=f"{start}:{start + len(text)}",
                    text=text,
                    hash=text_hash,
                    embedding=embedding,
                )
            )

        logger.info(
            "[MEMENTO] state built: doc=%r pages=%d sections=%d chunks=%d "
            "reused_embeddings=%d/%d",
            doc.title, doc.page_count, len(sections), len(ref_chunks),
            reused_embedding_count, len(ref_chunks),
        )
        return [doc], sections, ref_chunks

    def _is_placeholder_gateway_name(self, name: str) -> bool:
        key = self._normalize_text_key(name)
        if not key:
            return True
        if re.match(r"^(분기|gateway|gw|decision)\d*$", key):
            return True
        if re.match(r"^(분기|gateway|gw|decision)[_-]?\d+$", key):
            return True
        return False

    def _gateway_subject_from_text(self, text: str) -> str:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return ""
        s = re.sub(r"[\"'`]+", "", s)
        for p in (r"(.+?)(?:인지|인가)\s*여부", r"(.+?)\s*여부"):
            m = re.search(p, s)
            if m:
                subj = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" .,-_")
                if subj:
                    return subj
        # 흔한 게이트웨이 접미어/불용어 제거(폴백)
        for tok in ("분기", "판단", "확인", "검토", "결정", "여부"):
            s = s.replace(tok, " ")
        s = re.sub(r"\s+", " ", s).strip(" .,-_")
        return s

    def _derive_gateway_name(self, *, raw_name: str, description: str, idx: int) -> str:
        name = str(raw_name or "").strip()
        # description에서 주제가 잡히면 초기 이름도 항상 해당 규칙으로 통일
        subject = self._gateway_subject_from_text(description)
        if subject:
            return f"{subject} 여부 판단"
        if name and not self._is_placeholder_gateway_name(name):
            return name
        subject = self._gateway_subject_from_text(name)
        if subject:
            return f"{subject} 여부 판단"
        return f"의사결정 분기 {idx}"

    def _derive_true_false_conditions(self, *, gateway_name: str, gateway_description: str) -> Tuple[str, str]:
        subject = self._gateway_subject_from_text(gateway_description) or self._gateway_subject_from_text(gateway_name)
        if subject:
            return (f"{subject}인 경우", f"{subject}이 아닌 경우")
        return ("조건 충족인 경우", "조건 미충족인 경우")

    def _safe_json_loads(self, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v

    def _extract_teams_from_org_chart(self, chart: Dict[str, Any]) -> Dict[str, str]:
        """
        configuration(key=organization).value.chart 트리에서 팀(부서) 노드를 추출합니다.
        기대 구조(프론트 기준):
          { id, data: { isTeam: true, name }, children: [...] }
        """
        teams: Dict[str, str] = {}

        def walk(node: Any):
            if not node or not isinstance(node, dict):
                return
            node_id = str(node.get("id") or "")
            data = node.get("data") or {}
            if isinstance(data, dict) and data.get("isTeam"):
                name = str(data.get("name") or node_id or "").strip()
                if name and node_id:
                    teams[self._normalize_text_key(name)] = node_id
            children = node.get("children") or []
            if isinstance(children, list):
                for ch in children:
                    walk(ch)

        walk(chart)
        return teams

    def _index_org_chart(self, chart: Dict[str, Any]) -> Dict[str, Any]:
        """
        조직도(chart)에서 다음 인덱스를 생성합니다.
        - teams_by_name: normalized team name -> team node id
        - team_name_by_id: team node id -> team display name
        - members_by_team_id: team node id -> [member user_id...]

        프론트에서 조직도에 멤버/에이전트 추가 시 child 노드는 다음 형태를 가집니다:
          { id: <users.id>, name: <display>, data: <users row-ish>, children?: [...] }
        """
        teams_by_name: Dict[str, str] = {}
        team_name_by_id: Dict[str, str] = {}
        members_by_team_id: Dict[str, List[str]] = {}

        def walk(node: Any, current_team_id: Optional[str] = None):
            if not node or not isinstance(node, dict):
                return
            node_id = str(node.get("id") or "")
            data = node.get("data") or {}
            is_team = isinstance(data, dict) and bool(data.get("isTeam"))

            next_team_id = current_team_id
            if is_team and node_id:
                team_name = str(data.get("name") or node_id).strip()
                if team_name:
                    teams_by_name[self._normalize_text_key(team_name)] = node_id
                    team_name_by_id[node_id] = team_name
                next_team_id = node_id
                members_by_team_id.setdefault(node_id, [])
            else:
                # member/agent node under a team
                if current_team_id and node_id:
                    members_by_team_id.setdefault(current_team_id, [])
                    if node_id not in members_by_team_id[current_team_id]:
                        members_by_team_id[current_team_id].append(node_id)

            children = node.get("children") or []
            if isinstance(children, list):
                for ch in children:
                    walk(ch, next_team_id)

        walk(chart, None)
        return {
            "teams_by_name": teams_by_name,
            "team_name_by_id": team_name_by_id,
            "members_by_team_id": members_by_team_id,
        }

    async def _load_org_and_agents(self, tenant_id: str):
        """Supabase에서 조직도/유저/에이전트 목록을 로드하여 캐시합니다."""
        if self._org_loaded:
            return
        self._org_loaded = True

        if not self.supabase_client:
            logger.warning("[WARN] Supabase client unavailable: org/agent mapping will be skipped.")
            return

        # 1) organization chart (teams + members)
        try:
            org = (
                self.supabase_client.table("configuration")
                .select("uuid,value")
                .eq("key", "organization")
                .eq("tenant_id", tenant_id)
                .execute()
            )
            if org.data and len(org.data) > 0:
                self._org_config_uuid = org.data[0].get("uuid")
                value = org.data[0].get("value")
                value = self._safe_json_loads(value)
                if isinstance(value, dict):
                    self._org_value = value
                    chart = value.get("chart") or value
                    if isinstance(chart, dict):
                        self._org_chart = chart
                        idx = self._index_org_chart(chart)
                        self._org_teams_by_name = idx.get("teams_by_name") or {}
                        self._org_team_name_by_id = idx.get("team_name_by_id") or {}
                        self._org_members_by_team_id = idx.get("members_by_team_id") or {}
            logger.info(
                f"[ASSIGN] org loaded: tenant_id={tenant_id!r} chart={'yes' if isinstance(self._org_chart, dict) else 'no'} "
                f"teams={len(self._org_teams_by_name or {})} members_teams={len(self._org_members_by_team_id or {})}"
            )
        except Exception as e:
            logger.warning(f"[WARN] organization 로드 실패: {e}")

        # 2) users (agents + humans)
        try:
            users = (
                self.supabase_client.table("users")
                .select("id, username, role, endpoint, agent_type, alias, is_agent, email, goal, persona, description, tools, skills, model")
                .eq("tenant_id", tenant_id)
                .execute()
            )
            self._users = users.data or []
            self._agents = [u for u in self._users if isinstance(u, dict) and u.get("is_agent") is True]
            logger.info(f"[ASSIGN] users loaded: tenant_id={tenant_id!r} users={len(self._users)} agents={len(self._agents)}")
        except Exception as e:
            logger.warning(f"[WARN] users 로드 실패: {e}")

    def _pick_agent_for_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """역할명으로 users(is_agent=true) 중 가장 잘 맞는 agent를 선택."""
        key = self._normalize_text_key(role_name)
        if not key:
            return None

        # exact-ish match priority: username / role / alias
        for a in self._agents:
            if not isinstance(a, dict):
                continue
            if self._normalize_text_key(a.get("username")) == key:
                return a
            if self._normalize_text_key(a.get("role")) == key:
                return a
            if self._normalize_text_key(a.get("alias")) == key:
                return a

        # contains match
        for a in self._agents:
            if not isinstance(a, dict):
                continue
            cand = self._normalize_text_key(a.get("username")) or ""
            if cand and (cand in key or key in cand):
                return a
            cand = self._normalize_text_key(a.get("role")) or ""
            if cand and (cand in key or key in cand):
                return a
            cand = self._normalize_text_key(a.get("alias")) or ""
            if cand and (cand in key or key in cand):
                return a

        return None

    def _pick_user_for_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """
        역할명으로 users 전체(에이전트+사용자) 중 가장 잘 맞는 사용자를 선택합니다.
        우선순위:
        1) agent 먼저 매칭
        2) 그 다음 일반 사용자 매칭
        """
        key = self._normalize_text_key(role_name)
        if not key:
            return None

        def match_in(pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            # exact-ish match priority: username / role / alias
            for u in pool:
                if not isinstance(u, dict):
                    continue
                if self._normalize_text_key(u.get("username")) == key:
                    return u
                if self._normalize_text_key(u.get("role")) == key:
                    return u
                if self._normalize_text_key(u.get("alias")) == key:
                    return u
            # contains match
            for u in pool:
                if not isinstance(u, dict):
                    continue
                cand = self._normalize_text_key(u.get("username")) or ""
                if cand and (cand in key or key in cand):
                    return u
                cand = self._normalize_text_key(u.get("role")) or ""
                if cand and (cand in key or key in cand):
                    return u
                cand = self._normalize_text_key(u.get("alias")) or ""
                if cand and (cand in key or key in cand):
                    return u
            return None

        agent_hit = match_in(self._agents)
        if agent_hit:
            return agent_hit
        return match_in(self._users)

    def _get_org_team_candidates(self, role_name: str) -> List[Dict[str, Any]]:
        """역할명에 대해 후보 팀을 가볍게 필터링(LLM 입력 토큰 절약용)."""
        key = self._normalize_text_key(role_name)
        if not key:
            return []
        out: List[Dict[str, Any]] = []
        for norm_name, team_id in (self._org_teams_by_name or {}).items():
            if not team_id:
                continue
            if norm_name and (norm_name in key or key in norm_name):
                out.append({"team_id": team_id, "team_name": self._org_team_name_by_id.get(team_id) or ""})
        # IMPORTANT: do NOT provide unrelated fallback teams.
        # It increases token usage and often makes the model return action=none with confidence=0.
        return out[:30]

    def _get_user_candidates(self, role_name: str) -> List[Dict[str, Any]]:
        """역할명에 대해 후보 에이전트(users.is_agent=true)만 가볍게 필터링(LLM 입력 토큰 절약용).

        IMPORTANT:
        - 후보는 users 테이블에 존재하는 '에이전트'만이어야 한다.
        - 사람 사용자(is_agent=false)는 후보에 포함하지 않는다.
        """
        key = self._normalize_text_key(role_name)
        if not key:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        # Candidate pool is agents only
        for u in (self._agents or []):
            if not isinstance(u, dict) or not u.get("id"):
                continue
            uname = self._normalize_text_key(u.get("username")) or ""
            urole = self._normalize_text_key(u.get("role")) or ""
            ualias = self._normalize_text_key(u.get("alias")) or ""
            # NOTE: LLM이 "태스크 설명 ↔ 에이전트 설명" 매칭을 하려면
            # 에이전트의 description/goal/persona가 후보로 제공되어야 한다.
            # (하지만 토큰 절약을 위해 slim 단계에서만 포함/절단한다)
            udesc = self._normalize_text_key(u.get("description")) or ""
            ugoal = self._normalize_text_key(u.get("goal")) or ""
            upersona = self._normalize_text_key(u.get("persona")) or ""
            score = 0.0
            # 역할명/태스크명은 보통 username/role/alias에 가장 잘 걸리지만,
            # 최근 생성된 에이전트는 description/goal에만 힌트가 있는 경우가 있어 포함한다.
            for cand in (uname, urole, ualias, udesc, ugoal, upersona):
                if not cand:
                    continue
                if cand == key:
                    score = max(score, 1.0)
                elif cand in key or key in cand:
                    score = max(score, 0.8)
                elif any(tok and tok in cand for tok in (key[:3], key[-3:])) and len(key) >= 3:
                    score = max(score, 0.5)
            if score > 0:
                scored.append((score, u))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [u for _, u in scored[:30]]

        # if nothing matched, do NOT provide unrelated fallback humans; provide top agents only.
        if not picked:
            picked = (self._agents or [])[:20]
        # minimize fields
        slim: List[Dict[str, Any]] = []
        for u in picked:
            if not isinstance(u, dict):
                continue
            # LLM이 태스크 설명과 에이전트 설명을 비교할 수 있도록
            # (특히 에이전트의) 텍스트 프로필 일부를 제공한다.
            is_agent = True  # pool is agents only
            desc = str(u.get("description") or "").strip()
            goal = str(u.get("goal") or "").strip()
            persona = str(u.get("persona") or "").strip()
            if len(desc) > 220:
                desc = desc[:220] + "…"
            if len(goal) > 180:
                goal = goal[:180] + "…"
            if len(persona) > 220:
                persona = persona[:220] + "…"
            slim.append(
                {
                    "id": str(u.get("id") or ""),
                    "username": str(u.get("username") or ""),
                    "role": str(u.get("role") or ""),
                    "alias": str(u.get("alias") or ""),
                    "is_agent": is_agent,
                    "agent_type": str(u.get("agent_type") or ""),
                    "description": desc,
                    "goal": goal,
                    "persona": persona,
                }
            )
        return slim[:30]

    async def _call_openai_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """OpenAI 호출을 통해 JSON 객체를 반환(실패 시 None)."""
        if not self.openai_client:
            return None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            def _run():
                # Prefer JSON mode when supported; fallback gracefully if SDK/model doesn't support it.
                try:
                    return self.openai_client.chat.completions.create(
                        model=(model or self.openai_model),
                        messages=messages,
                        temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")) if temperature is None else float(temperature),
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    return self.openai_client.chat.completions.create(
                        model=(model or self.openai_model),
                        messages=messages,
                        temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")) if temperature is None else float(temperature),
                        max_tokens=max_tokens,
                    )

            # 원복: 헤지/하드타임아웃 없이 응답 완료까지 대기
            resp = await asyncio.to_thread(_run)
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return None
            parsed = self._parse_json_response_content(content)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception as e:
            logger.warning(f"[WARN] OpenAI JSON call failed: {e}")
            return None

    def _extract_json_block_from_markdown(self, text: str) -> Optional[str]:
        """LLM 응답에서 ``` ``` 코드블록(JSON)을 추출합니다."""
        if not text:
            return None
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            return (m.group(1) or "").strip()
        # fallback: try to find first '{'..last '}' span
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return text[s : e + 1].strip()
        return None

    def _parse_json_response_content(self, content: str) -> Optional[Any]:
        """
        JSON 파싱 보강:
        - raw json
        - fenced json
        - first '{' ~ last '}' span
        - 개행/탭 정규화 재시도
        """
        raw = (content or "").strip()
        if not raw:
            return None

        candidates: List[str] = [raw]
        block = self._extract_json_block_from_markdown(raw)
        if block and block not in candidates:
            candidates.append(block)

        if "{" in raw and "}" in raw:
            s = raw.find("{")
            e = raw.rfind("}")
            if e > s:
                span = raw[s : e + 1].strip()
                if span and span not in candidates:
                    candidates.append(span)

        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                pass
            try:
                c2 = re.sub(r"[\r\n\t]+", " ", cand)
                c2 = re.sub(r"\s{2,}", " ", c2).strip()
                return json.loads(c2)
            except Exception:
                pass
        return None

    async def _call_openai_process_definition(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int = 3500,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """프로세스 정의 생성용: JSON-only 출력 강제 + 파싱 보강.

        temperature 가 None 이면 환경변수 기본값을 쓴다(호출부 재시도 시 값 지정 가능).
        """
        if not self.openai_client:
            return None

        def _loads_with_newlines_removed(s: str) -> Any:
            """
            JSON 파싱 보강:
            - 정상 JSON은 json.loads로 바로 파싱됨(포맷팅 개행 포함 OK)
            - 하지만 LLM이 문자열 값 내부에 '실제 개행 문자'를 넣으면 JSON이 깨짐
              → 파싱 실패 시 \r/\n/\t 등을 공백으로 치환 후 재시도
            - 응답이 중간에서 잘린(truncated) 경우가 있어, 아래를 추가로 시도:
              1) 마지막 닫는 중괄호/대괄호까지 잘라서 파싱 가능한 최대 prefix를 찾기
              2) 괄호 수가 모자라는 경우(명백히 끝만 잘린 경우) 자동으로 닫아서 재시도
            """
            def _try_load(raw: str) -> Any:
                return json.loads(raw)

            def _sanitize_whitespace(raw: str) -> str:
                # Normalize raw newlines/tabs that sometimes appear inside string values.
                s2 = re.sub(r"[\r\n\t]+", " ", raw)
                s2 = re.sub(r"\s{2,}", " ", s2).strip()
                return s2

            def _best_effort_trim_to_json_prefix(raw: str) -> Optional[str]:
                """
                JSON이 뒤에서 잘린 경우(특히 로깅/전송/모델 출력 이슈),
                마지막에 "완전한 객체"로 끝나는 prefix를 찾아 파싱을 시도합니다.
                - 문자열 리터럴/escape를 고려한 간단 스캐너
                """
                start = raw.find("{")
                if start < 0:
                    return None
                in_str = False
                esc = False
                depth_obj = 0
                depth_arr = 0
                last_ok_end = None
                for i in range(start, len(raw)):
                    ch = raw[i]
                    if in_str:
                        if esc:
                            esc = False
                            continue
                        if ch == "\\":
                            esc = True
                            continue
                        if ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == "{":
                            depth_obj += 1
                        elif ch == "}":
                            depth_obj = max(0, depth_obj - 1)
                        elif ch == "[":
                            depth_arr += 1
                        elif ch == "]":
                            depth_arr = max(0, depth_arr - 1)
                        # 최상위 객체가 닫히는 지점 기록
                        if depth_obj == 0 and depth_arr == 0 and ch == "}":
                            last_ok_end = i
                if last_ok_end is not None:
                    return raw[start : last_ok_end + 1].strip()
                return None

            def _autoclose_brackets_if_obvious(raw: str) -> Optional[str]:
                """
                문자열 상태를 고려한 괄호 카운팅으로, 끝부분이 잘린 케이스에 한해
                부족한 ]/}를 뒤에 붙여 파싱을 시도합니다.
                """
                start = raw.find("{")
                if start < 0:
                    return None
                in_str = False
                esc = False
                opens_obj = 0
                closes_obj = 0
                opens_arr = 0
                closes_arr = 0
                for ch in raw[start:]:
                    if in_str:
                        if esc:
                            esc = False
                            continue
                        if ch == "\\":
                            esc = True
                            continue
                        if ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == "{":
                            opens_obj += 1
                        elif ch == "}":
                            closes_obj += 1
                        elif ch == "[":
                            opens_arr += 1
                        elif ch == "]":
                            closes_arr += 1
                # 문자열이 열린 채로 끝났으면(따옴표 미종료) auto-close는 위험해서 포기
                if in_str:
                    return None
                need_arr = max(0, opens_arr - closes_arr)
                need_obj = max(0, opens_obj - closes_obj)
                if need_arr == 0 and need_obj == 0:
                    return None
                # 배열을 먼저 닫고 객체를 닫는 것이 일반적으로 안전
                return (raw.strip() + ("]" * need_arr) + ("}" * need_obj)).strip()

            # 1) Raw parse
            try:
                return _try_load(s)
            except json.JSONDecodeError:
                pass

            # 2) Whitespace sanitize parse
            s2 = _sanitize_whitespace(s)
            try:
                return _try_load(s2)
            except json.JSONDecodeError:
                pass

            # 3) Trim to best valid JSON prefix
            trimmed = _best_effort_trim_to_json_prefix(s2)
            if trimmed:
                try:
                    return _try_load(trimmed)
                except json.JSONDecodeError:
                    pass

            # 4) Auto-close obvious missing brackets/braces
            closed = _autoclose_brackets_if_obvious(s2)
            if closed:
                return _try_load(closed)

            # Give up: let caller handle retry path
            return _try_load(s2)        # Prefer deterministic output for strict JSON parsing.
        if temperature is None:
            temperature = float(
                os.getenv(
                    "LLM_PROCESS_DEFINITION_TEMPERATURE",
                    os.getenv("LLM_PROCESS_TEMPERATURE", "0.0"),
                )
            )
        else:
            temperature = float(temperature)
        try:
            def _run():
                # Prefer JSON mode when supported; fallback gracefully if SDK/model doesn't support it.
                try:
                    return self.openai_client.chat.completions.create(
                        model=self.process_definition_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    return self.openai_client.chat.completions.create(
                        model=self.process_definition_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

            # 원복: 헤지/하드타임아웃 없이 응답 완료까지 대기
            resp = await asyncio.to_thread(_run)
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                logger.warning(
                    f"[PROCDEF][LLM] empty content returned (model={self.process_definition_model})"
                )
                return None

            # First try: content itself is JSON (when response_format=json_object worked)
            try:
                parsed = _loads_with_newlines_removed(content)
                if isinstance(parsed, dict):
                    elems = parsed.get("elements")
                    elems_len = len(elems) if isinstance(elems, list) else None
                    logger.info(
                        f"[PROCDEF][LLM] parsed ok (keys={list(parsed.keys())}, elements_len={elems_len})"
                    )
                else:
                    logger.warning(
                        f"[PROCDEF][LLM] parsed non-dict JSON (type={type(parsed).__name__})"
                    )
                return parsed
            except json.JSONDecodeError:
                # Fallback: attempt to recover JSON from markdown/codefence responses
                json_block = self._extract_json_block_from_markdown(content) or ""
                if not json_block:
                    logger.warning(
                        "[PROCDEF][LLM] JSON decode failed and no JSON block recovered "
                        f"(content_preview={content[:300]!r})"
                    )
                    return None
                try:
                    parsed = _loads_with_newlines_removed(json_block)
                    if isinstance(parsed, dict):
                        elems = parsed.get("elements")
                        elems_len = len(elems) if isinstance(elems, list) else None
                        logger.info(
                            f"[PROCDEF][LLM] parsed ok from recovered block (keys={list(parsed.keys())}, elements_len={elems_len})"
                        )
                    else:
                        logger.warning(
                            f"[PROCDEF][LLM] parsed non-dict JSON from recovered block (type={type(parsed).__name__})"
                        )
                    return parsed
                except json.JSONDecodeError:
                    logger.warning(
                        "[PROCDEF][LLM] JSON decode failed even after block recovery "
                        f"(block_preview={json_block[:300]!r}, content_preview={content[:300]!r})"
                    )
                    return None
        except Exception as e:
            logger.exception(
                f"[WARN] OpenAI process-definition call failed "
                f"(model={self.process_definition_model}): {type(e).__name__}: {e}"
            )
            return None

    async def _call_openai_json_messages(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int = 1200,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """OpenAI 호출을 통해 JSON 객체를 반환(메시지 배열 직접 전달)."""
        if not self.openai_client:
            return None
        try:
            def _run():
                # Prefer JSON mode when supported; fallback gracefully if SDK/model doesn't support it.
                try:
                    return self.openai_client.chat.completions.create(
                        model=(model or self.openai_model),
                        messages=messages,
                        temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")) if temperature is None else float(temperature),
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    return self.openai_client.chat.completions.create(
                        model=(model or self.openai_model),
                        messages=messages,
                        temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")) if temperature is None else float(temperature),
                        max_tokens=max_tokens,
                    )

            # 원복: 헤지/하드타임아웃 없이 응답 완료까지 대기
            resp = await asyncio.to_thread(_run)
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return None
            parsed = self._parse_json_response_content(content)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception as e:
            logger.warning(
                f"[WARN] OpenAI JSON(messages) call failed: {type(e).__name__}: {e} "
                f"(model={model or self.openai_model})"
            )
            return None

    async def _generate_process_outline_via_consulting_prompt(
        self,
        *,
        process_name: str,
        user_request: str,
        extracted: Dict[str, Any],
        hints_simplified: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        (프롬프트 개선) ProcessConsultingGenerator.js 시스템 프롬프트를 그대로 사용해
        '말로 된 프로세스 초안'을 먼저 생성합니다.
        - 출력(JSON)은 {content, answerType} 구조를 기대합니다.
        - 반환값은 content(markdown)만 추출합니다.
        """
        if not self.openai_client:
            return None

        system_consulting = get_process_consulting_system_prompt()
        system_guard = (
            "위 시스템 지시를 그대로 따르되, 이 호출에서는 고객에게 추가 질문을 하지 말고\n"
            "반드시 아래 JSON 형식으로만 응답하세요(JSON only, code fence 금지):\n"
            '{ "content": "...", "answerType": "consulting" }\n'
            "- content에는 '프로세스 초안'을 반드시 1. 2. 3. 번호 목록으로 작성하고, 흐름은 → 를 사용하세요.\n"
        )

        payload = {
            "process_name": process_name,
            "user_request": user_request,
            "extracted": extracted,
            "assignment_hints": hints_simplified or {},
        }
        user_prompt = (
            "아래 정보를 바탕으로 사용자가 만들고자 하는 비즈니스 프로세스의 **초안**을 작성하세요.\n"
            "- 시스템/도구/프로그램을 무엇을 쓰는지 묻지 마세요.\n"
            "- 답변은 JSON만 반환해야 합니다.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
        )

        obj = await self._call_openai_json_messages(
            messages=[
                {"role": "system", "content": system_consulting},
                {"role": "system", "content": system_guard},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            model=self.process_definition_model,
            temperature=float(os.getenv("LLM_PROCESS_TEMPERATURE", "0.2")),
        )
        if not isinstance(obj, dict):
            return None
        content = str(obj.get("content") or "").strip()
        return content or None

    def _generate_bpmn_xml_backend(
        self,
        *,
        model: Dict[str, Any],
        horizontal: Optional[bool] = None,
    ) -> Optional[str]:
        """백엔드에서 ProcessGPTBPMNXmlGenerator로 BPMN XML 생성."""
        try:
            return self._processgpt_bpmn_xml_generator.create_bpmn_xml(model, horizontal=horizontal)
        except Exception as e:
            logger.warning(f"[WARN] BPMN xml generation failed: {e}")
            return None

    def _elements_model_to_runtime_definition(self, elements_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        ProcessDefinitionGenerator(elements 기반) 출력 → proc_def.definition(activities/events/gateways/sequences 기반)으로 변환.
        (폼 생성/실행/UI 호환을 위해 런타임 구조를 사용)
        """
        out: Dict[str, Any] = {}
        for k in ("megaProcessId", "majorProcessId", "processDefinitionName", "processDefinitionId", "description", "isHorizontal"):
            if k in elements_model:
                out[k] = elements_model.get(k)

        out["data"] = elements_model.get("data") or []
        out["roles"] = elements_model.get("roles") or []
        out["events"] = []
        out["activities"] = []
        out["gateways"] = []
        out["sequences"] = []
        out["subProcesses"] = elements_model.get("subProcesses") or []
        out["participants"] = elements_model.get("participants") or []

        elems = elements_model.get("elements") or []
        if not isinstance(elems, list):
            return out

        def gw_type_map(t: str) -> str:
            t = (t or "").strip()
            if t.lower() in ("exclusivegateway", "exclusive_gateway"):
                return "exclusiveGateway"
            if t.lower() in ("parallelgateway", "parallel_gateway"):
                return "parallelGateway"
            if t.lower() in ("inclusivegateway", "inclusive_gateway"):
                return "inclusiveGateway"
            return t or "exclusiveGateway"

        for e in elems:
            if not isinstance(e, dict):
                continue
            et = str(e.get("elementType") or "").strip()
            if et.lower() == "event":
                t = str(e.get("type") or "").strip()
                if t == "StartEvent":
                    rt = "startEvent"
                elif t == "EndEvent":
                    rt = "endEvent"
                else:
                    rt = "intermediateCatchEvent"
                out["events"].append(
                    {
                        "id": e.get("id"),
                        "name": e.get("name") or "",
                        "role": e.get("role") or "",
                        "type": rt,
                        "process": out.get("processDefinitionId") or "",
                        "properties": "{}",
                        "description": e.get("description") or "",
                        "trigger": e.get("trigger") or "",
                    }
                )
            elif et.lower() == "activity":
                # element.type is "UserActivity" in ProcessGPT mode
                out["activities"].append(
                    {
                        "id": e.get("id"),
                        "name": e.get("name") or "",
                        "role": e.get("role") or "",
                        "tool": e.get("tool") or "",
                        "type": "userTask",
                        "process": out.get("processDefinitionId") or "",
                        "duration": int(e.get("duration") or 5) if str(e.get("duration") or "").isdigit() else 5,
                        "inputData": e.get("inputData") or [],
                        "outputData": e.get("outputData") or [],
                        "properties": "{}",
                        "description": e.get("description") or "",
                        "instruction": e.get("instruction") or "",
                        "skills": e.get("skills") or [],
                        "attachedEvents": None,
                        # agent fields will be filled later
                        "agent": None,
                        "agentMode": "none",
                        "orchestration": None,
                        "attachments": [],
                        "checkpoints": e.get("checkpoints") or [],
                    }
                )
            elif et.lower() == "gateway":
                out["gateways"].append(
                    {
                        "id": e.get("id"),
                        "name": e.get("name") or "",
                        "role": e.get("role") or "",
                        "type": gw_type_map(str(e.get("type") or "")),
                        "process": out.get("processDefinitionId") or "",
                        "condition": "",
                        "properties": "{}",
                        "description": e.get("description") or "",
                    }
                )
            elif et.lower() == "sequence":
                out["sequences"].append(
                    {
                        "id": e.get("id"),
                        "name": e.get("name") or "",
                        "source": e.get("source"),
                        "target": e.get("target"),
                        "condition": e.get("condition") or "",
                        "properties": "{}",
                    }
                )

        return out

    def _simplify_assignment_hints(self, hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        (t2) 역할/유저 매핑 결과를 프롬프트/로그/저장에 쓰기 좋은 "간소화 JSON"으로 변환.

        Shape:
          {
            "roles": {
              "<roleName>": {"endpoint": "<id or ''>", "default": "<id or ''>", "origin": "..."}
            },
            "activities": {
              "<activityId>": {"role": "...", "agent": "<userId or ''>", "agentMode": "draft|none", "orchestration": "crewai-action|"}
            }
          }
        """
        roles_out: Dict[str, Any] = {}
        acts_out: Dict[str, Any] = {}

        for r in (hints.get("roles") or []):
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            endpoint = ""
            default = ""
            ep = r.get("endpoint")
            df = r.get("default")
            if isinstance(ep, list) and ep:
                endpoint = str(ep[0])
            elif isinstance(ep, str):
                endpoint = ep
            if isinstance(df, list) and df:
                default = str(df[0])
            elif isinstance(df, str):
                default = df
            roles_out[name] = {
                "endpoint": endpoint,
                "default": default,
                "origin": str(r.get("origin") or ""),
            }

        for a in (hints.get("activities") or []):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if not aid:
                continue
            acts_out[aid] = {
                "role": str(a.get("role") or "").strip(),
                "agent": str(a.get("agent") or "").strip(),
                "agentMode": str(a.get("agentMode") or "").strip(),
                "orchestration": str(a.get("orchestration") or "").strip(),
            }

        return {"roles": roles_out, "activities": acts_out}

    def _snake_id(self, s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _should_fallback_to_extracted_elements(
        self,
        *,
        elements_model: Dict[str, Any],
        extracted: Dict[str, Any],
        process_name: str,
    ) -> tuple[bool, str]:
        """LLM 결과가 추출 데이터 대비 지나치게 빈약하면 extracted 기반 생성으로 폴백."""
        elems = elements_model.get("elements")
        if not isinstance(elems, list):
            return True, "elements_missing_or_not_list"

        llm_events = 0
        llm_activities = 0
        llm_gateways = 0
        llm_sequences = 0
        for e in elems:
            if not isinstance(e, dict):
                continue
            et = str(e.get("elementType") or "").strip().lower()
            if et == "event":
                llm_events += 1
            elif et == "activity":
                llm_activities += 1
            elif et == "gateway":
                llm_gateways += 1
            elif et == "sequence":
                llm_sequences += 1

        ex_tasks = extracted.get("tasks") or extracted.get("activities") or []
        ex_roles = extracted.get("roles") or []
        ex_gateways = extracted.get("gateways") or []
        ex_events = extracted.get("events") or []
        ex_flows = extracted.get("sequence_flows") or extracted.get("flows") or []

        ex_tasks_n = len(ex_tasks) if isinstance(ex_tasks, list) else 0
        ex_roles_n = len(ex_roles) if isinstance(ex_roles, list) else 0
        ex_gateways_n = len(ex_gateways) if isinstance(ex_gateways, list) else 0
        ex_events_n = len(ex_events) if isinstance(ex_events, list) else 0
        ex_flows_n = len(ex_flows) if isinstance(ex_flows, list) else 0

        # extracted gateway 중 '실제로 시퀀스 흐름에 연결된' 것만 따로 센다.
        # 추출 단계가 같은 의사결정에 대해 중복/고립 게이트웨이를 만드는 사례가 있어
        # (예: 'X 여부 판단' 게이트웨이 + 'X 에 따라 분기' 게이트웨이를 별도로),
        # ex_gateways_n 전체로 비교하면 LLM 이 올바르게 통합·정리한 결과를
        # gateway_count_severely_reduced 로 오판해 불필요하게 fallback 한다.
        _ex_gw_ids = set()
        for _g in (ex_gateways if isinstance(ex_gateways, list) else []):
            if isinstance(_g, dict):
                _gid = str(_g.get("gateway_id") or _g.get("id") or "").strip()
                if _gid:
                    _ex_gw_ids.add(_gid)
        _wired_gw_ids = set()
        for _f in (ex_flows if isinstance(ex_flows, list) else []):
            if not isinstance(_f, dict):
                continue
            for _k in ("from_id", "to_id", "source", "target"):
                _v = str(_f.get(_k) or "").strip()
                if _v and _v in _ex_gw_ids:
                    _wired_gw_ids.add(_v)
        ex_gateways_wired_n = len(_wired_gw_ids)

        if ex_tasks_n > 0 and llm_activities == 0:
            return True, "no_activities_while_extracted_has_tasks"
        if ex_gateways_wired_n >= 2 and llm_gateways == 0:
            return True, "no_gateways_while_extracted_has_gateways"
        if ex_flows_n > 2 and llm_sequences <= 1:
            return True, "no_sequences_while_extracted_has_flows"
        if ex_events_n >= 4 and llm_events <= 2 and llm_activities <= 1:
            return True, "event_count_collapsed_to_start_end_only"

        # NEW: gateway/sequence 의 "심각한 수축" 도 fallback.
        # - LLM 이 게이트웨이를 만들긴 했지만 절반 이하로 줄어들면, 분기/병합 의미가 사라져
        #   본 문서의 흐름을 보존하지 못한다. validator 의 single-branch gateway collapse 가
        #   이어서 동작하면서 게이트웨이가 0 까지 떨어지는 회귀가 관측되었기 때문에,
        #   이 시점에서 미리 컷한다.
        # - 단, '연결된(wired)' 게이트웨이 수로 비교한다. 추출 단계의 중복/고립 게이트웨이를
        #   세면 LLM 의 정상적인 dedup 을 오판하기 때문이다.
        if ex_gateways_wired_n >= 2 and llm_gateways * 2 < ex_gateways_wired_n:
            return True, (
                f"gateway_count_severely_reduced({llm_gateways}<{ex_gateways_wired_n}/2 "
                f"wired; extracted_total={ex_gateways_n})"
            )
        # - 마찬가지로 sequence 도 절반 이하이면 분기/병합/역방향 흐름이 다수 누락된 것으로
        #   판단해 fallback. (LLM 이 모든 노드를 직선 chain 으로 만들어버린 회귀 사례 대응)
        if ex_flows_n >= 6 and llm_sequences * 2 < ex_flows_n:
            return True, f"sequence_count_severely_reduced({llm_sequences}<{ex_flows_n}/2)"

        # NEW: task 시작점 무결성 — extracted.sequence_flows 에서 유일한 in-degree 0 task 가
        #      식별되는데, LLM elements 의 시작 task (sequence 에서 source 로만 등장하고 target 으로는
        #      등장하지 않는 activity) 의 이름이 그와 명백히 다르면, LLM 이 task 순서를 잘못 잡은
        #      hallucination 으로 본다. (예: "최종 결과 통보" 가 첫 task 가 되는 회귀)
        try:
            mismatch_reason = self._llm_start_task_mismatches_extracted(
                elements_model=elements_model,
                extracted=extracted,
                ex_tasks=ex_tasks if isinstance(ex_tasks, list) else [],
                ex_flows=ex_flows if isinstance(ex_flows, list) else [],
            )
            if mismatch_reason:
                return True, mismatch_reason
        except Exception:
            # 검증 실패는 정책상 보수적으로 무시 (LLM 결과 유지)
            pass

        logger.info(
            f"[PROCDEF][FALLBACK-CHECK] keep_llm process={process_name!r} "
            f"llm(events={llm_events},activities={llm_activities},gateways={llm_gateways},sequences={llm_sequences}) "
            f"extracted(tasks={ex_tasks_n},roles={ex_roles_n},gateways={ex_gateways_n}"
            f"[wired={ex_gateways_wired_n}],events={ex_events_n},flows={ex_flows_n})"
        )
        return False, "ok"

    def _llm_start_task_mismatches_extracted(
        self,
        *,
        elements_model: Dict[str, Any],
        extracted: Dict[str, Any],
        ex_tasks: List[Any],
        ex_flows: List[Any],
    ) -> Optional[str]:
        """LLM 결과의 "시작 activity" 가 extracted 의 시작 task 와 일치하는지 검사.

        검증 트리거 조건:
          - extracted.sequence_flows 가 충분히 있고 (>= 3)
          - sequence_flows 에서 한 번도 to_id 로 등장하지 않는 task id 가 정확히 1개일 때

        그 1개 시작 task 의 이름과, LLM elements 의 시작 activity (Sequence 에서 source 로만
        등장하고 target 으로 등장하지 않는 Activity) 의 이름을 비교한다.
        둘 다 식별되는데 이름이 명백히 다르면 (substring 매칭도 실패) hallucination 으로 본다.

        Returns:
          None: 검증 트리거 조건 불충족 또는 일치 → fallback 사유 없음
          str:  fallback 사유 (예: "start_task_mismatch(...)")
        """
        if not isinstance(ex_flows, list) or len(ex_flows) < 3:
            return None

        # extracted: task id -> name map (task type 만)
        ex_task_name_by_id: Dict[str, str] = {}
        for t in ex_tasks or []:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("task_id") or t.get("id") or "").strip()
            tname = str(t.get("name") or "").strip()
            if tid and tname:
                ex_task_name_by_id[tid] = tname
        if not ex_task_name_by_id:
            return None

        # extracted: task id 중 in-degree 0 (어디서도 to_id 로 안 가리키는 것)
        in_targets: set[str] = set()
        all_task_sources: set[str] = set()
        for f in ex_flows or []:
            if not isinstance(f, dict):
                continue
            ft = str(f.get("from_type") or "task").strip().lower()
            tt = str(f.get("to_type") or "task").strip().lower()
            fid = str(f.get("from_id") or f.get("from_task_id") or "").strip()
            tid = str(f.get("to_id") or f.get("to_task_id") or "").strip()
            if ft == "task" and fid:
                all_task_sources.add(fid)
            if tt == "task" and tid:
                in_targets.add(tid)

        starts = [
            tid for tid in ex_task_name_by_id
            if tid in all_task_sources and tid not in in_targets
        ]
        if len(starts) != 1:
            return None
        ex_start_name = ex_task_name_by_id[starts[0]]

        # LLM: Sequence 에서 source 로만 등장하고 target 으로 안 나오는 Activity
        llm_elems = elements_model.get("elements") or []
        if not isinstance(llm_elems, list):
            return None
        activity_name_by_id: Dict[str, str] = {}
        seq_sources: set[str] = set()
        seq_targets: set[str] = set()
        for e in llm_elems:
            if not isinstance(e, dict):
                continue
            et = str(e.get("elementType") or "").strip().lower()
            eid = str(e.get("id") or "").strip()
            if et == "activity" and eid:
                activity_name_by_id[eid] = str(e.get("name") or "").strip()
            elif et == "sequence":
                s = str(e.get("source") or "").strip()
                t = str(e.get("target") or "").strip()
                if s:
                    seq_sources.add(s)
                if t:
                    seq_targets.add(t)
        if not activity_name_by_id:
            return None
        llm_starts = [
            aid for aid in activity_name_by_id
            if aid in seq_sources and aid not in seq_targets
        ]
        # LLM 의 시작 activity 가 0 이거나 2 이상이면 다른 검증(sequence 손실)에 맡긴다
        if len(llm_starts) != 1:
            return None
        llm_start_name = activity_name_by_id[llm_starts[0]]
        if not ex_start_name or not llm_start_name:
            return None

        a = ex_start_name.lower().strip()
        b = llm_start_name.lower().strip()
        if a == b:
            return None
        # 부분 일치는 허용 (예: LLM 이 이름을 약간 다듬은 경우)
        if (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a):
            return None
        return f"start_task_mismatch(extracted='{ex_start_name}', llm='{llm_start_name}')"

    def _augment_sequence_flows_for_isolated_tasks(
        self,
        sorted_tasks: List[Any],
        raw_flows: List[Any],
    ) -> List[Any]:
        """task_order 가 부여된 sorted_tasks 중 sequence_flows 에 한 번도 안 나타나는
        isolated task 들을 task_order 인접 task 와 연결하는 backbone sequence 들을 추가.

        - (I) GLOBAL-ORDER 가 task_order 를 정확히 부여한 후에도 추출 LLM 이 분기 sequence
          를 누락하면 일부 task (예: "휴가 신청서 접수") 가 isolated 로 남는다. 이 경우 LLM
          이 BPMN 시작점을 다른 task 로 선택하는 회귀가 발생한다.
        - source text 위치 기반 task_order 만 신뢰 가능한 truth source 이므로, isolated 한
          task 들을 task_order 인접 task 와 연결하여 흐름의 backbone 을 회복한다.
        - 기존 sequence_flows 는 그대로 보존. 보강은 isolated task 가 양쪽 (prev/next) 와
          연결되지 않은 경우에만 적용 (중복 방지).
        """
        if not isinstance(sorted_tasks, list) or len(sorted_tasks) < 2:
            return raw_flows if isinstance(raw_flows, list) else []
        flows = list(raw_flows) if isinstance(raw_flows, list) else []

        task_ids_in_order: List[str] = []
        for t in sorted_tasks:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("task_id") or t.get("id") or "").strip()
            if tid:
                task_ids_in_order.append(tid)
        if len(task_ids_in_order) < 2:
            return flows

        in_count: Dict[str, int] = {tid: 0 for tid in task_ids_in_order}
        out_count: Dict[str, int] = {tid: 0 for tid in task_ids_in_order}
        existing_pairs: Set[Tuple[str, str]] = set()
        for f in flows:
            if not isinstance(f, dict):
                continue
            src = str(
                f.get("source") or f.get("from_id") or f.get("from_task_id") or ""
            ).strip()
            tgt = str(
                f.get("target") or f.get("to_id") or f.get("to_task_id") or ""
            ).strip()
            if src in out_count:
                out_count[src] += 1
            if tgt in in_count:
                in_count[tgt] += 1
            if src and tgt:
                existing_pairs.add((src, tgt))

        last_index = len(task_ids_in_order) - 1

        # (M) 완전 isolated (in==0 AND out==0) 인 task — 양쪽 모두 끊김.
        isolated_ids = {
            tid for tid in task_ids_in_order
            if in_count.get(tid, 0) == 0 and out_count.get(tid, 0) == 0
        }
        # (N) section 경계 gap task — in 또는 out 한 쪽만 끊긴 task.
        #     - in==0 이지만 task_order 가 첫번째가 아닌 task: 이전 section 의 어떤 task 에서도
        #       이 task 로 흐름이 오지 않음 → 이전 task_order task 에서 들어오게 보강.
        #     - out==0 이지만 task_order 가 마지막이 아닌 task: 이 task 에서 다음 section 으로
        #       나가는 흐름이 없음 → 다음 task_order task 로 나가게 보강.
        gap_in_ids: Set[str] = {
            tid for i, tid in enumerate(task_ids_in_order)
            if i > 0 and tid not in isolated_ids and in_count.get(tid, 0) == 0
        }
        gap_out_ids: Set[str] = {
            tid for i, tid in enumerate(task_ids_in_order)
            if i < last_index and tid not in isolated_ids and out_count.get(tid, 0) == 0
        }

        if not isolated_ids and not gap_in_ids and not gap_out_ids:
            return flows

        added: List[Dict[str, Any]] = []

        def _emit(prev_id: str, this_id: str, tag: str) -> None:
            if (prev_id, this_id) in existing_pairs or prev_id == this_id:
                return
            added.append({
                "source": prev_id,
                "target": this_id,
                "from_id": prev_id,
                "to_id": this_id,
                "from_type": "task",
                "to_type": "task",
                "condition": "",
                "name": "",
                "_synthesized": tag,
            })
            existing_pairs.add((prev_id, this_id))
            in_count[this_id] = in_count.get(this_id, 0) + 1
            out_count[prev_id] = out_count.get(prev_id, 0) + 1

        for i, tid in enumerate(task_ids_in_order):
            # case 1: (M) 완전 isolated — 우선 prev 와 연결, 없으면 next.
            if tid in isolated_ids:
                connected = False
                if i > 0:
                    _emit(task_ids_in_order[i - 1], tid, "isolated_backbone")
                    connected = (task_ids_in_order[i - 1], tid) in existing_pairs
                if not connected and i + 1 < len(task_ids_in_order):
                    _emit(tid, task_ids_in_order[i + 1], "isolated_backbone")
                continue
            # case 2: (N) gap-in — 이전 task_order task 에서 들어오게.
            if tid in gap_in_ids and i > 0:
                _emit(task_ids_in_order[i - 1], tid, "section_gap_in")
            # case 3: (N) gap-out — 다음 task_order task 로 나가게.
            if tid in gap_out_ids and i + 1 < len(task_ids_in_order):
                _emit(tid, task_ids_in_order[i + 1], "section_gap_out")

        if added:
            try:
                logger.info(
                    f"[PROCDEF][BACKBONE-AUGMENT] added {len(added)} task_order-backbone "
                    f"sequence_flows "
                    f"(isolated={len(isolated_ids)}, gap_in={len(gap_in_ids)}, "
                    f"gap_out={len(gap_out_ids)})"
                )
            except Exception:
                pass
        return flows + added

    def _topological_sort_tasks(
        self,
        raw_tasks: List[Any],
        raw_flows: List[Any],
    ) -> List[Any]:
        """detail.tasks 를 detail.sequence_flows 의 topological order 로 재정렬.

        - 추출 LLM 이 task.order 를 잘못 부여하는 경우 (예: 종결 task 인 "최종 결과 통보" 가
          가장 작은 order) `task_order` 기반 단순 정렬은 시작 task 를 오선택한다.
        - sequence_flows 의 (source/from_id → target/to_id) 관계로 DAG 를 구성하여
          Kahn 알고리즘으로 topological 정렬 (in-degree 0 후보가 여럿이면 reach_count 큰 것 우선,
          동률이면 task_order 작은 것 우선).
        - cycle 이 있거나 sequence_flows 에 포함되지 않은 task 는 task_order 순으로 fallback.
        - sequence_flows 가 빈약하면 (총 task 수의 25% 미만) topology 대신 task_order 기반 정렬로 fallback.
        """
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return raw_tasks or []

        # task index 및 task_order 보조 키
        task_by_id: Dict[str, Dict[str, Any]] = {}
        task_order_by_id: Dict[str, int] = {}
        original_index: Dict[str, int] = {}
        for i, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                continue
            tid = str(t.get("task_id") or t.get("id") or "").strip()
            if not tid:
                continue
            task_by_id[tid] = t
            original_index[tid] = i
            ov = t.get("task_order")
            if ov is None:
                ov = t.get("order")
            try:
                task_order_by_id[tid] = int(ov) if ov is not None else 10**9
            except Exception:
                task_order_by_id[tid] = 10**9

        if not task_by_id:
            return raw_tasks

        # task_order 기반 단순 정렬 (fallback 으로도 사용)
        def _task_order_sorted() -> List[Any]:
            with_id = [t for t in raw_tasks if isinstance(t, dict) and (t.get("task_id") or t.get("id"))]
            without_id = [t for t in raw_tasks if not (isinstance(t, dict) and (t.get("task_id") or t.get("id")))]
            with_id.sort(key=lambda t: (
                task_order_by_id.get(str(t.get("task_id") or t.get("id") or ""), 10**9),
                str(t.get("name") or ""),
            ))
            return with_id + without_id

        flows = raw_flows if isinstance(raw_flows, list) else []
        # sequence_flows 가 빈약하면 topology 신뢰 불가 — task_order fallback
        if len(flows) < max(2, len(task_by_id) // 4):
            return _task_order_sorted()

        # DAG 빌드 (task / gateway / event 모든 노드 포함)
        adjacency: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {tid: 0 for tid in task_by_id}
        all_nodes: Set[str] = set(task_by_id.keys())

        def _f_src(f: Dict[str, Any]) -> str:
            return str(
                f.get("source")
                or f.get("from_id")
                or f.get("from_task_id")
                or ""
            ).strip()

        def _f_tgt(f: Dict[str, Any]) -> str:
            return str(
                f.get("target")
                or f.get("to_id")
                or f.get("to_task_id")
                or ""
            ).strip()

        for f in flows:
            if not isinstance(f, dict):
                continue
            s = _f_src(f); t = _f_tgt(f)
            if not s or not t or s == t:
                continue
            all_nodes.add(s); all_nodes.add(t)
            adjacency.setdefault(s, []).append(t)
            in_degree[t] = in_degree.get(t, 0) + 1
            in_degree.setdefault(s, in_degree.get(s, 0))

        # reach_count: 노드 n 에서 도달 가능한 task 노드 수 (자기 자신 포함 시)
        reach_cache: Dict[str, int] = {}

        def _reach_tasks(start: str) -> int:
            if start in reach_cache:
                return reach_cache[start]
            from collections import deque as _dq
            visited: Set[str] = {start}
            dq = _dq([start])
            while dq:
                cur = dq.popleft()
                for nxt in adjacency.get(cur, []):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    dq.append(nxt)
            cnt = sum(1 for v in visited if v in task_by_id)
            reach_cache[start] = cnt
            return cnt

        # Kahn (greedy with priority)
        remaining_indeg = dict(in_degree)
        ordered_tasks: List[Dict[str, Any]] = []
        seen_tasks: Set[str] = set()
        # node 종류 판별을 위해 task_by_id 외의 노드는 non-task (gateway/event)
        # priority: task 우선 → **task_order asc** → reach desc → original index asc → id
        # CRITICAL: extract 단계에서 (I) GLOBAL-ORDER 가 source text 위치 기반으로 task.order 를
        # 부여했으므로 그 값을 reach_count 보다 우선시한다. reach_count 우선은 LLM 추출 누락으로
        # 시작점 task ("휴가 신청서 접수") 의 outgoing 이 없을 때, 다른 긴 chain (예: "증빙 보완
        # 요청" 부터 시작하는 7개 chain) 이 head 로 선정되는 회귀를 만든다.
        guard_steps = len(remaining_indeg) + 8
        while remaining_indeg and guard_steps > 0:
            guard_steps -= 1
            cands = [n for n, d in remaining_indeg.items() if d == 0]
            if not cands:
                # cycle → task_order 가장 작은 task 부터 강제 선택
                cands = list(remaining_indeg.keys())
                cands.sort(key=lambda n: (
                    0 if n in task_by_id else 1,
                    task_order_by_id.get(n, 10**9),
                    -_reach_tasks(n),
                    original_index.get(n, 10**9),
                    n,
                ))
                if not cands:
                    break
                chosen = cands[0]
            else:
                cands.sort(key=lambda n: (
                    0 if n in task_by_id else 1,
                    task_order_by_id.get(n, 10**9),
                    -_reach_tasks(n),
                    original_index.get(n, 10**9),
                    n,
                ))
                chosen = cands[0]
            for nxt in adjacency.get(chosen, []):
                if nxt in remaining_indeg:
                    remaining_indeg[nxt] = max(0, remaining_indeg[nxt] - 1)
            remaining_indeg.pop(chosen, None)
            if chosen in task_by_id and chosen not in seen_tasks:
                ordered_tasks.append(task_by_id[chosen])
                seen_tasks.add(chosen)

        # sequence_flows 에 등장하지 않은 잔여 task 는 task_order 순으로 뒤에 붙임
        leftover_ids = [tid for tid in task_by_id if tid not in seen_tasks]
        leftover_ids.sort(key=lambda tid: (
            task_order_by_id.get(tid, 10**9),
            original_index.get(tid, 10**9),
        ))
        for tid in leftover_ids:
            ordered_tasks.append(task_by_id[tid])

        # task_id 가 없는 항목들은 끝에 원래 순서로
        no_id_items = [t for t in raw_tasks if not (isinstance(t, dict) and (t.get("task_id") or t.get("id")))]
        ordered_tasks.extend(no_id_items)
        return ordered_tasks

    def _build_elements_model_from_extracted(
        self,
        *,
        process_name: str,
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """추출 결과를 기준으로 ProcessGPT elements 모델을 강제 구성."""
        tasks = extracted.get("tasks") or extracted.get("activities") or []
        roles = extracted.get("roles") or []
        gateways = extracted.get("gateways") or []
        sequence_flows = extracted.get("sequence_flows") or extracted.get("flows") or []

        if not isinstance(tasks, list):
            tasks = []
        if not isinstance(roles, list):
            roles = []
        if not isinstance(gateways, list):
            gateways = []
        if not isinstance(sequence_flows, list):
            sequence_flows = []

        # roles
        role_names: List[str] = []
        for r in roles:
            if not isinstance(r, dict):
                continue
            rn = str(r.get("name") or r.get("role") or "").strip()
            if rn and rn not in role_names:
                role_names.append(rn)
        if not role_names:
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                rn = str(t.get("performer_role") or t.get("role") or "").strip()
                if rn and rn not in role_names:
                    role_names.append(rn)
        if not role_names:
            role_names = ["담당자"]

        role_rows = [
            {
                "name": rn,
                "endpoint": f"role_{self._snake_id(rn) or 'owner'}",
                "resolutionRule": "",
                "origin": "created",
            }
            for rn in role_names
        ]
        default_role = role_names[0]

        # node registry
        node_name_to_id: Dict[str, str] = {}
        activity_rows: List[Dict[str, Any]] = []
        gateway_rows: List[Dict[str, Any]] = []

        # activities from tasks
        sorted_tasks: List[Dict[str, Any]] = []
        for t in tasks:
            if isinstance(t, dict):
                sorted_tasks.append(t)

        def _task_order_value(x: Dict[str, Any]) -> int:
            v = x.get("task_order")
            if v is None:
                v = x.get("order")
            if v is None:
                return 10**9
            try:
                return int(v)
            except Exception:
                return 10**9

        sorted_tasks.sort(
            key=lambda x: (
                _task_order_value(x),
                str(x.get("name") or ""),
            )
        )

        for idx, t in enumerate(sorted_tasks, start=1):
            name = str(t.get("name") or "").strip() or f"활동 {idx}"
            aid = self._snake_id(str(t.get("task_id") or t.get("id") or f"task_{idx}")) or f"task_{idx}"
            role = str(t.get("performer_role") or t.get("role") or default_role).strip() or default_role
            instruction = str(t.get("instruction") or "").strip()
            description = str(t.get("description") or "").strip()

            activity_rows.append(
                {
                    "elementType": "Activity",
                    "id": aid,
                    "name": name,
                    "role": role,
                    "type": "UserActivity",
                    "source": "",
                    "description": description,
                    "instruction": instruction,
                    "inputData": [],
                    "outputData": [f"{name} 결과"],
                    "checkpoints": [],
                    "duration": "5",
                }
            )
            node_name_to_id[name.lower().strip()] = aid

        # gateways
        for idx, g in enumerate(gateways, start=1):
            if not isinstance(g, dict):
                continue
            description = str(g.get("description") or "").strip()
            name = self._derive_gateway_name(
                raw_name=str(g.get("name") or "").strip(),
                description=description,
                idx=idx,
            )
            gid = self._snake_id(str(g.get("gateway_id") or g.get("id") or f"gateway_{idx}")) or f"gateway_{idx}"
            gw_type = str(g.get("gateway_type") or g.get("type") or "ExclusiveGateway").strip()
            gw_low = gw_type.lower()
            if "parallel" in gw_low:
                gw_type = "ParallelGateway"
            elif "inclusive" in gw_low:
                gw_type = "InclusiveGateway"
            else:
                gw_type = "ExclusiveGateway"
            role = str(g.get("role") or default_role).strip() or default_role
            gateway_rows.append(
                {
                    "elementType": "Gateway",
                    "id": gid,
                    "name": name,
                    "role": role,
                    "source": "",
                    "type": gw_type,
                    "description": description,
                }
            )
            node_name_to_id[name.lower().strip()] = gid

        start_id = "start_event"
        end_id = "end_event"
        elements: List[Dict[str, Any]] = [
            {
                "elementType": "Event",
                "id": start_id,
                "name": "프로세스 시작",
                "role": default_role,
                "source": "",
                "type": "StartEvent",
                "description": "",
                "trigger": "",
            }
        ]
        elements.extend(activity_rows)
        elements.extend(gateway_rows)
        elements.append(
            {
                "elementType": "Event",
                "id": end_id,
                "name": "프로세스 종료",
                "role": default_role,
                "source": "",
                "type": "EndEvent",
                "description": "",
                "trigger": "",
            }
        )

        seqs: List[Dict[str, Any]] = []
        seen_seq: Set[Tuple[str, str, str]] = set()

        def _resolve_node_id(name_or_id: Any) -> str:
            raw = str(name_or_id or "").strip()
            if not raw:
                return ""
            key = raw.lower().strip()
            if key in node_name_to_id:
                return node_name_to_id[key]
            raw_snake = self._snake_id(raw)
            for e in elements:
                if str(e.get("id") or "") == raw_snake:
                    return raw_snake
            return ""

        # explicit extracted sequence flows
        for f in sequence_flows:
            if not isinstance(f, dict):
                continue
            src = _resolve_node_id(
                f.get("source")
                or f.get("from_id")
                or f.get("from_task_id")
                or f.get("from_task")
                or f.get("from_name")
            )
            tgt = _resolve_node_id(
                f.get("target")
                or f.get("to_id")
                or f.get("to_task_id")
                or f.get("to_task")
                or f.get("to_name")
            )
            cond = str(f.get("condition") or "").strip()
            if not src or not tgt or src == tgt:
                continue
            k = (src, tgt, cond)
            if k in seen_seq:
                continue
            seen_seq.add(k)
            seqs.append(
                {
                    "elementType": "Sequence",
                    "id": f"seq_{src}_{tgt}",
                    "name": cond or "",
                    "source": src,
                    "target": tgt,
                    "condition": cond,
                }
            )

        # fallback linear chain when explicit flows are insufficient
        node_chain: List[str] = [start_id]
        node_chain.extend([str(a.get("id")) for a in activity_rows if str(a.get("id") or "")])
        node_chain.extend([str(g.get("id")) for g in gateway_rows if str(g.get("id") or "")])
        node_chain.append(end_id)
        if len(seqs) <= 1:
            for i in range(len(node_chain) - 1):
                src = node_chain[i]
                tgt = node_chain[i + 1]
                if not src or not tgt or src == tgt:
                    continue
                k = (src, tgt, "")
                if k in seen_seq:
                    continue
                seen_seq.add(k)
                seqs.append(
                    {
                        "elementType": "Sequence",
                        "id": f"seq_{src}_{tgt}",
                        "name": "",
                        "source": src,
                        "target": tgt,
                        "condition": "",
                    }
                )

        elements.extend(seqs)
        return {
            "processDefinitionId": str(uuid.uuid4()),
            "processDefinitionName": process_name,
            "description": f"{process_name} (extracted fallback)",
            "isHorizontal": True,
            "data": [],
            "roles": role_rows,
            "elements": elements,
            "subProcesses": [],
            "participants": [],
            "generated_from": "extracted_fallback",
        }

    def _enrich_tasks_with_role_from_graph(
        self,
        *,
        detail: Dict[str, Any],
        graph_elements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Neo4j graph snapshot(PERFORMED_BY edge)로 task.role_name/role을 보강.
        get_process_with_details()의 task row에는 역할명이 직접 들어있지 않아
        deterministic procdef에서 모든 task가 default role로 뭉개지는 문제를 방지한다.
        """
        out = dict(detail or {})
        tasks = out.get("tasks") or []
        roles = out.get("roles") or []
        elements = (graph_elements or {}).get("elements") or []
        if not isinstance(tasks, list) or not tasks:
            return out

        role_id_to_name: Dict[str, str] = {}
        for r in roles:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("role_id") or r.get("id") or "").strip()
            rname = str(r.get("name") or r.get("role") or "").strip()
            if rid and rname:
                role_id_to_name[rid] = rname

        task_id_to_role_name: Dict[str, str] = {}
        task_id_to_desc: Dict[str, str] = {}
        task_id_to_instruction: Dict[str, str] = {}
        instruction_node_text_by_node_id: Dict[str, str] = {}
        for el in elements:
            if not isinstance(el, dict):
                continue
            data = el.get("data") or {}
            if not isinstance(data, dict):
                continue
            etype = str(data.get("type") or "").strip()
            if etype == "PERFORMED_BY":
                src = str(data.get("source") or "").strip()  # Task:<task_id>
                tgt = str(data.get("target") or "").strip()  # Role:<role_id>
                if src.startswith("Task:") and tgt.startswith("Role:"):
                    task_id = src.split("Task:", 1)[1].strip()
                    role_id = tgt.split("Role:", 1)[1].strip()
                    role_name = role_id_to_name.get(role_id)
                    if task_id and role_name:
                        task_id_to_role_name[task_id] = role_name
                continue

            # Task node payload may include description/instruction
            if etype == "Task":
                task_id = str(data.get("task_id") or "").strip()
                if task_id:
                    desc = str(data.get("description") or "").strip()
                    inst = str(data.get("instruction") or "").strip()
                    if desc:
                        task_id_to_desc[task_id] = desc
                    if inst:
                        task_id_to_instruction[task_id] = inst
                continue

            # Instruction node payload is separated in graph snapshot
            if etype == "Instruction":
                task_id = str(data.get("task_id") or "").strip()
                inst = str(data.get("instruction") or "").strip()
                node_id = str(data.get("id") or "").strip()
                if not inst:
                    inst = str(data.get("label") or "").strip()
                if node_id and inst:
                    instruction_node_text_by_node_id[node_id] = inst
                if task_id and inst:
                    task_id_to_instruction[task_id] = inst
                continue

            # Some snapshots only provide HAS_INSTRUCTION edge + Instruction node.
            if etype == "HAS_INSTRUCTION":
                src = str(data.get("source") or "").strip()  # Task:<task_id>
                tgt = str(data.get("target") or "").strip()  # Instruction:<task_id>:instruction
                if src.startswith("Task:") and tgt:
                    task_id = src.split("Task:", 1)[1].strip()
                    inst = instruction_node_text_by_node_id.get(tgt) or ""
                    if task_id and inst and task_id not in task_id_to_instruction:
                        task_id_to_instruction[task_id] = inst

        if not task_id_to_role_name and not task_id_to_desc and not task_id_to_instruction:
            return out

        role_changed = 0
        text_changed = 0
        patched_tasks: List[Dict[str, Any]] = []
        for t in tasks:
            if not isinstance(t, dict):
                patched_tasks.append(t)
                continue
            tid = str(t.get("task_id") or t.get("id") or "").strip()
            if tid:
                role_name = task_id_to_role_name.get(tid)
                if role_name:
                    if str(t.get("role_name") or "").strip() != role_name:
                        t["role_name"] = role_name
                        role_changed += 1
                    if str(t.get("role") or "").strip() != role_name:
                        t["role"] = role_name

                # Fill only when missing
                if not str(t.get("description") or "").strip():
                    desc = task_id_to_desc.get(tid) or ""
                    if desc:
                        t["description"] = desc
                        text_changed += 1
                if not str(t.get("instruction") or "").strip():
                    inst = task_id_to_instruction.get(tid) or ""
                    if inst:
                        t["instruction"] = inst
                        text_changed += 1
            patched_tasks.append(t)

        if role_changed > 0 or text_changed > 0:
            logger.info(
                f"[EXTRACT][TASK-ENRICH] from graph edges/nodes: role_changed={role_changed} text_changed={text_changed} "
                f"task_role_pairs={len(task_id_to_role_name)}"
            )
        out["tasks"] = patched_tasks
        return out

    def _backfill_activity_content_from_extracted(
        self,
        *,
        runtime_def: Dict[str, Any],
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministic post-fix for generated JSON:
        - Do NOT add new activities.
        - For existing activities, fill missing role/description/instruction
          from extracted tasks by (id or normalized name).
        """
        out = dict(runtime_def or {})
        acts = out.get("activities") or []
        if not isinstance(acts, list) or not acts:
            return out

        ex_tasks = extracted.get("tasks") or extracted.get("activities") or []
        if not isinstance(ex_tasks, list) or not ex_tasks:
            return out

        by_id: Dict[str, Dict[str, Any]] = {}
        by_name: Dict[str, Dict[str, Any]] = {}
        ex_rows: List[Dict[str, Any]] = []
        for t in ex_tasks:
            if not isinstance(t, dict):
                continue
            ex_rows.append(t)
            tid = str(t.get("task_id") or t.get("id") or "").strip()
            tname = str(t.get("name") or "").strip()
            if tid:
                by_id[tid] = t
            if tname:
                by_name[self._normalize_text_key(tname)] = t

        # deterministic fallback by order when id/name mapping fails
        ex_rows_sorted = sorted(
            ex_rows,
            key=lambda x: (
                10**9 if x.get("order") is None else int(x.get("order") or 0),
                str(x.get("name") or ""),
            ),
        )

        changed = 0
        for idx, a in enumerate(acts):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            aname = str(a.get("name") or "").strip()
            src = by_id.get(aid) or by_name.get(self._normalize_text_key(aname))
            if not src and aname:
                anorm = self._normalize_text_key(aname)
                for row in ex_rows_sorted:
                    rn = self._normalize_text_key(str(row.get("name") or ""))
                    if rn and (rn in anorm or anorm in rn):
                        src = row
                        break
            if not src and idx < len(ex_rows_sorted):
                src = ex_rows_sorted[idx]
            if not src:
                continue

            src_role = str(src.get("performer_role") or src.get("role") or src.get("role_name") or "").strip()
            src_desc = str(src.get("description") or "").strip()
            src_inst = str(src.get("instruction") or "").strip()

            if src_role and not str(a.get("role") or "").strip():
                a["role"] = src_role
                changed += 1
            if src_desc and not str(a.get("description") or "").strip():
                a["description"] = src_desc
                changed += 1
            if src_inst and not str(a.get("instruction") or "").strip():
                a["instruction"] = src_inst
                changed += 1

        if changed > 0:
            logger.info(f"[PROCDEF][BACKFILL] filled missing activity fields from extracted: changed={changed}")
        out["activities"] = acts
        return out

    def _augment_runtime_with_gateway_dmn(
        self,
        *,
        runtime_def: Dict[str, Any],
        extracted: Dict[str, Any],
        approved_gateway_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Gateway 분기 조건을 검증 가능한 형태로 남기기 위해 DMN 메타를 보강합니다.
        - dmn_decisions: gateway별 의사결정 단위
        - dmn_rules: sequence condition별 true/false(또는 다중 분기) 규칙
        """
        out = dict(runtime_def or {})
        gateways = out.get("gateways") or []
        sequences = out.get("sequences") or []
        activities = out.get("activities") or []
        events = out.get("events") or []

        if not isinstance(gateways, list) or not isinstance(sequences, list):
            return out

        existing_decisions = out.get("dmn_decisions") if isinstance(out.get("dmn_decisions"), list) else []
        existing_rules = out.get("dmn_rules") if isinstance(out.get("dmn_rules"), list) else []

        # Keep extracted decisions/rules as base when available.
        ex_decisions = extracted.get("decisions") if isinstance(extracted.get("decisions"), list) else []
        ex_rules = extracted.get("rules") if isinstance(extracted.get("rules"), list) else []

        decisions_out: List[Dict[str, Any]] = []
        rules_out: List[Dict[str, Any]] = []
        seen_decision_ids: Set[str] = set()
        seen_rule_ids: Set[str] = set()

        def _append_decision(d: Dict[str, Any]):
            did = str(d.get("decision_id") or d.get("id") or "").strip()
            if not did:
                return
            if did in seen_decision_ids:
                return
            seen_decision_ids.add(did)
            decisions_out.append(d)

        def _append_rule(r: Dict[str, Any]):
            rid = str(r.get("rule_id") or r.get("id") or "").strip()
            if not rid:
                return
            if rid in seen_rule_ids:
                return
            seen_rule_ids.add(rid)
            rules_out.append(r)

        for d in (existing_decisions + ex_decisions):
            if isinstance(d, dict):
                did = str(d.get("decision_id") or d.get("id") or "").strip()
                if not did:
                    did = f"dmn_decision_{uuid.uuid4().hex[:8]}"
                    d = {**d, "decision_id": did}
                _append_decision(d)
        for r in (existing_rules + ex_rules):
            if isinstance(r, dict):
                rid = str(r.get("rule_id") or r.get("id") or "").strip()
                if not rid:
                    rid = f"dmn_rule_{uuid.uuid4().hex[:8]}"
                    r = {**r, "rule_id": rid}
                _append_rule(r)

        node_name_by_id: Dict[str, str] = {}
        for coll in (activities, events, gateways):
            if not isinstance(coll, list):
                continue
            for n in coll:
                if isinstance(n, dict):
                    nid = str(n.get("id") or "").strip()
                    if nid:
                        node_name_by_id[nid] = str(n.get("name") or "").strip()

        outgoing_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for s in sequences:
            if not isinstance(s, dict):
                continue
            src = str(s.get("source") or "").strip()
            if src:
                outgoing_by_source.setdefault(src, []).append(s)

        added_decisions = 0
        added_rules = 0
        for gw in gateways:
            if not isinstance(gw, dict):
                continue
            gid = str(gw.get("id") or "").strip()
            gtype = str(gw.get("type") or "").lower().strip()
            if not gid or "exclusive" not in gtype:
                continue
            # HITL: 사용자가 선택한 게이트웨이만 DMN 으로 변환 (None 이면 전체 자동)
            if approved_gateway_ids is not None and gid not in approved_gateway_ids:
                continue
            outs = outgoing_by_source.get(gid) or []
            if len(outs) < 2:
                continue

            gname = str(gw.get("name") or "").strip()
            gdesc = str(gw.get("description") or "").strip()
            decision_id = f"dmn_decision_{self._snake_id(gid) or gid}"
            decision = {
                "decision_id": decision_id,
                "name": gname or "분기 의사결정",
                "description": gdesc or f"{gname or gid}에 대한 분기 판단",
                "related_gateway_id": gid,
            }
            if decision_id not in seen_decision_ids:
                _append_decision(decision)
                added_decisions += 1

            for idx, s in enumerate(outs, start=1):
                cond = str(s.get("condition") or "").strip()
                if not cond:
                    continue
                target = str(s.get("target") or "").strip()
                target_name = node_name_by_id.get(target) or target
                rule_id = f"dmn_rule_{self._snake_id(gid) or gid}_{idx}"
                rule = {
                    "rule_id": rule_id,
                    "decision_id": decision_id,
                    "decision_name": decision.get("name"),
                    "when": cond,
                    "then": f"{target_name} 경로 선택",
                    "condition": cond,
                    "target": target,
                }
                if rule_id not in seen_rule_ids:
                    _append_rule(rule)
                    added_rules += 1

            # 분기 조건을 프론트(ConditionExampleField) 스키마의 good/bad 예시로 변환해
            # 각 분기 시퀀스플로우의 properties.examples 에 기록한다.
            # exclusive 게이트웨이에서 한 분기의 '좋은 예시'(= 이 분기 조건)와
            # '나쁜 예시'(= 다른 분기 조건)는 서로 반대 케이스이므로 LLM 없이 도출된다.
            conditioned: List[Tuple[Dict[str, Any], str, str]] = []
            for s in outs:
                c = str(s.get("condition") or "").strip()
                if not c:
                    continue
                tgt = str(s.get("target") or "").strip()
                tname = node_name_by_id.get(tgt) or tgt
                conditioned.append((s, c, tname))
            if len(conditioned) >= 2:
                now_iso = datetime.now(timezone.utc).isoformat()
                given_text = gname or "분기 판단"
                for s, cond, tname in conditioned:
                    good_examples = [{
                        "given": given_text,
                        "when": cond,
                        "then": f"{tname} 경로로 진행",
                        "valid_at": now_iso,
                    }]
                    bad_examples = [
                        {
                            "given": given_text,
                            "when": cond2,
                            "then": "이 경로로 진행하지 않음",
                            "invalid_at": now_iso,
                        }
                        for s2, cond2, _tn2 in conditioned
                        if s2 is not s
                    ]
                    try:
                        raw_props = s.get("properties")
                        if isinstance(raw_props, str) and raw_props.strip():
                            props = json.loads(raw_props)
                        elif isinstance(raw_props, dict):
                            props = dict(raw_props)
                        else:
                            props = {}
                    except Exception:
                        props = {}
                    if not isinstance(props, dict):
                        props = {}
                    props["examples"] = {
                        "good_examples": good_examples,
                        "bad_examples": bad_examples,
                    }
                    s["properties"] = json.dumps(props, ensure_ascii=False)

        if added_decisions > 0 or added_rules > 0:
            logger.info(
                f"[PROCDEF][DMN] gateway-derived dmn added: decisions={added_decisions} rules={added_rules}"
            )

        out["dmn_decisions"] = decisions_out
        out["dmn_rules"] = rules_out
        return out

    def _apply_extracted_roles_to_runtime_definition(
        self,
        *,
        runtime_def: Dict[str, Any],
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        If runtime_def roles collapse into one role, remap activity roles using extracted data.
        Priority:
        1) extracted.task_role_mappings
        2) extracted.tasks[].(performer_role|role|role_name)
        """
        out = dict(runtime_def or {})
        acts = out.get("activities") or []
        roles = out.get("roles") or []
        if not isinstance(acts, list) or not acts:
            return out

        # role pool from extracted
        role_pool: List[str] = []
        for r in (extracted.get("roles") or []):
            if isinstance(r, dict):
                rn = str(r.get("name") or r.get("role") or "").strip()
                if rn and rn not in role_pool:
                    role_pool.append(rn)

        # Build task->role map from explicit mappings
        task_role_map: Dict[str, str] = {}
        for m in (extracted.get("task_role_mappings") or []):
            if not isinstance(m, dict):
                continue
            tname = str(m.get("task_name") or m.get("task") or m.get("name") or "").strip()
            rname = str(m.get("role_name") or m.get("role") or m.get("performer_role") or "").strip()
            if tname and rname:
                task_role_map[self._normalize_text_key(tname)] = rname
                if rname not in role_pool:
                    role_pool.append(rname)

        # Build task->role map from tasks
        extracted_tasks = extracted.get("tasks") or extracted.get("activities") or []
        if isinstance(extracted_tasks, list):
            for t in extracted_tasks:
                if not isinstance(t, dict):
                    continue
                tname = str(t.get("name") or "").strip()
                rname = str(t.get("performer_role") or t.get("role") or t.get("role_name") or "").strip()
                if tname and rname:
                    task_role_map.setdefault(self._normalize_text_key(tname), rname)
                    if rname not in role_pool:
                        role_pool.append(rname)

        if not task_role_map:
            return out

        # remap activities
        changed = 0
        for a in acts:
            if not isinstance(a, dict):
                continue
            key = self._normalize_text_key(a.get("name"))
            mapped = task_role_map.get(key)
            if mapped:
                if str(a.get("role") or "").strip() != mapped:
                    a["role"] = mapped
                    changed += 1

        if changed > 0 and role_pool:
            # Keep role rows aligned with activity roles used after remap.
            used_roles = []
            for a in acts:
                rn = str((a or {}).get("role") or "").strip()
                if rn and rn not in used_roles:
                    used_roles.append(rn)
            if used_roles:
                out["roles"] = [
                    {
                        "name": rn,
                        "endpoint": f"role_{self._snake_id(rn) or 'owner'}",
                        "resolutionRule": "",
                        "origin": "created",
                    }
                    for rn in used_roles
                ]
            logger.info(f"[PROCDEF][ROLE-REMAP] remapped roles from extracted: changed={changed} used_roles={len(out.get('roles') or [])}")

        out["activities"] = acts
        return out

    def _validate_and_normalize_elements_model(
        self,
        elements_model: Dict[str, Any],
        *,
        process_name: str,
    ) -> Dict[str, Any]:
        """
        (t3) LLM 결과(elements 모델)를 더 엄격히 검증/정규화하여:
        - 끊긴 연결선/누락된 source/target을 복구
        - ids/elementType/type 등을 표준화
        - Activity의 outputData/tool 등 런타임/레이아웃에 필요한 최소 필드를 보정

        NOTE:
        - 비즈니스 내용을 새로 창작하지 않되, "기술적 필수 요소" (start/end, sequence 연결, 필수 필드) 보정은 허용.
        """
        m = dict(elements_model or {})

        # --- Diagnostics: "왜 start/end만 나오나"를 확정하기 위한 로그 ---
        try:
            raw_elems = m.get("elements")
            raw_elems_len = len(raw_elems) if isinstance(raw_elems, list) else None
            logger.info(
                f"[PROCDEF][NORMALIZE] begin: process={process_name!r} keys={list(m.keys())} elements_len={raw_elems_len}"
            )
        except Exception:
            pass

        # Ensure required identifiers
        m.setdefault("processDefinitionName", process_name)
        if not str(m.get("processDefinitionId") or "").strip():
            # IMPORTANT:
            # - proc_def 저장 키(processDefinitionId)는 충돌이 나면 기존 프로세스가 덮이거나(proc_def 갱신)
            #   proc_map / form_def 매핑이 깨질 수 있으므로 UUID로 강제합니다.
            m["processDefinitionId"] = str(uuid.uuid4())

        # Normalize elements list
        elems_raw = m.get("elements") or []
        elems: List[Dict[str, Any]] = []
        if isinstance(elems_raw, list):
            elems = [e for e in elems_raw if isinstance(e, dict)]
        elif isinstance(elems_raw, dict):
            elems = [e for e in elems_raw.values() if isinstance(e, dict)]
        else:
            elems = []

        if not elems:
            # 이 케이스면 이후 로직이 start/end(+직선 sequence)만 자동 삽입하게 되며,
            # 결국 proc_def.definition이 start/end만 남는 현상이 발생할 수 있습니다.
            logger.warning(
                f"[PROCDEF][NORMALIZE] elements is empty BEFORE repair. This will lead to start/end-only skeleton. "
                f"(process={process_name!r})"
            )

        # Normalize elementType casing & types
        def norm_element_type(et: str) -> str:
            t = (et or "").strip().lower()
            if t == "event":
                return "Event"
            if t == "sequence":
                return "Sequence"
            if t == "activity":
                return "Activity"
            if t == "gateway":
                return "Gateway"
            return et or ""

        # First pass: normalize ids (build mapping old->new)
        id_map: Dict[str, str] = {}
        for idx, e in enumerate(elems):
            et = norm_element_type(str(e.get("elementType") or ""))
            e["elementType"] = et
            if et == "Sequence":
                continue
            old = str(e.get("id") or "").strip()
            if not old:
                # generate deterministic-ish id by type
                base = "event" if et == "Event" else "gateway" if et == "Gateway" else "activity"
                old = f"{base}_{idx+1}"
            new = self._snake_id(old)
            if not new:
                new = f"node_{idx+1}"
            # ensure uniqueness
            if new in id_map.values():
                new = f"{new}_{uuid.uuid4().hex[:4]}"
            id_map[old] = new
            e["id"] = new

            # normalize event/activity/gateway type value
            if et == "Event":
                t = str(e.get("type") or "").strip()
                t_low = t.lower()
                if t_low in ("startevent", "start_event", "start"):
                    e["type"] = "StartEvent"
                elif t_low in ("endevent", "end_event", "end"):
                    e["type"] = "EndEvent"
                elif t:
                    # keep as-is but enforce Pascal-ish (fallback to IntermediateCatchEvent)
                    e["type"] = t if t[0].isupper() else "IntermediateCatchEvent"
                else:
                    e["type"] = "IntermediateCatchEvent"
            elif et == "Activity":
                # ProcessGPT only supports UserActivity
                e["type"] = "UserActivity"

                # required-ish fields for stability
                e.setdefault("name", f"활동 {idx+1}")
                if not isinstance(e.get("inputData"), list):
                    e["inputData"] = []
                if not isinstance(e.get("outputData"), list):
                    e["outputData"] = []
                if not e["outputData"]:
                    # 최소 1개는 필요 (프롬프트 규칙 + 실행/폼 안정성)
                    an = str(e.get("name") or "").strip()
                    e["outputData"] = [f"{an} 결과" if an else "결과"]
                e.setdefault("checkpoints", [])
                if not isinstance(e.get("checkpoints"), list):
                    e["checkpoints"] = []
                # tool은 런타임 변환에서 채워지지만, elements 모델에도 있으면 일관성 도움
                if not str(e.get("tool") or "").strip():
                    safe_pid = self._snake_id(str(m.get("processDefinitionId") or "process"))
                    safe_aid = self._snake_id(str(e.get("id") or "activity"))
                    e["tool"] = f"formHandler:{safe_pid}_{safe_aid}_form"
                # duration
                try:
                    d = int(e.get("duration") or 5)
                except Exception:
                    d = 5
                e["duration"] = d
            elif et == "Gateway":
                gt = str(e.get("type") or "").strip()
                gt_low = gt.lower()
                if gt_low in ("exclusivegateway", "exclusive_gateway"):
                    e["type"] = "ExclusiveGateway"
                elif gt_low in ("parallelgateway", "parallel_gateway"):
                    e["type"] = "ParallelGateway"
                elif gt_low in ("inclusivegateway", "inclusive_gateway"):
                    e["type"] = "InclusiveGateway"
                else:
                    e["type"] = gt or "ExclusiveGateway"

            # normalize source pointer if present
            if e.get("source"):
                e["source"] = id_map.get(str(e.get("source")), self._snake_id(str(e.get("source"))))

        # Second pass: normalize sequences, fix source/target and create missing sequences from 'source' pointers
        node_ids = {str(e.get("id")) for e in elems if e.get("elementType") != "Sequence" and e.get("id")}
        seq_pairs: Set[Tuple[str, str]] = set()
        seqs: List[Dict[str, Any]] = []

        # helper: find prev/next node id in element order
        node_order: List[str] = [str(e.get("id")) for e in elems if e.get("elementType") != "Sequence" and e.get("id")]
        for i, e in enumerate(elems):
            if e.get("elementType") != "Sequence":
                continue
            s = str(e.get("source") or "").strip()
            t = str(e.get("target") or "").strip()
            # remap
            s = id_map.get(s, self._snake_id(s)) if s else ""
            t = id_map.get(t, self._snake_id(t)) if t else ""

            # infer from surrounding nodes if missing
            if (not s) or (not t) or (s not in node_ids) or (t not in node_ids):
                # find nearest prev/next node in elems list
                prev_node = ""
                next_node = ""
                for j in range(i - 1, -1, -1):
                    if elems[j].get("elementType") != "Sequence" and elems[j].get("id"):
                        prev_node = str(elems[j].get("id"))
                        break
                for j in range(i + 1, len(elems)):
                    if elems[j].get("elementType") != "Sequence" and elems[j].get("id"):
                        next_node = str(elems[j].get("id"))
                        break
                if not s and prev_node:
                    s = prev_node
                if not t and next_node:
                    t = next_node

            if not s or not t or s == t or (s not in node_ids) or (t not in node_ids):
                continue

            e["source"] = s
            e["target"] = t
            if not str(e.get("id") or "").strip():
                e["id"] = f"seq_{uuid.uuid4().hex[:8]}"
            else:
                e["id"] = self._snake_id(str(e.get("id")))
            e.setdefault("name", "")
            e.setdefault("condition", "")
            seq_pairs.add((s, t))
            seqs.append(e)

        # create sequences from explicit 'source' pointers on nodes if missing
        for e in elems:
            if e.get("elementType") == "Sequence":
                continue
            src = str(e.get("source") or "").strip()
            tid = str(e.get("id") or "").strip()
            if src and tid and (src, tid) not in seq_pairs and src in node_ids and tid in node_ids and src != tid:
                seqs.append(
                    {
                        "elementType": "Sequence",
                        "id": f"seq_{src}_{tid}",
                        "name": "",
                        "source": src,
                        "target": tid,
                        "condition": "",
                    }
                )
                seq_pairs.add((src, tid))

        # Ensure start/end exist (technical requirement)
        has_start = any(e.get("elementType") == "Event" and e.get("type") == "StartEvent" for e in elems)
        has_end = any(e.get("elementType") == "Event" and e.get("type") == "EndEvent" for e in elems)
        if not has_start:
            sid = f"start_{uuid.uuid4().hex[:6]}"
            elems.insert(
                0,
                {
                    "elementType": "Event",
                    "id": sid,
                    "name": "프로세스 시작",
                    "role": (m.get("roles") or [{}])[0].get("name") if isinstance(m.get("roles"), list) and m.get("roles") else "",
                    "source": "",
                    "type": "StartEvent",
                    "description": "",
                    "trigger": "",
                },
            )
            node_order.insert(0, sid)
            node_ids.add(sid)
        if not has_end:
            eid = f"end_{uuid.uuid4().hex[:6]}"
            elems.append(
                {
                    "elementType": "Event",
                    "id": eid,
                    "name": "프로세스 종료",
                    "role": (m.get("roles") or [{}])[-1].get("name") if isinstance(m.get("roles"), list) and m.get("roles") else "",
                    "source": "",
                    "type": "EndEvent",
                    "description": "",
                    "trigger": "",
                }
            )
            node_order.append(eid)
            node_ids.add(eid)

        # Recompute node order after potential insertions (exclude sequences)
        node_order = [str(e.get("id")) for e in elems if e.get("elementType") != "Sequence" and e.get("id")]

        # Identify start/end ids and structural helpers
        start_event_ids: Set[str] = {
            str(e.get("id"))
            for e in elems
            if isinstance(e, dict)
            and e.get("elementType") == "Event"
            and str(e.get("type") or "").lower().startswith("start")
            and e.get("id")
        }
        end_event_ids: Set[str] = {
            str(e.get("id"))
            for e in elems
            if isinstance(e, dict)
            and e.get("elementType") == "Event"
            and str(e.get("type") or "").lower().startswith("end")
            and e.get("id")
        }
        non_event_ids = [nid for nid in node_order if nid not in start_event_ids and nid not in end_event_ids]

        outgoing_count: Dict[str, int] = {}
        incoming_count: Dict[str, int] = {}
        for sf in seqs:
            outgoing_count[str(sf.get("source"))] = outgoing_count.get(str(sf.get("source")), 0) + 1
            incoming_count[str(sf.get("target"))] = incoming_count.get(str(sf.get("target")), 0) + 1

        # Connectivity repair (conservative):
        # - Only insert chain edges when the graph is essentially empty (no real sequences),
        #   so we don't pollute extracted/LLM flows with fabricated ones.
        # - Otherwise: only ensure start has an outgoing edge, and that "leaf" nodes
        #   (no outgoing, non-start, non-end) flow into end_event.
        real_seq_count = sum(
            1
            for sf in seqs
            if str(sf.get("source")) and str(sf.get("target"))
            and str(sf.get("source")) not in start_event_ids
            and str(sf.get("target")) not in end_event_ids
        )

        if real_seq_count == 0:
            # graph is empty — fall back to a deterministic linear chain
            chain_order: List[str] = []
            chain_order.extend(sorted(start_event_ids))
            chain_order.extend(non_event_ids)
            chain_order.extend(sorted(end_event_ids))
            for i in range(len(chain_order) - 1):
                s = chain_order[i]
                t = chain_order[i + 1]
                if not s or not t or s == t:
                    continue
                if (s, t) in seq_pairs:
                    continue
                seqs.append(
                    {
                        "elementType": "Sequence",
                        "id": f"seq_{s}_{t}",
                        "name": "",
                        "source": s,
                        "target": t,
                        "condition": "",
                    }
                )
                seq_pairs.add((s, t))
        else:
            # ensure start has at least one outgoing edge
            #
            # 후보 선택 정책 (회귀 방지):
            # - non_event_ids 의 1차 순서는 elements 등록 순서(=task_order 순)인데,
            #   neo4j 의 task_order 또는 extractor LLM 이 잘못된 순서를 부여한 경우
            #   "최종 결과 통보" 같은 종결 task 가 list 앞에 와 잘못 선택된다.
            # - 따라서 incoming==0 인 후보가 여러 개일 때는 "그 노드로부터 도달 가능한
            #   non-event 노드 수가 가장 많은" 노드를 진짜 시작점으로 본다.
            #   (BPMN 진짜 시작점은 거의 모든 노드에 도달 가능, isolated 잘못된 노드는
            #    도달 가능 set 이 작다)
            adjacency: Dict[str, List[str]] = {}
            for s in seqs:
                src = str(s.get("source") or "")
                tgt = str(s.get("target") or "")
                if src and tgt:
                    adjacency.setdefault(src, []).append(tgt)

            def _reach_count(start_id: str) -> int:
                visited: Set[str] = {start_id}
                stack: List[str] = [start_id]
                while stack:
                    cur = stack.pop()
                    for nxt in adjacency.get(cur, []):
                        if nxt in visited:
                            continue
                        if nxt in start_event_ids or nxt in end_event_ids:
                            continue
                        visited.add(nxt)
                        stack.append(nxt)
                return len(visited)

            for sid in start_event_ids:
                if outgoing_count.get(sid, 0) > 0:
                    continue
                # 1순위: incoming==0 인 후보 중 "도달 가능한 노드 수" 가 가장 큰 것
                #        동률이면 outgoing 이 많은 쪽, 또 동률이면 non_event_ids 순서 유지
                zero_in_candidates = [
                    nid for nid in non_event_ids
                    if incoming_count.get(nid, 0) == 0
                ]
                preferred = ""
                if zero_in_candidates:
                    if len(zero_in_candidates) == 1:
                        preferred = zero_in_candidates[0]
                    else:
                        scored = [
                            (
                                _reach_count(nid),
                                outgoing_count.get(nid, 0),
                                -non_event_ids.index(nid),  # 안정 정렬 (앞쪽 우선)
                                nid,
                            )
                            for nid in zero_in_candidates
                        ]
                        scored.sort(reverse=True)
                        preferred = scored[0][3]
                        try:
                            logger.info(
                                f"[PROCDEF][START-PICK] candidates={zero_in_candidates} "
                                f"picked={preferred!r} scores={[(s[0], s[1], s[3]) for s in scored]}"
                            )
                        except Exception:
                            pass
                if not preferred and non_event_ids:
                    preferred = non_event_ids[0]
                if preferred and (sid, preferred) not in seq_pairs:
                    seqs.append(
                        {
                            "elementType": "Sequence",
                            "id": f"seq_{sid}_{preferred}",
                            "name": "",
                            "source": sid,
                            "target": preferred,
                            "condition": "",
                        }
                    )
                    seq_pairs.add((sid, preferred))
                    outgoing_count[sid] = outgoing_count.get(sid, 0) + 1
                    incoming_count[preferred] = incoming_count.get(preferred, 0) + 1

            # connect real "leaf" non-event nodes to end_event (no outgoing of their own)
            for eid in end_event_ids:
                for nid in non_event_ids:
                    if outgoing_count.get(nid, 0) > 0:
                        continue
                    if nid in start_event_ids or nid in end_event_ids:
                        continue
                    if (nid, eid) in seq_pairs:
                        continue
                    seqs.append(
                        {
                            "elementType": "Sequence",
                            "id": f"seq_{nid}_{eid}",
                            "name": "",
                            "source": nid,
                            "target": eid,
                            "condition": "",
                        }
                    )
                    seq_pairs.add((nid, eid))
                    outgoing_count[nid] = outgoing_count.get(nid, 0) + 1
                    incoming_count[eid] = incoming_count.get(eid, 0) + 1

        # ----------------------------------------------------------------
        # Orphan terminal task 보정 (회귀 방지)
        # ----------------------------------------------------------------
        # 추출 LLM 이 종결 task (예: "최종 결과 통보") 의 incoming sequence 를 누락하면
        # 해당 task 는 incoming==0 인 채로 outgoing 만 end_event 에 가는 고아 노드가 된다.
        # 이런 형태는 BPMN 상 무의미하므로 (start 와 연결도 없음) 다른 종결 path 들을
        # orphan_terminal 로 우회시켜 의미상 마지막 합류점으로 통합한다.
        #
        # 조건:
        #   - incoming_count == 0 (start-pick 이후 기준)
        #   - 모든 outgoing 이 end_event 로만 향함
        #   - 그런 task 가 정확히 1개일 때만 적용 (모호한 경우 미적용)
        # ----------------------------------------------------------------
        orphan_terminal_id = ""
        orphan_terminal_multiple = False
        for nid in non_event_ids:
            if incoming_count.get(nid, 0) > 0:
                continue
            if nid in start_event_ids or nid in end_event_ids:
                continue
            # start_event 에서 직접 outgoing 받은 node 는 (sid, nid) in seq_pairs 로 식별
            if any((sid, nid) in seq_pairs for sid in start_event_ids):
                continue
            my_outs = [s for s in seqs if str(s.get("source")) == nid]
            if not my_outs:
                continue
            if not all(str(s.get("target")) in end_event_ids for s in my_outs):
                continue
            if orphan_terminal_id and orphan_terminal_id != nid:
                orphan_terminal_multiple = True
                break
            orphan_terminal_id = nid

        if orphan_terminal_id and not orphan_terminal_multiple:
            other_end_inbounds = [
                s for s in seqs
                if str(s.get("target")) in end_event_ids
                and str(s.get("source")) != orphan_terminal_id
                and str(s.get("source")) not in start_event_ids
            ]
            if other_end_inbounds:
                try:
                    logger.info(
                        f"[PROCDEF][ORPHAN-TERMINAL] divert other end-inbound paths to orphan "
                        f"terminal={orphan_terminal_id!r} "
                        f"diverted_sources={[str(s.get('source')) for s in other_end_inbounds]}"
                    )
                except Exception:
                    pass
                for s in other_end_inbounds:
                    old_src = str(s.get("source") or "")
                    old_tgt = str(s.get("target") or "")
                    if not old_src or not old_tgt or old_src == orphan_terminal_id:
                        continue
                    # source → orphan_terminal 가 이미 있으면 기존 sequence 제거
                    if (old_src, orphan_terminal_id) in seq_pairs:
                        seq_pairs.discard((old_src, old_tgt))
                        incoming_count[old_tgt] = max(0, incoming_count.get(old_tgt, 0) - 1)
                        s["__drop__"] = True
                        continue
                    seq_pairs.discard((old_src, old_tgt))
                    s["target"] = orphan_terminal_id
                    s["id"] = f"seq_{old_src}_{orphan_terminal_id}"
                    seq_pairs.add((old_src, orphan_terminal_id))
                    incoming_count[old_tgt] = max(0, incoming_count.get(old_tgt, 0) - 1)
                    incoming_count[orphan_terminal_id] = incoming_count.get(orphan_terminal_id, 0) + 1
                seqs = [s for s in seqs if not (isinstance(s, dict) and s.get("__drop__"))]

        # Gateway branching: enforce explicit Gateway nodes for any branching point.
        # If a non-gateway node has multiple outgoing flows, insert an ExclusiveGateway and reroute.
        #
        # 이전에는 "condition 이 명시된 outgoing 이 하나라도 있어야" gateway 를 만들도록
        # 보수적으로 두었으나, 추출 LLM 이 분기 task 의 outgoing 들에 condition 을 빠뜨리면
        # multi-outgoing 이 그대로 task→task 다중 흐름으로 남아 BPMN 표준에 맞지 않는
        # "task 가 분기점" 패턴이 생겨버린다 (예: "휴가 신청서 접수" 가 outgoing 2개).
        # → multi-outgoing 인 non-gateway 노드는 모두 ExclusiveGateway 로 래핑한다.
        # 단, 이미 다운스트림에 gateway 가 있다면 (분기가 다른 곳에서 모델링됐다면) 건너뛴다.
        outgoing_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for s in seqs:
            outgoing_by_source.setdefault(str(s.get("source")), []).append(s)
        element_by_id_for_branching: Dict[str, Dict[str, Any]] = {
            str(e.get("id")): e for e in elems if isinstance(e, dict) and e.get("id")
        }

        # Does this source already feed directly into an existing Gateway? If so, real branching
        # is already represented downstream and we should not insert a second wrapper gateway.
        def _has_downstream_gateway(src_id: str, outs_list: List[Dict[str, Any]]) -> bool:
            for s in outs_list:
                tgt = str(s.get("target") or "").strip()
                tgt_el = element_by_id_for_branching.get(tgt) or {}
                if str(tgt_el.get("elementType") or "") == "Gateway":
                    return True
            return False

        for src_id, outs in list(outgoing_by_source.items()):
            if len(outs) <= 1:
                continue
            src_el = element_by_id_for_branching.get(src_id) or {}
            if str(src_el.get("elementType") or "") == "Gateway":
                continue

            # Skip if any outgoing already targets a Gateway (branching already modelled)
            if _has_downstream_gateway(src_id, outs):
                continue

            # NOTE: 과거에는 condition 이 하나라도 있어야만 wrap 했지만
            # 추출 LLM 의 condition 누락이 일상적이라 그 가드를 제거한다.
            # multi-outgoing 자체가 BPMN 표준상 분기 의미이고, 게이트웨이로 표현해야 옳다.
            try:
                logger.info(
                    f"[PROCDEF][GW-WRAP] wrapping multi-outgoing source={src_id!r} "
                    f"branch_count={len(outs)} "
                    f"branch_targets={[str(s.get('target') or '') for s in outs]}"
                )
            except Exception:
                pass

            src_name = str(src_el.get("name") or "").strip()
            src_desc = str(src_el.get("description") or "").strip()
            gw_base = self._snake_id(f"{src_id}_gateway") or f"gateway_{uuid.uuid4().hex[:6]}"
            gw_id = gw_base
            suffix = 1
            while any(str(e.get("id")) == gw_id for e in elems):
                suffix += 1
                gw_id = f"{gw_base}_{suffix}"

            gw_name = self._derive_gateway_name(
                raw_name=f"{src_name or src_id} 분기",
                description=(src_desc or f"{src_name or src_id} 여부 판단"),
                idx=1,
            )
            gw_role = str(src_el.get("role") or "").strip()
            gw_element = {
                "elementType": "Gateway",
                "id": gw_id,
                "name": gw_name,
                "role": gw_role,
                "source": src_id,
                "type": "ExclusiveGateway",
                "description": (src_desc or f"{src_name or src_id} 분기 판단"),
            }
            elems.append(gw_element)
            element_by_id_for_branching[gw_id] = gw_element

            # Remove original src->* branch edges, then add src->gateway and gateway->* edges.
            original_out_targets = []
            remaining_seqs: List[Dict[str, Any]] = []
            for s in seqs:
                if str(s.get("source")) == src_id:
                    original_out_targets.append(dict(s))
                    continue
                remaining_seqs.append(s)
            seqs = remaining_seqs

            seqs.append(
                {
                    "elementType": "Sequence",
                    "id": f"seq_{src_id}_{gw_id}",
                    "name": "",
                    "source": src_id,
                    "target": gw_id,
                    "condition": "",
                }
            )
            for s in original_out_targets:
                tgt = str(s.get("target") or "").strip()
                if not tgt or tgt == gw_id:
                    continue
                cond = str(s.get("condition") or "").strip()
                seqs.append(
                    {
                        "elementType": "Sequence",
                        "id": f"seq_{gw_id}_{tgt}",
                        "name": cond,
                        "source": gw_id,
                        "target": tgt,
                        "condition": cond,
                    }
                )

        # Gateway branching: ensure conditions exist when a gateway has multiple outgoing.
        # Also: handle degenerate gateways (<=1 outgoing).
        #   기존: 무조건 collapse(삭제) 했음 → 추출 단계에서 연결선이 빠진 게이트웨이가
        #         곧장 사라짐.
        #   변경: 가능한 한 누락된 outgoing 을 자동 보강한 뒤 게이트웨이를 유지하고,
        #         보강 후보가 정말 없을 때만 collapse 한다.
        outgoing_by_source = {}
        incoming_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for s in seqs:
            outgoing_by_source.setdefault(str(s.get("source")), []).append(s)
            incoming_by_target.setdefault(str(s.get("target")), []).append(s)

        gateway_ids = {str(e.get("id")) for e in elems if e.get("elementType") == "Gateway" and e.get("id")}

        removed_gateway_ids: Set[str] = set()
        if gateway_ids:
            element_by_id_for_gw: Dict[str, Dict[str, Any]] = {
                str(e.get("id")): e for e in elems if isinstance(e, dict) and e.get("id")
            }

            def _gw_collect_candidates(
                gid: str,
                outs_list: List[Dict[str, Any]],
                ins_list: List[Dict[str, Any]],
            ) -> List[Tuple[str, str]]:
                """게이트웨이의 누락된 outgoing 을 보강할 후보 (target_id, condition) 목록.

                보강 휴리스틱:
                  (1) incoming source task 의 _다른_ outgoing(gateway 외) 의 target 들
                      LLM 이 분기 task 의 multi-outgoing 을 그대로 놔두고 그중 일부만
                      게이트웨이로 우회시킨 케이스 → sibling 들을 게이트웨이로 모은다.
                  (2) gateway 의 name/description 에 task name 이 substring 으로 등장하는
                      같은 process 내 Activity (보수적 매칭, len(ename) >= 3 강제).
                """
                existing_out_target_ids: Set[str] = {
                    str(s.get("target") or "") for s in outs_list
                }
                candidates: List[Tuple[str, str]] = []
                seen_targets: Set[str] = set(existing_out_target_ids)

                # (1) sibling outgoing
                for inc_ in ins_list:
                    src_id = str(inc_.get("source") or "")
                    if not src_id:
                        continue
                    src_el = element_by_id_for_gw.get(src_id) or {}
                    if str(src_el.get("elementType") or "") == "Gateway":
                        continue
                    for sib in outgoing_by_source.get(src_id) or []:
                        sib_tgt = str(sib.get("target") or "")
                        if not sib_tgt or sib_tgt == gid or sib_tgt in seen_targets:
                            continue
                        sib_el = element_by_id_for_gw.get(sib_tgt) or {}
                        if str(sib_el.get("elementType") or "") in ("Event", "Gateway"):
                            continue
                        cond_ = str(sib.get("condition") or "").strip()
                        candidates.append((sib_tgt, cond_))
                        seen_targets.add(sib_tgt)

                # (2) keyword / substring 매칭 — 보수적
                gw_el_kw = element_by_id_for_gw.get(gid) or {}
                gw_text_parts = [
                    str(gw_el_kw.get("name") or ""),
                    str(gw_el_kw.get("description") or ""),
                ]
                gw_text = " ".join([p for p in gw_text_parts if p]).strip()
                same_proc_id = str(
                    gw_el_kw.get("processId") or gw_el_kw.get("process_id") or ""
                )
                if gw_text:
                    for el in elems:
                        if not isinstance(el, dict):
                            continue
                        if str(el.get("elementType") or "") != "Activity":
                            continue
                        eid = str(el.get("id") or "")
                        if not eid or eid in seen_targets or eid == gid:
                            continue
                        ename = str(el.get("name") or "").strip()
                        if not ename or len(ename) < 3:
                            continue
                        if same_proc_id:
                            el_proc = str(
                                el.get("processId") or el.get("process_id") or ""
                            )
                            if el_proc and el_proc != same_proc_id:
                                continue
                        # substring (양방향) 매칭 — 둘 다 길이 3 이상일 때만
                        if ename in gw_text or (len(gw_text) >= 3 and gw_text in ename):
                            candidates.append((eid, ""))
                            seen_targets.add(eid)

                return candidates

            for gid in list(gateway_ids):
                outs = outgoing_by_source.get(gid) or []
                if len(outs) >= 2:
                    continue
                ins = incoming_by_target.get(gid) or []

                # ── 보강 시도
                candidates = _gw_collect_candidates(gid, outs, ins)
                added_count = 0
                if candidates:
                    needed = max(0, 2 - len(outs))
                    for tgt_id, cond in candidates:
                        if added_count >= needed:
                            break
                        if not tgt_id or tgt_id == gid:
                            continue
                        if (gid, tgt_id) in seq_pairs:
                            continue
                        new_seq = {
                            "elementType": "Sequence",
                            "id": f"seq_{gid}_{tgt_id}",
                            "name": cond or "",
                            "source": gid,
                            "target": tgt_id,
                            "condition": cond or "",
                            "_synthesized": "gateway_outgoing_augment",
                        }
                        seqs.append(new_seq)
                        seq_pairs.add((gid, tgt_id))
                        outgoing_by_source.setdefault(gid, []).append(new_seq)
                        incoming_by_target.setdefault(tgt_id, []).append(new_seq)
                        added_count += 1

                outs_after = outgoing_by_source.get(gid) or []
                if added_count > 0:
                    try:
                        logger.info(
                            f"[PROCDEF][GW-AUGMENT] gateway={gid!r} "
                            f"outs_before={len(outs)} added={added_count} "
                            f"outs_after={len(outs_after)} keep_gateway={len(outs_after) >= 2}"
                        )
                    except Exception:
                        pass

                if len(outs_after) >= 2:
                    # 게이트웨이 유지
                    continue

                # ── 보강 실패 → 기존 collapse 로직: incoming.source → outgoing.target 직선화
                out_target = str(outs_after[0].get("target")) if outs_after else ""
                for inc in ins:
                    src = str(inc.get("source") or "")
                    if src and out_target and src != out_target and (src, out_target) not in seq_pairs:
                        seqs.append(
                            {
                                "elementType": "Sequence",
                                "id": f"seq_{src}_{out_target}",
                                "name": "",
                                "source": src,
                                "target": out_target,
                                "condition": "",
                            }
                        )
                        seq_pairs.add((src, out_target))
                removed_gateway_ids.add(gid)
                try:
                    logger.info(
                        f"[PROCDEF][GW-COLLAPSE] gateway={gid!r} "
                        f"outs={len(outs_after)} ins={len(ins)} "
                        f"reason=no_augment_candidate"
                    )
                except Exception:
                    pass

            if removed_gateway_ids:
                seqs = [
                    s
                    for s in seqs
                    if str(s.get("source")) not in removed_gateway_ids and str(s.get("target")) not in removed_gateway_ids
                ]
                elems = [
                    e
                    for e in elems
                    if not (e.get("elementType") == "Gateway" and str(e.get("id")) in removed_gateway_ids)
                ]
                gateway_ids = {str(e.get("id")) for e in elems if e.get("elementType") == "Gateway" and e.get("id")}

        # recompute outgoing after collapse
        outgoing_by_source = {}
        for s in seqs:
            outgoing_by_source.setdefault(str(s.get("source")), []).append(s)

        for gid in gateway_ids:
            outs = outgoing_by_source.get(gid) or []
            if len(outs) <= 1:
                continue
            element_by_id: Dict[str, Dict[str, Any]] = {
                str(e.get("id")): e for e in elems if isinstance(e, dict) and e.get("id")
            }
            gw_el = element_by_id.get(gid) or {}
            gw_name = str(gw_el.get("name") or "").strip()
            gw_desc = str(gw_el.get("description") or "").strip()
            if self._is_placeholder_gateway_name(gw_name):
                gw_el["name"] = self._derive_gateway_name(raw_name=gw_name, description=gw_desc, idx=1)
                gw_name = str(gw_el.get("name") or "").strip()

            def _make_gateway_condition(seq: Dict[str, Any], idx: int) -> str:
                tgt_id = str(seq.get("target") or "").strip()
                tgt = element_by_id.get(tgt_id) or {}
                tgt_name = str(tgt.get("name") or "").strip()
                tgt_desc = str(tgt.get("description") or "").strip()
                hint = tgt_name or tgt_desc
                if hint:
                    hint = re.sub(r"\s+", " ", hint).strip()
                    if len(hint) > 60:
                        hint = hint[:60].rstrip() + "..."
                    # relation label/시스템 엣지명 등 비즈니스 조건이 아닌 문자열은 제거
                    bad = {"has task", "has instruction", "performed_by", "has gateway", "has event"}
                    if self._normalize_text_key(hint) in {self._normalize_text_key(x) for x in bad}:
                        hint = ""
                if hint:
                    return f"{hint}인 경우"
                return f"{gw_name or '해당 조건'} 충족인 경우"

            for j, s in enumerate(outs, start=1):
                cond = str(s.get("condition") or "").strip()
                cond_low = self._normalize_text_key(cond)
                if cond_low in {"hastask", "hasinstruction", "performedby", "hasgateway", "hasevent"}:
                    cond = ""
                if re.match(r"^(조건|분기|case)\d+$", cond_low):
                    cond = ""
                if not cond:
                    if len(outs) == 2:
                        c_true, c_false = self._derive_true_false_conditions(
                            gateway_name=gw_name,
                            gateway_description=gw_desc,
                        )
                        s["condition"] = c_true if j == 1 else c_false
                    else:
                        s["condition"] = _make_gateway_condition(s, j)
                if not str(s.get("name") or "").strip():
                    s["name"] = str(s.get("condition") or "").strip()

        # Merge normalized elements list: keep non-seq + normalized seqs at end (stable parsing)
        non_seq = [e for e in elems if e.get("elementType") != "Sequence"]
        m["elements"] = non_seq + seqs

        # Summary counts for debugging
        try:
            all_elems = m.get("elements") or []
            if isinstance(all_elems, list):
                c_event = sum(1 for e in all_elems if isinstance(e, dict) and e.get("elementType") == "Event")
                c_act = sum(1 for e in all_elems if isinstance(e, dict) and e.get("elementType") == "Activity")
                c_gw = sum(1 for e in all_elems if isinstance(e, dict) and e.get("elementType") == "Gateway")
                c_seq = sum(1 for e in all_elems if isinstance(e, dict) and e.get("elementType") == "Sequence")
                logger.info(
                    f"[PROCDEF][NORMALIZE] end counts: events={c_event} activities={c_act} gateways={c_gw} sequences={c_seq} "
                    f"(process={process_name!r})"
                )
        except Exception:
            pass

        return m

    async def _prepare_assignment_hints_from_extraction(
        self,
        *,
        tenant_id: str,
        process_name: str,
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        (유저 생성/지정 LLM) 단계:
        - Neo4j 추출정보(roles/tasks)를 보고 role별 endpoint/default(사용자/에이전트)와
          activity별 담당 role 힌트를 만든다.
        - 필요 시 에이전트 자동 생성(users insert) + 조직도 반영까지 수행.
        """
        await self._load_org_and_agents(tenant_id)

        # Extract role names from extraction
        role_names: List[str] = []
        roles = extracted.get("roles") or []
        if isinstance(roles, list):
            for r in roles:
                if isinstance(r, dict):
                    rn = str(r.get("name") or r.get("role_name") or "").strip()
                    if rn and rn not in role_names:
                        role_names.append(rn)

        tasks = extracted.get("tasks") or extracted.get("activities") or []
        if isinstance(tasks, list):
            for t in tasks:
                if isinstance(t, dict):
                    rn = str(t.get("role_name") or t.get("role") or "").strip()
                    if rn and rn not in role_names:
                        role_names.append(rn)

        hints_roles: List[Dict[str, Any]] = []

        # Cache by id for quick lookup
        users_by_id = {str(u.get("id")): u for u in (self._users or []) if isinstance(u, dict) and u.get("id")}

        for rn in role_names:
            # 1) existing agent/user by name
            u = self._pick_user_for_role(rn)
            if u and u.get("id"):
                uid = str(u.get("id"))
                # roles.default/endpoint should be array when user id is used (frontend supports both)
                hints_roles.append(
                    {
                        "name": rn,
                        "default": [uid],
                        "endpoint": [uid],
                        "origin": "used",
                    }
                )
                continue

            # 2) existing team
            team_id = (self._org_teams_by_name or {}).get(self._normalize_text_key(rn))
            if team_id:
                hints_roles.append(
                    {
                        "name": rn,
                        "default": [],
                        "endpoint": [team_id],
                        "origin": "used",
                    }
                )
                continue

            # 3) LLM-based: recommend/possibly create agent
            rec = await self._llm_recommend_assignee(
                tenant_id=tenant_id,
                process_name=process_name,
                role_name=rn,
                activities_context=[],
            )
            if isinstance(rec, dict) and str(rec.get("action") or "") == "create_agent":
                create_agent = rec.get("create_agent") or {}
                team_id_for_new = str(create_agent.get("team_id") or rec.get("target_team_id") or "").strip()
                team_name = self._org_team_name_by_id.get(team_id_for_new) or "미분류"
                user_input = str(create_agent.get("user_input") or "").strip() or f"역할 '{rn}' 업무를 수행할 에이전트를 생성해주세요."
                mcp_tools = self._safe_json_loads(os.getenv("MCP_TOOLS_JSON", "")) or {}
                agent_profile = await self._llm_generate_agent_profile(
                    team_name=team_name,
                    user_input=user_input,
                    mcp_tools=mcp_tools,
                )
                created = None
                if agent_profile:
                    created = await self._insert_agent_user(
                        tenant_id=tenant_id,
                        agent_profile=agent_profile,
                        agent_type=str(create_agent.get("agent_type") or "agent"),
                    )
                if created and created.get("id"):
                    if team_id_for_new:
                        await self._update_org_chart_add_member(
                            tenant_id=tenant_id,
                            team_id=team_id_for_new,
                            member_user=created,
                        )
                    uid = str(created.get("id"))
                    users_by_id[uid] = created
                    hints_roles.append(
                        {
                            "name": rn,
                            "default": [uid],
                            "endpoint": [uid],
                            "origin": "created",
                        }
                    )
                    continue

            # fallback: created team role without assignee
            hints_roles.append({"name": rn, "default": [], "endpoint": [], "origin": "created"})

        # Activity hints: role + optional agent id if role endpoint resolves to an agent
        hints_activities: List[Dict[str, Any]] = []
        role_to_agent_id: Dict[str, Optional[str]] = {}
        for r in hints_roles:
            if not isinstance(r, dict):
                continue
            rn = str(r.get("name") or "").strip()
            endpoint = r.get("endpoint") or []
            if isinstance(endpoint, list) and endpoint:
                eid = str(endpoint[0])
                u = users_by_id.get(eid)
                if u and u.get("is_agent") is True:
                    role_to_agent_id[rn] = eid
                else:
                    role_to_agent_id[rn] = None
            else:
                role_to_agent_id[rn] = None

        if isinstance(tasks, list):
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                tid = str(t.get("task_id") or t.get("id") or t.get("name") or "").strip()
                tname = str(t.get("name") or "").strip()
                rn = str(t.get("role_name") or t.get("role") or "").strip()
                agent_id = role_to_agent_id.get(rn)
                hints_activities.append(
                    {
                        "id": tid,
                        "name": tname,
                        "role": rn,
                        "agent": agent_id,
                        "agentMode": "draft" if agent_id else "none",
                        "orchestration": "crewai-action" if agent_id else None,
                    }
                )

        hints = {"roles": hints_roles, "activities": hints_activities}
        hints["simplified"] = self._simplify_assignment_hints(hints)
        return hints

    # =========================================================================
    # (DEPRECATED) 컨설팅 모드 — extracted 중간 변환 단계
    # -------------------------------------------------------------------------
    # 더 이상 사용되지 않는다. 컨설팅 모드는 컨설팅 텍스트 + JSON 생성 규칙으로
    # 곧바로 프로세스 정의를 생성한다 (executor.py 의 컨설팅 분기 참고).
    # 아래 4개 함수는 향후 일괄 삭제 예정.
    # =========================================================================
    def _normalize_consulting_extracted_detail(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 이 만든 컨설팅 추출 결과를 downstream 파이프라인이 기대하는 detail 구조로 정규화."""
        process = raw.get("process") if isinstance(raw.get("process"), dict) else {}

        tasks_in = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
        tasks: List[Dict[str, Any]] = []
        role_names_seen: List[str] = []
        for i, t in enumerate(tasks_in, start=1):
            if not isinstance(t, dict):
                continue
            name = str(t.get("name") or "").strip()
            if not name:
                continue
            tid = str(t.get("task_id") or t.get("id") or f"task_{i}").strip() or f"task_{i}"
            order_val = t.get("task_order")
            if order_val is None:
                order_val = t.get("order")
            try:
                order_int = int(order_val)
            except Exception:
                order_int = i
            role = str(t.get("role") or t.get("performer_role") or "").strip()
            if role and role not in role_names_seen:
                role_names_seen.append(role)
            tasks.append({
                "task_id": tid,
                "name": name,
                "instruction": str(t.get("instruction") or "").strip(),
                "description": str(t.get("description") or "").strip(),
                "role": role,
                "task_order": order_int,
            })

        roles_in = raw.get("roles") if isinstance(raw.get("roles"), list) else []
        roles: List[Dict[str, Any]] = []
        role_names_added: Set[str] = set()
        for i, r in enumerate(roles_in, start=1):
            if isinstance(r, dict):
                rn = str(r.get("name") or r.get("role") or "").strip()
                rid = str(r.get("role_id") or r.get("id") or f"role_{i}").strip() or f"role_{i}"
            elif isinstance(r, str):
                rn = r.strip()
                rid = f"role_{i}"
            else:
                continue
            if rn and rn not in role_names_added:
                role_names_added.add(rn)
                roles.append({"role_id": rid, "name": rn})
        # tasks 에서 등장했지만 roles 에 없는 역할 보강
        for rn in role_names_seen:
            if rn and rn not in role_names_added:
                role_names_added.add(rn)
                roles.append({"role_id": f"role_{len(roles) + 1}", "name": rn})
        if not roles:
            roles = [{"role_id": "role_1", "name": "담당자"}]
        default_role = roles[0]["name"]
        for t in tasks:
            if not t.get("role"):
                t["role"] = default_role

        gateways_in = raw.get("gateways") if isinstance(raw.get("gateways"), list) else []
        gateways: List[Dict[str, Any]] = []
        for i, g in enumerate(gateways_in, start=1):
            if not isinstance(g, dict):
                continue
            gid = str(g.get("gateway_id") or g.get("id") or f"gw_{i}").strip() or f"gw_{i}"
            gateways.append({
                "gateway_id": gid,
                "name": str(g.get("name") or "").strip(),
                "gateway_type": str(g.get("gateway_type") or g.get("type") or "ExclusiveGateway").strip(),
                "condition": str(g.get("condition") or "").strip(),
                "description": str(g.get("description") or "").strip(),
                "role": str(g.get("role") or default_role).strip() or default_role,
            })

        events_in = raw.get("events") if isinstance(raw.get("events"), list) else []
        events: List[Dict[str, Any]] = []
        has_start = has_end = False
        for e in events_in:
            if not isinstance(e, dict):
                continue
            etype = str(e.get("event_type") or e.get("type") or "").strip()
            eid = str(e.get("event_id") or e.get("id") or "").strip()
            low = (etype + " " + eid).lower()
            if "start" in low:
                has_start = True
                eid = eid or "start_event"
                etype = "StartEvent"
            elif "end" in low:
                has_end = True
                eid = eid or "end_event"
                etype = "EndEvent"
            if not eid:
                continue
            events.append({
                "event_id": eid,
                "event_type": etype or "IntermediateEvent",
                "name": str(e.get("name") or "").strip(),
            })
        if not has_start:
            events.insert(0, {"event_id": "start_event", "event_type": "StartEvent", "name": "프로세스 시작"})
        if not has_end:
            events.append({"event_id": "end_event", "event_type": "EndEvent", "name": "프로세스 종료"})

        flows_in = raw.get("sequence_flows") if isinstance(raw.get("sequence_flows"), list) else []
        if not flows_in and isinstance(raw.get("flows"), list):
            flows_in = raw.get("flows")
        flows: List[Dict[str, Any]] = []
        for f in flows_in:
            if not isinstance(f, dict):
                continue
            src = str(f.get("source") or f.get("from_id") or "").strip()
            tgt = str(f.get("target") or f.get("to_id") or "").strip()
            if not src or not tgt:
                continue
            flows.append({
                "source": src,
                "target": tgt,
                "condition": str(f.get("condition") or "").strip(),
            })
        # flow 가 비어 있으면 task_order 순서로 직선 흐름을 합성한다.
        if not flows and tasks:
            ordered = sorted(tasks, key=lambda x: x.get("task_order") or 0)
            chain = ["start_event"] + [t["task_id"] for t in ordered] + ["end_event"]
            for a, b in zip(chain, chain[1:]):
                flows.append({"source": a, "target": b, "condition": ""})

        return {
            "process": {
                "name": str(process.get("name") or "").strip(),
                "description": str(process.get("description") or "").strip(),
            },
            "tasks": tasks,
            "roles": roles,
            "gateways": gateways,
            "events": events,
            "sequence_flows": flows,
        }

    def _fallback_consulting_extracted_detail(
        self,
        *,
        user_request: str,
        consulting_outline: str,
    ) -> Dict[str, Any]:
        """LLM 변환 실패 시: 컨설팅 초안의 번호/불릿 목록을 단계로 파싱한 최소 구조."""
        steps: List[str] = []
        for line in (consulting_outline or "").splitlines():
            s = line.strip()
            if not s:
                continue
            m = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", s)
            if m:
                text = m.group(1).strip()
                # "단계명: 설명" 형태면 앞부분만 단계명으로
                text = re.split(r"[:：]", text, 1)[0].strip() or text
                if text:
                    steps.append(text[:60])
        if not steps:
            base = (user_request or consulting_outline or "업무 처리").strip()
            steps = [f"{base[:40]} 처리"]

        tasks = [
            {
                "task_id": f"task_{i}",
                "name": name,
                "instruction": "",
                "description": "",
                "role": "담당자",
                "task_order": i,
            }
            for i, name in enumerate(steps, start=1)
        ]
        detail = {
            "process": {"name": "", "description": (user_request or "").strip()[:200]},
            "tasks": tasks,
            "roles": [{"role_id": "role_1", "name": "담당자"}],
            "gateways": [],
            "events": [],
            "sequence_flows": [],
        }
        return self._normalize_consulting_extracted_detail(detail)

    def _derive_process_name_from_consulting(self, user_request: str, consulting_outline: str) -> str:
        """프로세스명이 비었을 때 요청/초안에서 적당한 이름을 유도."""
        text = (user_request or "").strip() or (consulting_outline or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return "신규 프로세스"
        name = text[:30].strip()
        if not name.endswith("프로세스"):
            name = f"{name} 프로세스"
        return name

    async def _build_extracted_by_proc_id_from_consulting(
        self,
        *,
        consulting_payload: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """[DEPRECATED] 더 이상 사용되지 않는다.

        컨설팅 모드는 컨설팅 텍스트를 그대로 LLM 입력으로 사용해 프로세스 정의를 생성한다.
        이 함수는 향후 일괄 삭제 예정.
        """
        raise NotImplementedError(
            "_build_extracted_by_proc_id_from_consulting is deprecated; "
            "consulting mode now feeds consulting text directly to the JSON generation prompt."
        )

    async def _generate_processgpt_definition_and_bpmn(
        self,
        *,
        tenant_id: str,
        process_name: str,
        extracted: Dict[str, Any],
        user_request: str,
        consulting_outline: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        (프로세스 생성 LLM) 단계:
        - 프론트 ProcessDefinitionGenerator 프롬프트와 동일한 규칙을 사용해 elements 모델 생성
        - proc_def.definition(런타임) 구조로 변환하여 함께 반환

        IMPORTANT:
        - BPMN XML은 폼 생성/참조정보(inputData) 확장 이후에 최종값으로 생성/저장해야 합니다.
          (초기 생성 후 확장 단계에서 tool/form id 등이 변경되므로, XML을 먼저 만들면 stale 됩니다.)
        """
        # --- Diagnostics: extracted input summary (direct cause for empty elements) ---
        try:
            ex_tasks = extracted.get("tasks") or extracted.get("activities") or []
            ex_roles = extracted.get("roles") or []
            ex_gws = extracted.get("gateways") or []
            ex_events = extracted.get("events") or []
            ex_flows = extracted.get("sequence_flows") or extracted.get("flows") or []
            task_names = []
            if isinstance(ex_tasks, list):
                for t in ex_tasks:
                    if isinstance(t, dict):
                        n = str(t.get("name") or "").strip()
                        if n:
                            task_names.append(n)
            logger.info(
                f"[PROCDEF][INPUT] process={process_name!r} "
                f"tasks={len(ex_tasks) if isinstance(ex_tasks, list) else 'n/a'} "
                f"roles={len(ex_roles) if isinstance(ex_roles, list) else 'n/a'} "
                f"gateways={len(ex_gws) if isinstance(ex_gws, list) else 'n/a'} "
                f"events={len(ex_events) if isinstance(ex_events, list) else 'n/a'} "
                f"flows={len(ex_flows) if isinstance(ex_flows, list) else 'n/a'} "
                f"task_samples={task_names[:5]!r} "
                f"user_request_empty={not bool(str(user_request or '').strip())}"
            )
        except Exception:
            pass

        # 1) 컨설팅 초안 처리:
        #    - 파일 모드: consulting_outline=None → 추출 정보 + 생성 규칙만으로 바로 생성
        #    - 컨설팅 모드: 호출자가 컨설팅 초안을 넘겨주면, extracted 와 함께 생성 LLM 의
        #      추가 근거(grounding)로 사용한다.
        consulting_outline = (consulting_outline or "").strip() or None
        if consulting_outline:
            logger.info(f"[PROCDEF][CONSULTING] consulting outline provided (process={process_name!r})")
        else:
            logger.info(f"[PROCDEF][CONSULTING] skipped (direct generation from extracted, process={process_name!r})")

        # 2) Build prompt inputs for create-only process definition generation
        # NOTE:
        # - This backend is create-only; ask/modification rules are intentionally excluded from the LLM prompt
        #   to avoid ambiguity and {"error":"cannot_comply"} fallbacks.
        extracted_summary = {
            "process_name": process_name,
            "extracted": extracted,
        }
        messages = build_process_definition_messages(
            base_system_prompt="",
            hints_simplified={},
            consulting_outline=consulting_outline,
            extracted_summary=extracted_summary,
            user_request=user_request,
        )
        # 3) Build elements model:
        # - deterministic mode: always from extracted
        # - llm mode: llm first, fallback to extracted when degraded
        if not self._use_llm_procdef_enrich:
            logger.info(f"[PROCDEF][MODE] deterministic (USE_LLM_PROCDEF_ENRICH=false, process={process_name!r})")
            elements_model = self._build_elements_model_from_extracted(
                process_name=process_name,
                extracted=extracted,
            )
        else:
            if not self.openai_client:
                logger.warning(f"[PROCDEF][LLM] openai_client unavailable; fallback to deterministic (process={process_name!r})")
                elements_model = self._build_elements_model_from_extracted(
                    process_name=process_name,
                    extracted=extracted,
                )
            else:
                # LLM 으로 elements 모델 생성.
                # 생성 실패(JSON 미생성 / elements 누락 / 추출 대비 심각히 빈약)면
                # 바로 추출 기반 폴백하지 않고 최대 3회까지 재시도한다.
                # 재시도는 절단(truncation) 대비로 max_tokens 를 키우고, 결정론적
                # 실패를 벗어나도록 temperature 를 조금씩 올린다.
                # 3회 모두 실패할 때만 추출 기반 결정론 폴백으로 넘어간다.
                _gen_attempts = [
                    {"max_tokens": 8000, "temperature": None},
                    {"max_tokens": 11000, "temperature": 0.2},
                    {"max_tokens": 14000, "temperature": 0.4},
                ]
                _n_attempts = len(_gen_attempts)
                elements_model = None
                for _attempt_i, _opt in enumerate(_gen_attempts, 1):
                    try:
                        _candidate = await self._call_openai_process_definition(
                            messages=messages,
                            max_tokens=_opt["max_tokens"],
                            temperature=_opt["temperature"],
                        )
                    except Exception as _gen_e:
                        logger.warning(
                            f"[PROCDEF][LLM] 생성 시도 {_attempt_i}/{_n_attempts} 예외: "
                            f"{type(_gen_e).__name__}: {_gen_e} (process={process_name!r})"
                        )
                        _candidate = None

                    if not isinstance(_candidate, dict):
                        logger.warning(
                            f"[PROCDEF][LLM] 생성 시도 {_attempt_i}/{_n_attempts}: "
                            f"JSON 미생성(not dict) — {'재시도' if _attempt_i < _n_attempts else '재시도 소진'} "
                            f"(process={process_name!r})"
                        )
                        continue

                    _cand_elems = _candidate.get("elements")
                    if not (isinstance(_cand_elems, list) and len(_cand_elems) > 0):
                        logger.warning(
                            f"[PROCDEF][LLM] 생성 시도 {_attempt_i}/{_n_attempts}: "
                            f"elements 누락/빈값 — {'재시도' if _attempt_i < _n_attempts else '재시도 소진'} "
                            f"(process={process_name!r})"
                        )
                        continue

                    # 추출 대비 심각히 빈약하면 그것도 '생성 실패'로 보고 재시도한다.
                    try:
                        _degraded, _deg_reason = self._should_fallback_to_extracted_elements(
                            elements_model=_candidate,
                            extracted=extracted,
                            process_name=process_name,
                        )
                    except Exception as _deg_e:
                        logger.exception(
                            f"[PROCDEF][FALLBACK] fallback decision failed: "
                            f"{type(_deg_e).__name__}: {_deg_e}"
                        )
                        _degraded, _deg_reason = False, ""
                    if _degraded:
                        logger.warning(
                            f"[PROCDEF][LLM] 생성 시도 {_attempt_i}/{_n_attempts}: "
                            f"추출 대비 빈약({_deg_reason}) — "
                            f"{'재시도' if _attempt_i < _n_attempts else '재시도 소진'} "
                            f"(process={process_name!r})"
                        )
                        continue

                    elements_model = _candidate
                    logger.info(
                        f"[PROCDEF][LLM] 생성 성공 (시도 {_attempt_i}/{_n_attempts}, "
                        f"elements_len={len(_cand_elems)}, keys={list(_candidate.keys())}, "
                        f"process={process_name!r})"
                    )
                    break

                if elements_model is None:
                    logger.warning(
                        f"[PROCDEF][FALLBACK] LLM 생성 {_n_attempts}회 모두 실패 -> "
                        f"추출 기반 결정론 폴백 (process={process_name!r})"
                    )
                    elements_model = self._build_elements_model_from_extracted(
                        process_name=process_name,
                        extracted=extracted,
                    )

        # 5) Strict validate/normalize elements model (connectivity + ids + required fields)
        elements_model = self._validate_and_normalize_elements_model(elements_model, process_name=process_name)

        # 5.5) Force proc_def id to UUID (avoid collisions on save)
        # NOTE:
        # - We intentionally IGNORE model-provided processDefinitionId to prevent accidental reuse.
        # - BPMN XML generator does not use this id for <bpmn:process id="..."> (it uses Process_1),
        #   so UUID starting with digits is safe.
        # - 하이픈 없는 형태('-' → '_')로 강제한다. form id 는 "formHandler:<snake(pid)>_..."
        #   처럼 _snake_id 로 만들어져 하이픈이 '_'로 바뀌는데, 프론트(FormWorkItem.vue)는
        #   `${processDefinitionId}_${activity_id}_form` 으로 processDefinitionId 를 '그대로'
        #   이어 붙여 form id 를 재구성한다. processDefinitionId 에 하이픈이 있으면 이
        #   재구성 결과(하이픈)와 실제 form_def.id(언더스코어)가 어긋나 폼/입력값을 못 찾는다.
        #   → 애초에 하이픈 없는 id 로 강제하면 snake 변환 전후가 동일해 항상 일치한다.
        prev_proc_def_id = str(elements_model.get("processDefinitionId") or "").strip()
        forced_proc_def_id = str(uuid.uuid4()).replace("-", "_")
        elements_model["processDefinitionId"] = forced_proc_def_id
        # processDefinitionId 를 새로 강제하면, 그 직전 id 로 이미 만들어진 form id 참조
        # (activity.tool = "formHandler:<snake(pid)>_<activity_id>_form") 가 새 id 와 어긋난다.
        # 이 어긋남이 남으면: form_def.id / todolist.output 키 / activity.tool 은 옛 접두사를
        # 쓰는데 processDefinitionId 만 새 값이라, 프론트(FormWorkItem.vue)가 form id 를
        # `${processDefinitionId}_${activity_id}_form` 으로 재구성할 때 폼을 찾지 못해
        # todolist '입력값'·instance '산출물' 화면에 값이 표시되지 않는다.
        # → 강제 변경 시 activity.tool 의 접두사를 새 id 로 즉시 재작성해 일관성을 유지한다.
        if prev_proc_def_id and prev_proc_def_id != forced_proc_def_id:
            _old_tool_pfx = f"formHandler:{self._snake_id(prev_proc_def_id)}_"
            _new_tool_pfx = f"formHandler:{self._snake_id(forced_proc_def_id)}_"
            _rewired = 0
            for _e in (elements_model.get("elements") or []):
                if not isinstance(_e, dict):
                    continue
                _tool = str(_e.get("tool") or "")
                if _tool.startswith(_old_tool_pfx):
                    _e["tool"] = _new_tool_pfx + _tool[len(_old_tool_pfx):]
                    _rewired += 1
            if _rewired:
                logger.info(
                    f"[PROCDEF] processDefinitionId 강제 변경 "
                    f"({prev_proc_def_id} → {forced_proc_def_id}); "
                    f"form id 접두사 {_rewired}건을 새 id 로 재작성"
                )

        # 6) Convert to runtime definition + enrich + assignment(again, as safety)
        runtime_def = self._elements_model_to_runtime_definition(elements_model)
        # extracted 기준 역할 보정(역할이 하나로 뭉개지는 현상 완화)
        runtime_def = self._apply_extracted_roles_to_runtime_definition(
            runtime_def=runtime_def,
            extracted=extracted,
        )
        runtime_def = self._enrich_process_definition(
            runtime_def,
            process_name=str(runtime_def.get("processDefinitionName") or process_name),
            process_definition_id=str(elements_model.get("processDefinitionId") or runtime_def.get("processDefinitionId")),
        )
        runtime_def = self._backfill_activity_content_from_extracted(
            runtime_def=runtime_def,
            extracted=extracted,
        )
        runtime_def = self._ensure_end_event_connectivity(
            runtime_def,
            process_name=process_name,
        )

        # runtime_def summary (this will show when activities are empty -> start/end-only)
        try:
            acts = runtime_def.get("activities") or []
            evs = runtime_def.get("events") or []
            gws = runtime_def.get("gateways") or []
            seqs = runtime_def.get("sequences") or []
            logger.info(
                f"[PROCDEF][RUNTIME] activities={len(acts) if isinstance(acts, list) else 'n/a'} "
                f"events={len(evs) if isinstance(evs, list) else 'n/a'} "
                f"gateways={len(gws) if isinstance(gws, list) else 'n/a'} "
                f"sequences={len(seqs) if isinstance(seqs, list) else 'n/a'} "
                f"(process={process_name!r})"
            )
        except Exception:
            pass

        # NOTE: 담당자/에이전트 매핑은 forms + inputData 확장 이후 마지막 단계에서 수행 후 저장한다.
        return {"elements_model": elements_model, "definition": runtime_def}

    async def _llm_recommend_assignee(
        self,
        *,
        tenant_id: str,
        process_name: str,
        role_name: str,
        activities_context: List[Dict[str, Any]],
        extracted_context: Optional[Dict[str, Any]] = None,
        allow_create_agent: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        LLM으로 역할 담당자(기존 user/agent 또는 팀) 추천.
        반환 예시:
          {
            "action": "existing_user"|"team"|"create_agent"|"none",
            "target_user_id": "...",
            "target_team_id": "...",
            "confidence": 0.0-1.0,
            "reason": "...",
            "create_agent": { ... }  # action=create_agent일 때만
          }
        """
        if not (self._enable_llm_role_mapping and self.openai_client and self.openai_api_key):
            return None

        # role_name can include hard line breaks from PDF OCR; normalize to keep matching stable
        role_name_clean = " ".join(str(role_name or "").split())
        # Cache key must include process_name to avoid cross-process leakage
        cache_key = self._normalize_text_key(f"{process_name}|{role_name_clean}")
        if cache_key and cache_key in self._role_assignment_cache:
            return self._role_assignment_cache.get(cache_key)

        # Candidate agents MUST come from users table (is_agent=true) only.
        candidates_users = self._get_user_candidates(role_name_clean)
        candidates_teams = self._get_org_team_candidates(role_name_clean)

        # also provide team members for candidate teams
        team_members: Dict[str, List[Dict[str, Any]]] = {}
        users_by_id = {str(u.get("id")): u for u in (self._users or []) if isinstance(u, dict) and u.get("id")}
        for t in candidates_teams:
            tid = str(t.get("team_id") or "")
            if not tid:
                continue
            mids = (self._org_members_by_team_id or {}).get(tid) or []
            mlist: List[Dict[str, Any]] = []
            for mid in mids[:30]:
                u = users_by_id.get(str(mid))
                if not u:
                    continue
                is_agent = bool(u.get("is_agent") is True)
                desc = str(u.get("description") or "").strip()
                goal = str(u.get("goal") or "").strip()
                persona = str(u.get("persona") or "").strip()
                if len(desc) > 220:
                    desc = desc[:220] + "…"
                if len(goal) > 180:
                    goal = goal[:180] + "…"
                if len(persona) > 220:
                    persona = persona[:220] + "…"
                mlist.append(
                    {
                        "id": str(u.get("id") or ""),
                        "username": str(u.get("username") or ""),
                        "role": str(u.get("role") or ""),
                        "is_agent": is_agent,
                        "agent_type": str(u.get("agent_type") or ""),
                        "alias": str(u.get("alias") or ""),
                        "description": (desc if is_agent else ""),
                        "goal": (goal if is_agent else ""),
                        "persona": (persona if is_agent else ""),
                    }
                )
            if mlist:
                team_members[tid] = mlist

        create_agent_rule = (
            "- 후보 에이전트가 없다면, 자동화 이득이 큰 태스크에 한해 create_agent를 선택하세요(없으면 none).\n"
            if allow_create_agent
            else "- IMPORTANT: 이번 호출에서는 create_agent 선택이 금지됩니다. action은 existing_user/team/none 중에서만 선택하세요.\n"
        )

        system_prompt = (
            "당신은 BPM 프로세스 정의에서 '역할(Role)'을 시스템의 실제 담당자(User/Agent) 또는 팀(조직도)으로 매핑하는 전문가입니다.\n"
            "\n"
            "당신의 목표는 2가지입니다.\n"
            "1) 태스크(activities_context)의 설명/지침을 보고, **자동화하면 이득인 경우에만** 에이전트(기존 agent user)를 매핑한다.\n"
            "2) 이미 조직도/유저 목록에 유사한 agent가 있으면 **반드시 재사용**하고, 중복 에이전트를 새로 만들지 않는다.\n"
            "\n"
            "입력 데이터 설명:\n"
            "- activities_context에는 activityName/instruction/description/tool이 포함됩니다. 이것이 '태스크 설명'입니다.\n"
            "- users/team_members의 각 항목에는 username/role/alias/description/goal/persona가 포함될 수 있습니다. 이것이 '에이전트 설명'입니다.\n"
            "\n"
            "중요 규칙:\n"
            "- existing_user/team을 선택하는 경우에는 반드시 제공된 후보 목록(users/teams/team_members) 안에서만 선택해야 합니다.\n"
            "- existing_user를 선택할 때는 target_user_id가 **is_agent=true인 사용자**여야 합니다. (사람 사용자 is_agent=false는 선택 금지)\n"
            "- IMPORTANT: users 후보는 users 테이블의 '에이전트(is_agent=true)'만 포함합니다. 사람 사용자는 후보가 아닙니다.\n"
            + create_agent_rule
            + "- 아래 조건 중 하나라도 강하게 해당되면 action=none 으로 두세요(에이전트 미매핑):\n"
            + "  - 사람이 직접 해야 하는 신청/등록/접수/결제/입금/서명/대면/회의/면담 진행/출석/실물 확인/법적 승인 등\n"
            + "  - 최종 승인/책임 소재가 중요한 의사결정(정책/권한/결재)으로 자동화가 부적절한 경우\n"
            + "- 반대로 아래 유형은 자동화 이득이 큰 편이므로, 유사한 agent가 있으면 적극적으로 existing_user를 선택하세요:\n"
            + "  - 문서/콘텐츠 생성(초안 작성, 퀴즈 생성, 안내문 생성), 요약/정리/분류, 검증/체크리스트, 검색/조회, 채점/스코어링, 결과 취합/리포트\n"
            + "- 에이전트 매칭은 반드시 **태스크 설명 ↔ 에이전트 설명**을 비교해 수행하세요.\n"
            + "  - (나쁜 매칭) 공통 키워드 1~2개(예: '평가')만으로 선택\n"
            + "  - (좋은 매칭) '사전평가/퀴즈/문항'처럼 구체 업무가 겹치고, agent 설명에도 같은 업무가 명시됨\n"
            + "- create_agent는 정말 최후의 수단입니다.\n"
            + "  - activities_context가 자동화 이득이 크고,\n"
            + "  - users/team_members 후보 중 어떤 agent도 태스크를 제대로 커버하지 못할 때만 선택하세요.\n"
            + "  - 유사 agent가 존재한다면(이름/역할/설명에서 업무가 겹친다면) **절대 create_agent를 선택하지 마세요.**\n"
            + "- create_agent를 선택하는 경우에도 생성될 에이전트는 **너무 단일 태스크 전용으로 쪼개지지 않도록** '중간 정도 범위(상세도 6/10)'로 설계하세요.\n"
            + "\n"
            + "예시(반드시 이런 방향으로 판단):\n"
            + "1) 태스크: '사전 평가 생성' / 설명: '사전평가 퀴즈 문항을 생성하고 난이도/정답을 검증'\n"
            + "   - 기존 agent 후보에 '사전평가 퀴즈 메이커'가 있으면 => action=existing_user (그 agent)\n"
            + "   - '강의 평가 봇'처럼 '평가'만 겹치는 agent는 선택하지 않음\n"
            + "2) 태스크: '인터뷰 검증' / 설명: '면접 질문/답변을 기준에 따라 검증하고 리포트 생성'\n"
            + "   - 기존 agent 후보에 '인터뷰 검증 에이전트'가 있으면 => action=existing_user (그 agent)\n"
            + "3) 태스크: '수강 신청' / 설명: '수강자가 직접 신청서를 제출하고 승인 대기'\n"
            + "   - 자동화(대리 신청)는 부적절 => action=none\n"
            + "\n"
            + "- 확신이 낮으면 none을 반환하세요.\n"
            + "- 출력은 JSON ONLY 입니다.\n"
        )

        def _brief_extracted_for_prompt(ex: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            """LLM 입력 토큰을 과도하게 늘리지 않도록 extracted를 간단 요약."""
            if not isinstance(ex, dict):
                return {}

            def _brief_list(items: Any, *, keys: List[str], limit: int) -> List[Dict[str, Any]]:
                if not isinstance(items, list):
                    return []
                out: List[Dict[str, Any]] = []
                for it in items[: max(0, int(limit))]:
                    if not isinstance(it, dict):
                        continue
                    row: Dict[str, Any] = {}
                    for k in keys:
                        v = it.get(k)
                        if v is None:
                            continue
                        # keep compact strings only
                        if isinstance(v, str):
                            v = v.strip()
                            if len(v) > 240:
                                v = v[:240] + "…"
                        row[k] = v
                    if row:
                        out.append(row)
                return out

            return {
                "process": ex.get("process") if isinstance(ex.get("process"), dict) else {},
                "roles": _brief_list(ex.get("roles"), keys=["name", "role_name", "description"], limit=30),
                "tasks": _brief_list(ex.get("tasks") or ex.get("activities"), keys=["id", "task_id", "name", "role", "role_name", "description", "instruction"], limit=50),
                "events": _brief_list(ex.get("events"), keys=["id", "name", "eventType", "description"], limit=20),
                "gateways": _brief_list(ex.get("gateways"), keys=["id", "name", "gatewayType", "description"], limit=20),
                "sequence_flows": _brief_list(ex.get("sequence_flows") or ex.get("flows"), keys=["source", "target", "condition"], limit=60),
            }

        user_prompt = (
            f"테넌트: {tenant_id}\n"
            f"프로세스: {process_name}\n"
            f"매핑할 역할명: {role_name_clean}\n\n"
            f"이 역할이 수행하는 태스크 컨텍스트(요약):\n{json.dumps(activities_context[:15], ensure_ascii=False)}\n\n"
            + (
                (
                    "추출된 원문/Neo4j 정보(요약, 참고용):\n"
                    f"{json.dumps(_brief_extracted_for_prompt(extracted_context), ensure_ascii=False)}\n\n"
                )
                if isinstance(extracted_context, dict)
                else ""
            )
            + f"users 후보(최대 30, agents 포함):\n{json.dumps(candidates_users, ensure_ascii=False)}\n\n"
            + f"teams 후보(최대 30):\n{json.dumps(candidates_teams, ensure_ascii=False)}\n\n"
            + f"team_members(팀별 멤버/에이전트 후보, 없을 수 있음):\n{json.dumps(team_members, ensure_ascii=False)}\n\n"
            + "다음 JSON 형식으로만 응답하세요:\n"
            + "{\n"
            + '  "action": "existing_user" | "team" | "create_agent" | "none",\n'
            + '  "target_user_id": "existing_user일 때만. users 후보의 id",\n'
            + '  "target_team_id": "team/create_agent일 때 권장. teams 후보의 team_id (없으면 빈 문자열 가능)",\n'
            + '  "confidence": 0.0,  // 0~1 숫자 (반드시 숫자)\n'
            + '  "reason": "한두 문장 근거",\n'
            + '  "create_agent": {\n'
            + '    "team_id": "생성 에이전트를 소속시킬 team_id (가능하면, 없으면 빈 문자열)",\n'
            + '    "user_input": "OrganizationAgentGenerator에 넣을 사용자 요구사항(한국어, 3~6문장). 단일 태스크 전용이 아니라 관련 업무를 포괄하는 중간 범위(6/10)로 작성",\n'
            + '    "agent_type": "agent" \n'
            + "  }\n"
            + "}\n"
        )

        logger.info(
            f"[ASSIGN] recommend role={role_name_clean!r} candidates: users={len(candidates_users)} teams={len(candidates_teams)} team_members={len(team_members)}"
        )
        obj = await self._call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=900,
            model=self.user_mapping_model,
            temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")),
        )
        if not isinstance(obj, dict):
            return None

        def _is_agent_user_id_local(user_id: str) -> bool:
            u = users_by_id.get(str(user_id) or "")
            return bool(u and u.get("is_agent") is True and u.get("id"))

        def _looks_automatable_from_ctx(ctx: List[Dict[str, Any]]) -> bool:
            """activities_context 기반으로 '자동화 가치'가 있는지 매우 가볍게 판단(폴백용)."""
            if not isinstance(ctx, list) or not ctx:
                return False
            text = " ".join(
                [
                    str(x.get("activityName") or "")
                    + " "
                    + str(x.get("instruction") or "")
                    + " "
                    + str(x.get("description") or "")
                    for x in ctx[:8]
                    if isinstance(x, dict)
                ]
            )
            key = self._normalize_text_key(text)
            if not key:
                return False
            human_kws = [
                "신청", "등록", "접수", "제출", "결재", "결제", "입금", "납부", "승인", "서명",
                "대면", "회의", "면담", "전화", "방문", "출석", "참석", "수령", "발급", "실물", "현장",
            ]
            auto_kws = [
                "자동", "에이전트", "봇", "생성", "요약", "정리", "분석", "검증", "추출", "조회", "검색",
                "분류", "추천", "채점", "퀴즈", "문항", "리포트", "보고서", "취합", "집계",
            ]
            if any(self._normalize_text_key(k) in key for k in human_kws):
                # if explicit automation hint exists, allow it
                return any(self._normalize_text_key(k) in key for k in auto_kws)
            return any(self._normalize_text_key(k) in key for k in auto_kws)

        # Post-validate against provided candidates to prevent hallucinated assignments.
        # - If model says existing_user but provides an invalid/unknown id, force action=none.
        # - Same for team id.
        action_raw = str(obj.get("action") or "").strip()
        cand_user_ids = {str(u.get("id")) for u in (candidates_users or []) if isinstance(u, dict) and u.get("id")}
        cand_team_ids = {str(t.get("team_id")) for t in (candidates_teams or []) if isinstance(t, dict) and t.get("team_id")}

        if action_raw == "existing_user":
            target_user_id = str(obj.get("target_user_id") or "").strip()
            invalid = (not target_user_id) or (cand_user_ids and target_user_id not in cand_user_ids) or (not _is_agent_user_id_local(target_user_id))
            if invalid:
                # If no suitable candidate agent exists, and task looks automatable, prefer create_agent.
                if allow_create_agent and (not cand_user_ids) and _looks_automatable_from_ctx(activities_context):
                    obj["action"] = "create_agent"
                    obj["target_user_id"] = ""
                    obj["target_team_id"] = ""
                    ca = obj.get("create_agent") if isinstance(obj.get("create_agent"), dict) else {}
                    if not isinstance(ca, dict):
                        ca = {}
                    ca.setdefault("team_id", "")
                    ca.setdefault(
                        "user_input",
                        (
                            "다음 업무를 자동화할 에이전트를 생성해주세요.\n"
                            f"- 프로세스: {process_name}\n"
                            f"- 역할/컨텍스트: {role_name_clean}\n"
                            f"- 태스크: {json.dumps(activities_context[:3], ensure_ascii=False)}\n"
                            "사용자에게 필요한 입력이 있으면 확인을 요청하고, 결과를 정리/검증해 주세요."
                        ),
                    )
                    ca.setdefault("agent_type", "agent")
                    obj["create_agent"] = ca
                    obj["reason"] = (
                        "후보 에이전트(users.is_agent=true) 중 적절한 대상이 없어, 자동화 가치가 있는 태스크로 판단되어 create_agent로 전환했습니다."
                    )
                else:
                    obj["action"] = "none"
                    obj["target_user_id"] = ""
                    obj["target_team_id"] = ""
                    obj["reason"] = (
                        "existing_user로 매핑하려 했으나 target_user_id가 후보(agents)에 없거나 유효하지 않아 미매핑(none) 처리했습니다."
                    )
        elif action_raw == "team":
            target_team_id = str(obj.get("target_team_id") or "").strip()
            if (not target_team_id) or (cand_team_ids and target_team_id not in cand_team_ids):
                obj["action"] = "none"
                obj["target_team_id"] = ""
                obj["confidence"] = float(obj.get("confidence") or 0.0) if str(obj.get("confidence") or "").strip() else 0.0
                obj["reason"] = "team으로 매핑하려 했으나 target_team_id가 후보(teams)에 없어 미매핑(none) 처리했습니다."

        # If create_agent is not allowed in this call, force it to none.
        if not allow_create_agent and str(obj.get("action") or "") == "create_agent":
            obj["action"] = "none"
            obj["target_user_id"] = ""
            obj["target_team_id"] = ""
            obj["reason"] = "human_required 태스크이므로 신규 에이전트 생성(create_agent)은 금지되어 미매핑(none) 처리했습니다."

        # basic validation + threshold
        conf = obj.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
            logger.warning(f"[ASSIGN] role={role_name_clean!r} LLM response missing/invalid confidence. keys={list(obj.keys())}")
        if conf_f < self._llm_assignment_min_conf:
            prev_action = str(obj.get("action") or "")
            # do not block create_agent purely by confidence: if model explicitly requests creation,
            # let downstream creation pipeline decide (it can still fail safely).
            if str(obj.get("action") or "") != "create_agent":
                obj["action"] = "none"
                logger.info(
                    f"[ASSIGN] role={role_name_clean!r} LLM confidence {conf_f:.2f} < {self._llm_assignment_min_conf:.2f} -> action forced to none (was {prev_action!r})"
                )

        if cache_key:
            self._role_assignment_cache[cache_key] = obj
        return obj

    async def _llm_plan_assignments_for_process(
        self,
        *,
        tenant_id: str,
        process_name: str,
        proc_json: Dict[str, Any],
        extracted_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        프로세스 "단건" 맥락으로 자동화 가능 여부 + 담당자(에이전트) 매핑/생성 계획을 한번에 수립합니다.

        목표:
        - 태스크별로 "자동화 불가/부분 자동화/자동화 가능"을 판단
        - 자동화(부분 자동화 포함) 가치가 있으면:
          1) users 테이블에 존재하는 에이전트(is_agent=true) 중 적합한 것을 매핑(existing_user)
          2) 없으면 create_agent로 신규 생성 계획을 제안
        - 자동화 불가(사람의 의사/행동이 필수)면 none

        반환(JSON):
        {
          "decisions": [
            {
              "activity_id": "...",
              "automation": "none" | "partial" | "full",
              "action": "existing_user" | "create_agent" | "none",
              "target_user_id": "",
              "confidence": 0.0,
              "reason": "...",
              "create_agent": { "team_id": "", "user_input": "...", "agent_type": "agent" }
            }
          ]
        }
        """
        if not (self._enable_llm_role_mapping and self.openai_client and self.openai_api_key):
            return None
        if not isinstance(proc_json, dict):
            return None

        activities = proc_json.get("activities") or []
        sequences = proc_json.get("sequences") or []
        if not isinstance(activities, list):
            activities = []
        if not isinstance(sequences, list):
            sequences = []

        # Candidate agents MUST come from users table (is_agent=true) only.
        agents_payload: List[Dict[str, Any]] = []
        for a in (self._agents or [])[:80]:
            if not isinstance(a, dict) or not a.get("id"):
                continue
            # keep compact text fields
            desc = str(a.get("description") or "").strip()
            goal = str(a.get("goal") or "").strip()
            persona = str(a.get("persona") or "").strip()
            if len(desc) > 220:
                desc = desc[:220] + "…"
            if len(goal) > 180:
                goal = goal[:180] + "…"
            if len(persona) > 220:
                persona = persona[:220] + "…"
            agents_payload.append(
                {
                    "id": str(a.get("id") or ""),
                    "username": str(a.get("username") or ""),
                    "role": str(a.get("role") or ""),
                    "alias": str(a.get("alias") or ""),
                    "description": desc,
                    "goal": goal,
                    "persona": persona,
                    "agent_type": str(a.get("agent_type") or ""),
                }
            )

        tasks_payload: List[Dict[str, Any]] = []
        for t in activities[:200]:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or "").strip()
            if not tid:
                continue
            tasks_payload.append(
                {
                    "id": tid,
                    "name": str(t.get("name") or ""),
                    "role": str(t.get("role") or ""),
                    "description": str(t.get("description") or ""),
                    "instruction": str(t.get("instruction") or ""),
                    "tool": str(t.get("tool") or ""),
                }
            )

        flows_payload: List[Dict[str, Any]] = []
        for s in sequences[:250]:
            if not isinstance(s, dict):
                continue
            flows_payload.append(
                {
                    "source": str(s.get("source") or ""),
                    "target": str(s.get("target") or ""),
                    "condition": str(s.get("condition") or ""),
                }
            )

        # small extracted summary (reuse helper from _llm_recommend_assignee via local copy)
        def _brief_extracted(ex: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(ex, dict):
                return {}
            # keep only the essentials for automation judgment
            out: Dict[str, Any] = {}
            for k in ("process",):
                if isinstance(ex.get(k), dict):
                    out[k] = ex.get(k)
            for k in ("roles", "tasks", "activities"):
                v = ex.get(k)
                if isinstance(v, list):
                    out[k] = v[:40]
            return out

        system_prompt = (
            "당신은 '생성된 특정 프로세스(단건)'를 보고, 각 태스크를 자동화할지/말지와 담당 에이전트 매핑을 설계하는 전문가입니다.\n"
            "\n"
            "핵심 원칙(중요):\n"
            "1) 이 판단은 '이 프로세스 단건'의 맥락으로만 하세요. 넓게 일반화하지 마세요.\n"
            "2) 자동화는 '풀 자동화' 뿐 아니라 '부분 자동화(검증/채점/정리/초안)'도 포함합니다.\n"
            "3) people must do:\n"
            "   - 사람이 '의사/선호'를 결정해야 하는 선택(예: 어떤 강의를 듣고 싶은지 선택) 또는\n"
            "   - 학생/신청자가 직접 수행해야 하는 행위(예: 퀴즈 풀이/답변 제출)\n"
            "   => 이런 태스크는 automation=none, action=none\n"
            "4) 자동화(부분 자동화 포함)가 유의미하면 담당 에이전트를 붙입니다:\n"
            "   - 먼저, 후보 에이전트(agents 목록: users.is_agent=true) 중에서 적합한 것을 existing_user로 선택\n"
            "   - 적합한 에이전트가 없으면 create_agent로 신규 생성 계획을 세움\n"
            "5) existing_user를 선택할 때는 반드시 agents 후보 목록의 id만 쓸 수 있습니다(목록 밖 선택 금지).\n"
            "\n"
            "수강신청 프로세스 예시 기준(참고):\n"
            "- 수강 신청: 사람이 어떤 강의를 듣고 싶은지 정하고 신청 => 자동화 불가\n"
            "- 사전평가 생성: 기준에 따라 퀴즈 문항 생성/검증 가능 + '사전평가 퀴즈메이커'가 있으면 매핑\n"
            "- 인터뷰 진행: 학생이 직접 퀴즈 풀이/답변 제출 => 자동화 불가\n"
            "- 인터뷰 검토: 사람이 최종 검토하되 정답/오답 검증/채점/요약은 자동화 가능 + '인터뷰 검토 에이전트'가 있으면 매핑\n"
            "- 승인 여부 결정: 기준(점수/조건)으로 자동 승인/반려 가능하면 자동화 가능. 에이전트가 없으면 생성\n"
            "\n"
            "출력은 JSON ONLY 입니다.\n"
        )

        user_prompt = (
            f"테넌트: {tenant_id}\n"
            f"프로세스명: {process_name}\n\n"
            f"프로세스 정의(태스크 목록):\n{json.dumps(tasks_payload, ensure_ascii=False)}\n\n"
            f"프로세스 흐름(시퀀스):\n{json.dumps(flows_payload, ensure_ascii=False)}\n\n"
            f"후보 에이전트 목록(users.is_agent=true):\n{json.dumps(agents_payload, ensure_ascii=False)}\n\n"
            + (
                f"추출된 원문/Neo4j 정보(요약):\n{json.dumps(_brief_extracted(extracted_context), ensure_ascii=False)}\n\n"
                if isinstance(extracted_context, dict)
                else ""
            )
            + "다음 형식으로만 응답하세요:\n"
            + "{\n"
            + '  "decisions": [\n'
            + '    {\n'
            + '      "activity_id": "tasks.id 중 하나",\n'
            + '      "automation": "none" | "partial" | "full",\n'
            + '      "action": "existing_user" | "create_agent" | "none",\n'
            + '      "target_user_id": "action=existing_user일 때만, agents 후보의 id",\n'
            + '      "confidence": 0.0,\n'
            + '      "reason": "한두 문장 근거",\n'
            + '      "create_agent": {\n'
            + '        "team_id": "",\n'
            + '        "user_input": "새 에이전트 생성 요구사항(한국어, 3~6문장). 해당 태스크의 자동화 기준/입력/출력/검증을 포함",\n'
            + '        "agent_type": "agent"\n'
            + "      }\n"
            + "    }\n"
            + "  ]\n"
            + "}\n"
        )

        obj = await self._call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(os.getenv("LLM_ASSIGNMENT_PROCESS_MAX_TOKENS", "1400")),
            model=self.user_mapping_model,
            temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")),
        )
        if not isinstance(obj, dict):
            return None
        # Parse recovery: model may wrap decisions inside data/result/plan or output partial shapes.
        decisions = obj.get("decisions")
        if not isinstance(decisions, list):
            for k in ("data", "result", "plan", "output"):
                nested = obj.get(k)
                if isinstance(nested, dict) and isinstance(nested.get("decisions"), list):
                    decisions = nested.get("decisions")
                    break
        if not isinstance(decisions, list):
            return None

        normalized_decisions: List[Dict[str, Any]] = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            nd = dict(d)
            # accept alternate keys from imperfect outputs
            if not str(nd.get("activity_id") or "").strip():
                nd["activity_id"] = (
                    nd.get("activityId")
                    or nd.get("task_id")
                    or nd.get("taskId")
                    or ""
                )
            if not str(nd.get("action") or "").strip():
                nd["action"] = "none"
            normalized_decisions.append(nd)

        if not normalized_decisions:
            return None
        return {"decisions": normalized_decisions}

    async def _llm_generate_agent_profile(
        self,
        *,
        team_name: str,
        user_input: str,
        mcp_tools: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """OrganizationAgentGenerator.js 프롬프트 스타일로 에이전트 프로필 생성(JSON)."""
        if not self.openai_client:
            return None
        mcp_tools = mcp_tools or {}
        mcp_tools_text = json.dumps(mcp_tools, ensure_ascii=False)
        system_prompt = (
            "당신은 조직에서 사용할 AI 에이전트의 정보를 생성하는 전문가입니다.\n"
            f'사용자가 입력한 요구사항을 바탕으로 "{team_name}" 팀에 적합한 에이전트의 상세 정보를 JSON 형식으로 생성해주세요.\n\n'
            "다음 형식에 맞춰 응답해주세요:\n\n"
            "{\n"
            '  "name": "에이전트의 이름 (한국어)",\n'
            '  "role": "에이전트의 역할 (간단명료하게)",\n'
            '  "goal": "에이전트의 목표 (구체적이고 측정 가능하게)",\n'
            '  "persona": "에이전트의 성격과 특징 (상세하게 기술)",\n'
            '  "tools": "필요한 MCP 도구들 (쉼표로 구분)"\n'
            "}\n\n"
            "## 지침:\n"
            "1. name은 한국어로 직관적이고 명확하게\n"
            "2. role은 한 문장으로 핵심 역할만\n"
            "3. goal은 SMART 원칙에 따라 구체적이고 측정 가능하게\n"
            "4. persona는 에이전트의 성격, 말투, 전문성 등을 포함하여 상세히\n"
            "5. tools는 업무 수행에 필요한 MCP 도구들을 쉼표로 구분하여 나열, 도구는 우리 회사 MCP 도구 목록에 있는 도구만 사용할 수 있습니다.\n\n"
            "6. (중요) 에이전트는 너무 세분화된 '단일 태스크 전용'으로 만들지 마세요.\n"
            "   - 상세도 기준 1~10 중 **6 정도**로, 관련 업무를 묶어 포괄하는 형태가 좋습니다.\n"
            "   - 예: '수강 신청 도우미', '수강 관리 도우미', '강의 개설 도우미'처럼 과도하게 쪼개지 말고,\n"
            "         가능하면 '교육/수강 운영 도우미'처럼 하나로 포괄하세요.\n"
            "   - 단, 너무 광범위한 전사 공용 에이전트(예: '만능 도우미')도 피하세요.\n\n"
            f"도구 목록:\n{mcp_tools_text}\n\n"
            "반드시 JSON ONLY로 응답하세요.\n"
        )
        user_prompt = (
            "## 팀 컨텍스트:\n"
            f"- 소속 팀: {team_name}\n"
            f'- "{team_name}" 팀의 업무 특성과 목표를 고려하여 에이전트를 설계해주세요\n'
            "- 팀 내에서 실제로 활용 가능하고 업무 효율성을 높일 수 있는 에이전트여야 합니다\n"
            "- 팀원들과의 협업과 소통을 원활하게 도울 수 있는 특성을 포함해주세요\n\n"
            f"사용자 요구사항: {user_input}\n"
        )
        obj = await self._call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=900,
            model=self.user_mapping_model,
            temperature=float(os.getenv("LLM_ASSIGNMENT_TEMPERATURE", "0.0")),
        )
        if not isinstance(obj, dict):
            return None
        return obj

    async def _insert_agent_user(
        self,
        *,
        tenant_id: str,
        agent_profile: Dict[str, Any],
        agent_type: str = "agent",
    ) -> Optional[Dict[str, Any]]:
        """users 테이블에 에이전트를 insert 하고, 성공 시 row(dict)를 반환."""
        if not self.supabase_client:
            return None
        try:
            new_id = str(uuid.uuid4())
            username = str(agent_profile.get("name") or "").strip() or "자동생성 에이전트"
            role = str(agent_profile.get("role") or "").strip()
            goal = str(agent_profile.get("goal") or "").strip()
            persona = str(agent_profile.get("persona") or "").strip()
            tools = str(agent_profile.get("tools") or "").strip()

            # 중복 생성 방지: username/role이 거의 같은 agent가 이미 있으면 그걸 재사용
            key_name = self._normalize_text_key(username)
            key_role = self._normalize_text_key(role)
            for u in (self._agents or []):
                if not isinstance(u, dict):
                    continue
                if key_name and self._normalize_text_key(u.get("username")) == key_name:
                    return u
                if key_role and self._normalize_text_key(u.get("role")) == key_role:
                    return u
                # fuzzy reuse: if role/name is largely contained, reuse (prevents micro-agent explosion)
                exist_role = self._normalize_text_key(u.get("role")) or ""
                exist_name = self._normalize_text_key(u.get("username")) or ""
                if key_role and exist_role and (key_role in exist_role or exist_role in key_role) and (len(key_role) >= 4 or len(exist_role) >= 4):
                    return u
                if key_name and exist_name and (key_name in exist_name or exist_name in key_name) and (len(key_name) >= 4 or len(exist_name) >= 4):
                    return u

            row = {
                "id": new_id,
                "tenant_id": tenant_id,
                "username": username,
                "role": role,
                "goal": goal,
                "persona": persona,
                "tools": tools,
                "is_agent": True,
                "agent_type": agent_type,
                "model": os.getenv("DEFAULT_NEW_AGENT_MODEL", self.openai_model),
                "alias": None,
                "endpoint": None,
                "description": None,
                "skills": None,
            }
            self.supabase_client.table("users").insert(row).execute()
            logger.info(f"[ASSIGN] users insert(agent) ok: id={new_id} username={username!r} role={role!r}")

            # 캐시 갱신
            self._users.append(row)
            self._agents.append(row)
            return row
        except Exception as e:
            logger.warning(f"[WARN] users insert(agent) failed: {e}")
            return None

    async def _update_org_chart_add_member(
        self,
        *,
        tenant_id: str,
        team_id: str,
        member_user: Dict[str, Any],
    ) -> bool:
        """configuration(key=organization)에 에이전트/사용자 노드를 팀 children에 추가(가능한 경우)."""
        if not self.supabase_client:
            return False
        if not self._org_chart or not team_id:
            return False
        try:
            chart = self._org_chart

            def walk(node: Any) -> bool:
                if not node or not isinstance(node, dict):
                    return False
                if str(node.get("id") or "") == str(team_id):
                    children = node.get("children")
                    if not isinstance(children, list):
                        children = []
                        node["children"] = children
                    # 이미 있으면 skip
                    mid = str(member_user.get("id") or "")
                    for ch in children:
                        if isinstance(ch, dict) and str(ch.get("id") or "") == mid:
                            return True
                    # 프론트의 OrganizationAddDialog가 push하는 형태를 따라감
                    child_node = {
                        "id": mid,
                        "name": str(member_user.get("username") or ""),
                        "data": {
                            "id": mid,
                            "name": str(member_user.get("username") or ""),
                            "username": str(member_user.get("username") or ""),
                            "role": str(member_user.get("role") or ""),
                            "goal": member_user.get("goal"),
                            "persona": member_user.get("persona"),
                            "endpoint": member_user.get("endpoint"),
                            "description": member_user.get("description"),
                            "skills": member_user.get("skills"),
                            "model": member_user.get("model"),
                            "alias": member_user.get("alias"),
                            "tools": member_user.get("tools"),
                            "isAgent": True,
                            "is_agent": True,
                            "agent_type": member_user.get("agent_type"),
                            "type": member_user.get("agent_type"),
                        },
                    }
                    children.append(child_node)
                    return True

                children = node.get("children") or []
                if isinstance(children, list):
                    for ch in children:
                        if walk(ch):
                            return True
                return False

            updated = walk(chart)
            if not updated:
                return False

            # configuration 업데이트 (uuid가 있으면 uuid로, 없으면 key+tenant_id 기준)
            # NOTE: value 전체를 덮어쓰지 않고, 기존 value가 있으면 chart만 교체합니다.
            value_root: Dict[str, Any] = {}
            if isinstance(self._org_value, dict):
                value_root = dict(self._org_value)
            # chart key가 없었던 레거시 구조면, chart만 가진 value로 저장
            value_root["chart"] = chart

            payload = {"key": "organization", "tenant_id": tenant_id, "value": value_root}
            if self._org_config_uuid:
                self.supabase_client.table("configuration").update(payload).eq("uuid", self._org_config_uuid).execute()
            else:
                # fallback: key+tenant_id
                existing = (
                    self.supabase_client.table("configuration")
                    .select("uuid")
                    .eq("key", "organization")
                    .eq("tenant_id", tenant_id)
                    .execute()
                )
                if existing.data and len(existing.data) > 0:
                    self._org_config_uuid = existing.data[0].get("uuid")
                    if self._org_config_uuid:
                        self.supabase_client.table("configuration").update(payload).eq("uuid", self._org_config_uuid).execute()
                else:
                    self.supabase_client.table("configuration").insert(payload).execute()

            # 인덱스 재생성(다음 매핑에 반영)
            self._org_value = value_root
            idx = self._index_org_chart(chart)
            self._org_teams_by_name = idx.get("teams_by_name") or {}
            self._org_team_name_by_id = idx.get("team_name_by_id") or {}
            self._org_members_by_team_id = idx.get("members_by_team_id") or {}
            return True
        except Exception as e:
            logger.warning(f"[WARN] organization chart update failed: {e}")
            return False

    async def _apply_assignment_and_maybe_create_agents(
        self,
        *,
        proc_json: Dict[str, Any],
        tenant_id: str,
        process_name: str,
        extracted: Optional[Dict[str, Any]] = None,
    ) -> None:
        """lane(role)-skill 규칙 기반 에이전트 배정/생성."""
        try:
            await self._assign_or_create_agents_by_lane_skill(
                proc_json=proc_json,
                tenant_id=tenant_id,
                process_name=process_name,
            )
        except Exception as e:
            logger.warning(f"[WARN] lane-skill assignment failed: {e}")

    # =========================================================================
    # HITL 사전 후보 추출 — 실제 생성/INSERT 없이 사용자에게 보여줄 후보만 만든다.
    # =========================================================================

    async def _collect_agent_candidates(
        self,
        *,
        proc_json: Dict[str, Any],
        tenant_id: str,
        process_name: str,
    ) -> List[Dict[str, Any]]:
        """신규로 만들 가능성이 있는 에이전트 후보 목록을 반환 (INSERT 없음).

        각 후보:
          {
            "candidate_id":  "agent_<role>_<skill>",   # 식별자
            "role_name":     "심사역",
            "skill_name":    "검토 처리",
            "activity_ids":  [...],
            "existing_id":   "<user.id>" | None,       # 매칭된 기존 에이전트 (있으면 신규 생성 불필요)
            "profile":       {...} | None,             # 신규일 때 LLM 으로 미리 생성한 프로필
            "label":         "심사역 — 검토 처리",
            "description":   "name/role/goal/persona 요약",
          }
        """
        try:
            await self._load_org_and_agents(tenant_id)
        except Exception as e:
            logger.warning(f"[CAND][AGENT] org/agents 로드 실패: {e}")

        proc_skills = proc_json.get("skills") or []
        skill_name_by_id = {
            str(s.get("id") or "").strip(): str(s.get("name") or "").strip()
            for s in proc_skills
            if isinstance(s, dict)
        }
        activities = proc_json.get("activities") or []
        activity_by_id = {
            str(a.get("id") or "").strip(): a
            for a in activities
            if isinstance(a, dict) and str(a.get("id") or "").strip()
        }

        processor = ProcessPostProcessor(
            min_ratio=self._skill_extraction_min_ratio,
            min_count=self._skill_extraction_min_count,
            lane_skill_min_tasks=self._agent_creation_min_tasks_per_skill_per_lane,
            require_automation=self._agent_creation_require_automation,
        )
        raw_candidates = processor.collect_lane_skill_candidates(proc_json)
        out: List[Dict[str, Any]] = []
        for cand in raw_candidates:
            role_name = str(cand.get("role") or "").strip()
            skill_id = str(cand.get("skill_id") or "").strip()
            activity_ids = [str(x).strip() for x in (cand.get("activity_ids") or []) if str(x).strip()]
            if not role_name or not skill_id or not activity_ids:
                continue
            skill_name = skill_name_by_id.get(skill_id) or skill_id
            cand_id = f"agent_{self._normalize_text_key(role_name)}_{self._normalize_text_key(skill_name)}"
            existing = self._pick_existing_agent_for_lane_skill(
                role_name=role_name,
                skill_names=[skill_name],
            )
            existing_id = str(existing.get("id")) if existing and existing.get("id") else None

            profile: Optional[Dict[str, Any]] = None
            description = ""
            if not existing_id:
                # 신규 에이전트 후보 — LLM 으로 미리 프로필 생성해 미리보기 제공.
                team_id = self._org_teams_by_name.get(self._normalize_text_key(role_name)) or ""
                team_name = self._org_team_name_by_id.get(team_id) or role_name or "미분류"
                snippets: List[str] = []
                for aid in activity_ids[:6]:
                    a = activity_by_id.get(aid, {})
                    snippets.append(
                        " ".join(
                            str(x or "")
                            for x in (
                                a.get("name"),
                                a.get("instruction"),
                                a.get("description"),
                            )
                        )
                    )
                user_input = (
                    f"프로세스 '{process_name}'의 역할 '{role_name}'에 대해 다음 공통 스킬을 수행할 에이전트를 설계하세요.\n"
                    f"- 공통 스킬: {skill_name}\n"
                    f"- 대표 태스크 맥락: {' | '.join(snippets)}\n"
                    "불필요한 일반 업무는 제외하고 자동화 가능한 작업 중심으로 설계하세요."
                )
                mcp_tools = self._safe_json_loads(os.getenv("MCP_TOOLS_JSON", "")) or {}
                try:
                    profile = await self._llm_generate_agent_profile(
                        team_name=team_name,
                        user_input=user_input,
                        mcp_tools=mcp_tools,
                    )
                except Exception as e:
                    logger.warning(f"[CAND][AGENT] profile 생성 실패: role={role_name} err={e}")
                    profile = None
                if profile:
                    description = " · ".join(
                        str(x).strip() for x in (
                            profile.get("role"),
                            profile.get("goal"),
                            profile.get("persona"),
                        ) if str(x or "").strip()
                    )[:160]

            out.append({
                "candidate_id": cand_id,
                "role_name": role_name,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "activity_ids": activity_ids,
                "existing_id": existing_id,
                "profile": profile,
                "label": (
                    f"{role_name} — {skill_name}"
                    + (" (기존 에이전트 재사용)" if existing_id else " (신규)")
                ),
                "description": (
                    f"담당 활동 {len(activity_ids)}개"
                    + (f" · {description}" if description else "")
                    + (f" · 기존 에이전트 ID: {existing_id}" if existing_id else "")
                ),
            })
        return out

    async def _apply_approved_agents(
        self,
        *,
        proc_json: Dict[str, Any],
        tenant_id: str,
        candidates: List[Dict[str, Any]],
        approved_ids: Set[str],
    ) -> Dict[str, Set[str]]:
        """승인된 후보만 실제 처리:
          - existing_id 있는 후보 → 그 에이전트 그대로 할당
          - 신규 후보 (existing_id 없음) → profile 기반으로 _insert_agent_user 호출
          - 거부된 후보 → activity.agent=None 처리
        반환: {agent_id: {skill_name...}}
        """
        agent_skill_names: Dict[str, Set[str]] = {}
        if not isinstance(proc_json, dict):
            return agent_skill_names
        activities = proc_json.get("activities") or []
        roles = proc_json.get("roles") or []
        activity_by_id = {
            str(a.get("id") or "").strip(): a
            for a in activities
            if isinstance(a, dict) and str(a.get("id") or "").strip()
        }
        role_agent_by_name: Dict[str, str] = {}

        for cand in candidates:
            cand_id = str(cand.get("candidate_id") or "")
            if cand_id not in approved_ids:
                continue
            role_name = str(cand.get("role_name") or "").strip()
            skill_name = str(cand.get("skill_name") or "").strip()
            activity_ids = [str(x).strip() for x in (cand.get("activity_ids") or []) if str(x).strip()]
            existing_id = cand.get("existing_id")

            agent_id = existing_id
            if not agent_id:
                profile = cand.get("profile") or {}
                if profile:
                    try:
                        created = await self._insert_agent_user(
                            tenant_id=tenant_id,
                            agent_profile=profile,
                            agent_type="agent",
                        )
                        if created and created.get("id"):
                            agent_id = str(created.get("id"))
                            team_id = self._org_teams_by_name.get(self._normalize_text_key(role_name)) or ""
                            if team_id:
                                await self._update_org_chart_add_member(
                                    tenant_id=tenant_id,
                                    team_id=team_id,
                                    member_user=created,
                                )
                    except Exception as e:
                        logger.warning(f"[APPLY][AGENT] INSERT 실패: {cand_id} err={e}")
                        continue
            if not agent_id:
                continue

            role_agent_by_name[role_name] = agent_id
            agent_skill_names.setdefault(agent_id, set()).add(skill_name)

            for aid in activity_ids:
                a = activity_by_id.get(aid)
                if not isinstance(a, dict):
                    continue
                a["agent"] = agent_id
                a["agentMode"] = "complete"
                a["orchestration"] = "deepagents"

        # 후보에 없거나 거부된 activity 는 명시적으로 none 처리 (스킬 부착된 건 제외)
        for a in activities:
            if not isinstance(a, dict):
                continue
            if isinstance(a.get("skills"), list) and a.get("skills"):
                a["agentMode"] = "complete"
                a["orchestration"] = "deepagents"
                continue
            if str(a.get("agent") or "").strip():
                continue
            a["agent"] = None
            a["agentMode"] = "none"
            a["orchestration"] = None

        # roles.endpoint 갱신
        for r in roles:
            if not isinstance(r, dict):
                continue
            rname = str(r.get("name") or "").strip()
            if not rname:
                continue
            agent_id = role_agent_by_name.get(rname)
            if agent_id:
                r["endpoint"] = agent_id
                r["origin"] = "used"

        proc_json["activities"] = activities
        proc_json["roles"] = roles
        return agent_skill_names

    def _collect_dmn_candidates_from_proc_json(
        self,
        *,
        proc_json: Dict[str, Any],
        proc_def_id: str,
        process_name: str,
    ) -> List[Dict[str, Any]]:
        """실제 생성될 proc_json 의 게이트웨이 중 DMN 의사결정 테이블로 변환 가능한 것만 후보로 반환.

        후보 소스가 추출 단계 원시 엔티티(state.gateways)가 아니라, 검증/병합/축약을
        모두 거친 최종 proc_json.gateways 이므로:
          - "실제로 생성될 게이트웨이"만 노출된다 (유령/중복 게이트웨이 제거).
          - 분기 수/조건을 proc_json.sequences 에서 실측하므로 "분기 0개" 오류가 없다.

        자격 조건 (_augment_runtime_with_gateway_dmn 의 게이트와 동일):
          - ExclusiveGateway (parallel/inclusive 는 조건 평가 의미가 달라 제외)
          - outgoing sequence 2개 이상

        각 후보 (같은 이름의 게이트웨이는 하나로 병합 — HITL 옵션 중복 방지):
          {
            "candidate_id": "dmn::<proc_def_id>::<gid1>~<gid2>~...",
            "proc_def_id":  "...",
            "gateway_ids":  ["...", ...],
            "gateway_name": "...",
            "label":        "[프로세스명] 게이트웨이명",
            "description":  "분기 N개: 조건1 → 대상1 / 조건2 → 대상2",
          }
        """
        out: List[Dict[str, Any]] = []
        if not isinstance(proc_json, dict):
            return out
        gateways = proc_json.get("gateways") or []
        sequences = proc_json.get("sequences") or []
        if not isinstance(gateways, list) or not isinstance(sequences, list):
            return out

        # 분기 대상 노드 이름 lookup (description 표시용)
        node_name_by_id: Dict[str, str] = {}
        for coll in (
            proc_json.get("activities") or [],
            proc_json.get("events") or [],
            gateways,
        ):
            if isinstance(coll, list):
                for n in coll:
                    if isinstance(n, dict) and str(n.get("id") or "").strip():
                        node_name_by_id[str(n.get("id")).strip()] = str(n.get("name") or "").strip()

        outgoing_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for s in sequences:
            if isinstance(s, dict):
                src = str(s.get("source") or "").strip()
                if src:
                    outgoing_by_source.setdefault(src, []).append(s)

        # 1) DMN 자격 게이트웨이 수집 (exclusive + 분기 2개 이상)
        gw_entries: List[Dict[str, Any]] = []
        for gw in gateways:
            if not isinstance(gw, dict):
                continue
            gid = str(gw.get("id") or "").strip()
            gtype = str(gw.get("type") or "").lower().strip()
            if not gid:
                continue
            # parallel/inclusive 게이트웨이는 DMN 의사결정 테이블 대상이 아니다.
            if "exclusive" not in gtype:
                continue
            outs = outgoing_by_source.get(gid) or []
            if len(outs) < 2:
                # 분기 2개 미만 → 의사결정 테이블로 만들 의미가 없다.
                continue

            gname = str(gw.get("name") or "").strip() or gid
            branch_descs: List[str] = []
            for s in outs:
                cond = str(s.get("condition") or s.get("expression") or "").strip()
                tgt = str(s.get("target") or "").strip()
                tgt_name = node_name_by_id.get(tgt) or tgt
                if cond and tgt_name:
                    branch_descs.append(f"{cond} → {tgt_name}"[:80])
                elif cond:
                    branch_descs.append(cond[:80])
                elif tgt_name:
                    branch_descs.append(f"→ {tgt_name}"[:80])
            gw_entries.append({
                "gid": gid,
                "gname": gname,
                "branch_count": len(outs),
                "branch_descs": branch_descs,
            })

        # 2) 게이트웨이 이름 기준으로 중복 병합.
        #    추출 단계에서 같은 의사결정에 대해 게이트웨이가 여러 개 만들어지는 일이 잦다
        #    (예: "긴급휴가 여부 판단" 게이트웨이 2개). 같은 이름은 한 후보로 합쳐
        #    HITL 옵션 중복을 없애고, 선택 시 같은 이름의 모든 게이트웨이에 DMN 을 적용한다.
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for e in gw_entries:
            key = self._normalize_text_key(e["gname"]) or e["gid"]
            if key not in by_name:
                by_name[key] = []
                order.append(key)
            by_name[key].append(e)

        for key in order:
            group = by_name[key]
            # 대표(분기 수가 가장 많은 항목)로 라벨/설명을 구성
            primary = max(group, key=lambda x: x["branch_count"])
            gname = primary["gname"]
            # 같은 게이트웨이 id 가 중복 등장하는 경우까지 제거
            gids = list(dict.fromkeys(e["gid"] for e in group))
            desc = f"분기 {primary['branch_count']}개" + (
                ": " + " / ".join(primary["branch_descs"]) if primary["branch_descs"] else ""
            )
            if len(gids) > 1:
                desc = f"(동일 이름 게이트웨이 {len(gids)}개 통합) " + desc
            out.append({
                "candidate_id": f"dmn::{proc_def_id}::{'~'.join(gids)}",
                "proc_def_id": str(proc_def_id),
                "gateway_ids": gids,
                "gateway_name": gname,
                "label": f"[{process_name}] {gname}" if process_name else gname,
                "description": desc[:200],
            })
        return out

    def _remove_rejected_skills_from_proc_json(
        self,
        *,
        proc_json: Dict[str, Any],
        approved_skill_keys: Set[str],
    ) -> int:
        """사용자가 거부한 스킬을 proc_json 에서 제거.
        - proc_json['skills'] 에서 빠짐
        - activities[*].skills 에서 해당 id 제거
        반환: 제거된 카드 수.
        """
        if not isinstance(proc_json, dict):
            return 0
        removed = 0
        skills = proc_json.get("skills") or []
        if isinstance(skills, list):
            new_skills = []
            removed_ids: Set[str] = set()
            for s in skills:
                if not isinstance(s, dict):
                    continue
                key = self._normalize_skill_key(str(s.get("safe_name") or s.get("name") or s.get("id") or ""))
                if key in approved_skill_keys:
                    new_skills.append(s)
                else:
                    sid = str(s.get("id") or s.get("safe_name") or "").strip()
                    if sid:
                        removed_ids.add(sid)
                    removed += 1
            proc_json["skills"] = new_skills
            # activities 에서 거부된 id 제거
            for a in proc_json.get("activities") or []:
                if not isinstance(a, dict):
                    continue
                a_skills = a.get("skills")
                if isinstance(a_skills, list) and removed_ids:
                    a["skills"] = [x for x in a_skills if str(x or "").strip() not in removed_ids]
        return removed

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

    async def _validate_generated_process(
        self,
        *,
        proc_def_id: str,
        process_name: str,
        proc_json: Dict[str, Any],
        forms_result: Dict[str, Any],
        extracted: Dict[str, Any],
        tenant_id: str,
        event_queue: EventQueue,
        context_id: str,
        task_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """생성 완료된 프로세스를 실행 엔진으로 검증/자동개선한다.

        process-gpt-completion 의 실제 /initiate·/complete 로 start→end 를 실행시켜
        (다음 태스크 결정·분기 평가는 폴링 서비스가 수행) 결함을 찾고, 결함이 있으면
        LLM 으로 proc_json 을 교정 후 재저장한다. 다음 활동 조회/검증 인스턴스 정리는
        Supabase 로 DB 를 직접 read/delete 한다.
        실패해도 예외를 던지지 않고 리포트(dict)를 반환한다 — todo 는 정상 완료시킨다.
        """
        if not getattr(self, "validation_enabled", False):
            return {"proc_def_id": proc_def_id, "skipped": True,
                    "skip_reason": "검증 비활성화(PDF2BPMN_VALIDATION_ENABLED=false)",
                    "passed": None, "process_name": process_name}

        if not self.supabase_client:
            return {"proc_def_id": proc_def_id, "skipped": True,
                    "skip_reason": "Supabase 클라이언트 없음 — 검증 불가",
                    "passed": None, "process_name": process_name}

        try:
            from src.pdf2bpmn.validation import ProcessValidator
        except Exception as e:
            logger.warning(f"[VALIDATION] 모듈 import 실패: {e}")
            return {"proc_def_id": proc_def_id, "skipped": True,
                    "skip_reason": f"검증 모듈 import 실패: {e}",
                    "passed": None, "process_name": process_name}

        forms = (forms_result or {}).get("forms") or {}

        async def _progress(message: str, pct: int, extra: Dict[str, Any] = None):
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                message, "tool_usage_started", pct, extra or {},
            )

        async def _save_definition(pdid: str, definition: Dict[str, Any]) -> bool:
            return await self._update_proc_def_definition_only(
                proc_def_id=pdid, tenant_id=tenant_id, definition=definition,
            )

        async def _llm(messages, max_tokens):
            return await self._call_openai_json_messages(
                messages=messages, max_tokens=max_tokens,
                model=self.process_definition_model, temperature=0.0,
            )

        # ③ 다음 활동 조회: bpm_proc_inst 를 DB 에서 직접 읽는다.
        #   제출하면 폴링 서비스가 다음 태스크를 찾아 current_activity_ids 에 반영하므로,
        #   검증기는 그 결과만 읽으면 된다(다음-태스크 탐색 로직을 재구현하지 않는다).
        def _fetch_instance_state_sync(proc_inst_id: str) -> Dict[str, Any]:
            rows = (
                self.supabase_client.table("bpm_proc_inst")
                .select("proc_inst_id,status,current_activity_ids")
                .or_(f"proc_inst_id.eq.{proc_inst_id},root_proc_inst_id.eq.{proc_inst_id}")
                .eq("tenant_id", tenant_id)
                .execute()
                .data
            ) or []
            status = "RUNNING"
            active: List[str] = []
            for row in rows:
                cids = row.get("current_activity_ids") or []
                if isinstance(cids, str):
                    cids = [cids]
                if row.get("proc_inst_id") == proc_inst_id:
                    status = row.get("status") or "RUNNING"
                    active.extend(str(c) for c in cids if c)
                elif str(row.get("status") or "").upper() == "RUNNING":
                    # 서브프로세스 자식 인스턴스의 활성 활동도 합친다.
                    active.extend(str(c) for c in cids if c)
            return {"status": status, "current_activity_ids": list(dict.fromkeys(active))}

        async def _fetch_instance_state(proc_inst_id: str) -> Dict[str, Any]:
            return await asyncio.to_thread(_fetch_instance_state_sync, proc_inst_id)

        # ④ 검증용 인스턴스 정리: 테스트 인스턴스/워크아이템 row 를 DB 에서 직접 삭제.
        def _cleanup_instance_sync(proc_inst_id: str) -> None:
            for table in ("todolist", "bpm_proc_inst"):
                try:
                    (
                        self.supabase_client.table(table)
                        .delete()
                        .or_(f"proc_inst_id.eq.{proc_inst_id},root_proc_inst_id.eq.{proc_inst_id}")
                        .eq("tenant_id", tenant_id)
                        .execute()
                    )
                except Exception as ce:
                    logger.debug(f"[VALIDATION] cleanup({table}) 실패(무시): {ce}")

        async def _cleanup_instance(proc_inst_id: str) -> None:
            if not getattr(self, "validation_cleanup", True):
                return
            await asyncio.to_thread(_cleanup_instance_sync, proc_inst_id)

        # 검증 실행 시 /initiate·/complete 에 넘길 행위자 이메일.
        # 실제 사용자가 있으면 그 사용자로(휴먼 우선), 없으면 None → 검증기 기본값 사용.
        # (엔진은 미등록 이메일도 graceful 처리하므로 흐름 검증엔 어떤 값이든 무방.)
        actor_email = None
        try:
            for _u in (self._users or []):
                if isinstance(_u, dict) and _u.get("email") and not _u.get("is_agent"):
                    actor_email = _u.get("email")
                    break
            if not actor_email:
                for _u in (self._users or []):
                    if isinstance(_u, dict) and _u.get("email"):
                        actor_email = _u.get("email")
                        break
        except Exception:
            actor_email = None

        # 검증 흐름 상세 리포트(.md) 파일 경로 — output/validation/ 아래.
        report_path = None
        try:
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _safe_pdid = "".join(
                c if (c.isalnum() or c in "-_") else "_" for c in str(proc_def_id)
            )[:80]
            report_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "output", "validation",
                f"validation_{_safe_pdid}_{_ts}.md",
            )
        except Exception:
            report_path = None

        validator = ProcessValidator(
            llm_call=_llm,
            save_definition=_save_definition,
            engine_base_url=self.completion_engine_url,
            tenant_id=tenant_id,
            fetch_instance_state=_fetch_instance_state,
            cleanup_instance=_cleanup_instance,
            max_iters=self.validation_max_iters,
            advance_timeout=self.validation_advance_timeout,
            actor_email=actor_email,
            report_path=report_path,
            logger=logger,
            progress=_progress,
        )
        await self._send_progress_event(
            event_queue, context_id, task_id, job_id,
            f"[VALIDATION] 생성된 프로세스 실행 검증 시작: {process_name}",
            "tool_usage_started", 83,
            {"proc_def_id": proc_def_id, "process_name": process_name},
        )
        report = await validator.validate_and_repair(
            proc_def_id=proc_def_id,
            process_name=process_name,
            proc_json=proc_json,
            forms=forms,
            extracted=extracted,
        )
        if isinstance(report, dict) and report_path:
            report["report_path"] = report_path  # 상세 리포트 파일 위치
        # 결과 요약 이벤트
        if report.get("skipped"):
            msg = f"[VALIDATION] 검증 건너뜀: {process_name} — {report.get('skip_reason')}"
        elif report.get("passed"):
            msg = (f"[VALIDATION] 검증 통과: {process_name} "
                   f"({report.get('iterations')}회차, start→end 정상 실행)")
        else:
            msg = (f"[VALIDATION] 검증 미통과: {process_name} — "
                   f"잔여 결함 {len(report.get('remaining_defects') or [])}건, "
                   f"{report.get('iterations')}회 개선 시도")
        await self._send_progress_event(
            event_queue, context_id, task_id, job_id,
            msg, "tool_usage_finished", 89,
            {"proc_def_id": proc_def_id,
             "passed": report.get("passed"),
             "skipped": report.get("skipped"),
             "iterations": report.get("iterations"),
             "repaired": report.get("repaired")},
        )
        logger.info(
            f"[VALIDATION] {proc_def_id}: passed={report.get('passed')} "
            f"skipped={report.get('skipped')} iters={report.get('iterations')} "
            f"repaired={report.get('repaired')}"
        )
        return report

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

    def _enrich_process_definition(
        self,
        proc_json: Dict[str, Any],
        *,
        process_name: str,
        process_definition_id: str,
    ) -> Dict[str, Any]:
        """
        proc_def.definition(JSON)이 "바로 실행 가능한 수준"에 가깝도록 최소 필드를 보정합니다.

        원칙:
        - 어떤 문서/형식이 와도 비어있지 않게(roles/activities/sequences 최소 1개) 보정
        - 추출/변환 결과를 최대한 존중하되, 필수 필드가 비면 안전한 기본값을 채움
        """
        # STRICT MODE:
        # - 문서에 없는 비즈니스 내용을 생성하지 않습니다.
        # - roles/tasks/events/sequences/data를 새로 "추가 생성"하지 않습니다.
        # - 단, 시스템 실행을 위한 기술적 필드(tool 등)는 비어있으면 기본값을 채울 수 있습니다.
        strict = os.getenv("STRICT_DEFINITION_MODE", "true").lower() == "true"

        result = proc_json or {}
        result["processDefinitionName"] = process_name or result.get("processDefinitionName") or "프로세스"
        result["processDefinitionId"] = process_definition_id or result.get("processDefinitionId") or ""

        # Ensure container keys exist
        for k in ("data", "roles", "events", "activities", "gateways", "sequences", "subProcesses", "participants"):
            if k not in result or result[k] is None:
                result[k] = []

        roles: List[Dict[str, Any]] = result.get("roles", []) if isinstance(result.get("roles"), list) else []
        activities: List[Dict[str, Any]] = result.get("activities", []) if isinstance(result.get("activities"), list) else []
        events: List[Dict[str, Any]] = result.get("events", []) if isinstance(result.get("events"), list) else []
        sequences: List[Dict[str, Any]] = result.get("sequences", []) if isinstance(result.get("sequences"), list) else []

        # STRICT: roles/participants 신규 생성 금지
        if strict:
            pass
        else:
            # Build role pool if missing (legacy behavior)
            if not roles:
                role_names = []
                for a in activities:
                    rn = (a.get("role") or "").strip()
                    if rn and rn not in role_names:
                        role_names.append(rn)
                if not role_names:
                    role_names = ["사용자"]
                roles = [{"name": rn, "endpoint": "", "resolutionRule": None, "default": ""} for rn in role_names]
                result["roles"] = roles

            # Ensure participants exist (Pool)
            participants: List[Dict[str, Any]] = result.get("participants", []) if isinstance(result.get("participants"), list) else []
            if not participants:
                participants = [{"id": f"Participant_{process_definition_id}", "name": result["processDefinitionName"], "processRef": result["processDefinitionId"]}]
                result["participants"] = participants

        primary_role = (roles[0].get("name") if roles else "") or ""

        # Build role lookup table
        role_by_name = {str(r.get("name", "")).strip(): r for r in roles if isinstance(r, dict) and r.get("name")}

        # Ensure each activity has required-ish fields
        for idx, a in enumerate(activities):
            if not isinstance(a, dict):
                continue
            a.setdefault("id", f"Activity_{idx+1}")
            a.setdefault("name", f"활동 {idx+1}")
            a.setdefault("type", a.get("type") or "userTask")
            # STRICT: role이 없으면 채우지 않음(문서 근거 없는 역할 생성 금지)
            if (not strict) and (not (a.get("role") or "").strip()) and primary_role:
                a["role"] = primary_role
            a.setdefault("description", "")
            a.setdefault("instruction", a.get("instruction") or "")
            if isinstance(a.get("instruction"), str):
                # escaped newline("\\n")이 한 줄 텍스트로 남지 않도록 실제 개행으로 복원
                a["instruction"] = (
                    a.get("instruction", "")
                    .replace("\r\n", "\n")
                    .replace("\\r\\n", "\n")
                    .replace("\\n", "\n")
                    .replace("\\r", "\n")
                )
            a.setdefault("duration", a.get("duration") or 5)

            # tool(form) - 없으면 안정적으로 생성
            if not (a.get("tool") or "").strip():
                safe_pid = re.sub(r"[^a-z0-9_]+", "_", (process_definition_id or "process").lower()).strip("_")
                safe_aid = re.sub(r"[^a-z0-9_]+", "_", str(a.get("id", f"activity_{idx+1}")).lower()).strip("_")
                a["tool"] = f"formHandler:{safe_pid}_{safe_aid}_form"

            # input/output data
            if not isinstance(a.get("inputData"), list):
                a["inputData"] = []
            if not isinstance(a.get("outputData"), list):
                a["outputData"] = []
            # STRICT: outputData 신규 생성 금지 (문서에 없는 데이터 변수 생성 금지)
            if not isinstance(a.get("checkpoints"), list):
                a["checkpoints"] = []

            # Agent execution fields (optional but makes process runnable)
            a.setdefault("agent", None)
            a.setdefault("agentMode", "none")
            a.setdefault("orchestration", None)
            a.setdefault("attachments", [])
            if not isinstance(a.get("skills"), list):
                a["skills"] = []
            a.setdefault("customProperties", [])

        # STRICT: 이벤트 신규 생성 금지
        if not strict:
            def _has_event_type(type_name: str) -> bool:
                for e in events:
                    if isinstance(e, dict) and (e.get("type") == type_name):
                        return True
                return False

            if not _has_event_type("startEvent"):
                events.insert(0, {"id": "Event_Start", "name": "시작", "type": "startEvent", "role": primary_role, "process": process_definition_id})
            if not _has_event_type("endEvent"):
                events.append({"id": "Event_End", "name": "종료", "type": "endEvent", "role": primary_role, "process": process_definition_id})
            result["events"] = events

        # STRICT: 시퀀스/데이터 신규 생성 금지.
        # 다만 XML→JSON 변환 결과가 condition을 name에 넣었을 경우, condition 복원은 "내용 생성"이 아니라 필드 정규화로 간주.
        for s in sequences:
            if not isinstance(s, dict):
                continue
            if (not s.get("condition")) and s.get("name"):
                s["condition"] = s.get("name")
        result["sequences"] = sequences

        return result

    def _ensure_end_event_connectivity(
        self,
        proc_json: Dict[str, Any],
        *,
        process_name: str = "",
    ) -> Dict[str, Any]:
        """
        Ensure at least one sequence reaches each endEvent.
        If missing, connect a terminal node (prefer last activity) -> endEvent.
        Also mirrors the added edge into `elements` list when present.
        """
        result = dict(proc_json or {})
        activities = result.get("activities") or []
        gateways = result.get("gateways") or []
        events = result.get("events") or []
        sequences = result.get("sequences") or []
        if not isinstance(activities, list):
            activities = []
        if not isinstance(gateways, list):
            gateways = []
        if not isinstance(events, list):
            events = []
        if not isinstance(sequences, list):
            sequences = []

        end_ids: List[str] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            et = str(e.get("type") or "").strip().lower()
            if et == "endevent":
                eid = str(e.get("id") or "").strip()
                if eid:
                    end_ids.append(eid)
        if not end_ids:
            result["sequences"] = sequences
            return result

        node_ids: List[str] = []
        for a in activities:
            if isinstance(a, dict):
                aid = str(a.get("id") or "").strip()
                if aid:
                    node_ids.append(aid)
        for g in gateways:
            if isinstance(g, dict):
                gid = str(g.get("id") or "").strip()
                if gid:
                    node_ids.append(gid)
        for e in events:
            if isinstance(e, dict):
                eid = str(e.get("id") or "").strip()
                if eid:
                    node_ids.append(eid)

        outgoing: Set[str] = set()
        incoming: Dict[str, int] = {}
        pair_set: Set[Tuple[str, str]] = set()
        for s in sequences:
            if not isinstance(s, dict):
                continue
            src = str(s.get("source") or "").strip()
            tgt = str(s.get("target") or "").strip()
            if src:
                outgoing.add(src)
            if tgt:
                incoming[tgt] = incoming.get(tgt, 0) + 1
            if src and tgt:
                pair_set.add((src, tgt))

        terminal_ids: List[str] = [nid for nid in node_ids if nid not in outgoing and nid not in end_ids]
        # Prefer last activity id that is terminal.
        last_activity_id = ""
        for a in reversed(activities):
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if aid:
                last_activity_id = aid
                if aid in terminal_ids:
                    break

        added = 0
        for end_id in end_ids:
            if incoming.get(end_id, 0) > 0:
                continue

            source_id = ""
            if last_activity_id and last_activity_id not in end_ids:
                source_id = last_activity_id
            elif terminal_ids:
                source_id = terminal_ids[-1]
            else:
                candidates = [nid for nid in node_ids if nid not in end_ids]
                source_id = candidates[-1] if candidates else ""

            if not source_id or source_id == end_id:
                continue
            if (source_id, end_id) in pair_set:
                continue

            seq_id = f"seq_{self._snake_id(source_id)}_{self._snake_id(end_id)}"
            sequences.append(
                {
                    "id": seq_id,
                    "name": "",
                    "source": source_id,
                    "target": end_id,
                    "condition": "",
                    "properties": "{}",
                }
            )
            pair_set.add((source_id, end_id))
            incoming[end_id] = incoming.get(end_id, 0) + 1
            added += 1

            # If elements list exists, mirror sequence there too.
            elems = result.get("elements")
            if isinstance(elems, list):
                e_pair_set: Set[Tuple[str, str]] = set()
                for el in elems:
                    if not isinstance(el, dict):
                        continue
                    if str(el.get("elementType") or "").strip().lower() != "sequence":
                        continue
                    es = str(el.get("source") or "").strip()
                    et = str(el.get("target") or "").strip()
                    if es and et:
                        e_pair_set.add((es, et))
                if (source_id, end_id) not in e_pair_set:
                    elems.append(
                        {
                            "id": seq_id,
                            "name": "",
                            "source": source_id,
                            "target": end_id,
                            "elementType": "Sequence",
                        }
                    )
                    result["elements"] = elems

        result["sequences"] = sequences
        if added > 0:
            logger.info(
                f"[PROCDEF][CONNECTIVITY] appended terminal->end sequences: added={added} "
                f"(process={process_name!r})"
            )
        return result

    def _convert_xml_to_json(self, bpmn_xml: str) -> Dict[str, Any]:
        """
        BPMN XML을 ProcessGPT JSON 형식으로 변환
        ProcessDefinitionModule.vue의 convertXMLToJSON과 유사한 로직
        """
        try:
            # Prefer robust converter (ported from old_pdf2bpmn) if available.
            # 로컬/컨테이너 어디서 실행되든 `src/`를 sys.path에 추가해 import 가능하게 합니다.
            try:
                repo_root = Path(__file__).resolve().parent
                src_dir = str(repo_root / "src")
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from pdf2bpmn.bpmn_to_json import BPMNToJSONConverter  # type: ignore

                converter = BPMNToJSONConverter()
                # 아래 2개 값은 호출자가 바깥에서 세팅하므로 여기서는 더미로 채움
                return converter.convert(bpmn_xml, process_definition_id="", process_name="")
            except Exception as e:
                logger.warning(f"[WARN] 고급 BPMN→JSON 변환기 로드 실패. 단순 변환으로 fallback 합니다. err={e}")

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
                "processDefinitionId": str(uuid.uuid4()),
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
            bpmn_val = proc_def.get("bpmn")
            bpmn_len = len(bpmn_val) if isinstance(bpmn_val, str) else 0
            logger.info(f"[DB-PROC_DEF] name={proc_def.get('name')}, bpmn_length={bpmn_len}")
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

    def _parse_todolist_output_field(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    async def _load_fresh_todolist_output(self, task_id: str) -> Dict[str, Any]:
        if not self.supabase_client or not task_id:
            return {}
        from src.pdf2bpmn.hitl import _read_todolist_output  # type: ignore

        return await asyncio.to_thread(
            _read_todolist_output, self.supabase_client, str(task_id)
        )

    def _get_hitl_resume_checkpoint_if_fb_requested(
        self, row: Dict[str, Any], output: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        out = output if isinstance(output, dict) else self._parse_todolist_output_field(
            row.get("output")
        )
        cp = out.get("hitl_checkpoint")
        if not isinstance(cp, dict) or cp.get("stage") != "unified_post_procgen":
            return None
        task_type = str(row.get("task_type") or "").upper()
        phase = str(out.get("pdf2bpmn_phase") or "").lower()
        todo_id = str(row.get("id") or "")

        def _log_resume(reason: str) -> Dict[str, Any]:
            logger.info(
                "[HITL][RESUME] %s — todo_id=%s process_index=%s task_type=%s phase=%s",
                reason,
                todo_id,
                cp.get("process_index"),
                task_type,
                phase,
            )
            return cp

        if task_type == "FB_REQUESTED":
            return _log_resume("FB_REQUESTED 재진입")
        if phase == "post_hitl_generate":
            return _log_resume("post_hitl_generate phase")
        if out.get("hitl_paused") or phase == "awaiting_hitl":
            # 검사 대상 = '실제로 사용자에게 물어본' 질문 id 목록.
            #   hitl_pending_question_ids 는 pause_for_hitl 이 qids_to_wait(=실제 생성된
            #   질문) 만 기록한다. 반면 checkpoint.question_ids 의 'dmn' 키는 DMN 후보가
            #   0개여서 질문이 생성되지 않았어도 항상 채워지므로, 그걸로 검사하면
            #   '묻지도 않은 DMN 답변' 을 기다리다 재개에 실패 → 처음부터 무한 재실행한다.
            qids: List[str] = []
            pending = out.get("hitl_pending_question_ids")
            if isinstance(pending, list):
                qids = [str(q) for q in pending if str(q or "").strip()]
            if not qids:
                # fallback: hitl_pending_question_ids 가 없을 때만 checkpoint 에서 유도.
                # question_ids 값은 문자열(dmn) 또는 {proc_def_id: qid} dict(skills/agents).
                qids_map = cp.get("question_ids") if isinstance(cp.get("question_ids"), dict) else {}
                for v in qids_map.values():
                    if isinstance(v, dict):
                        qids.extend(str(x) for x in v.values() if str(x or "").strip())
                    elif str(v or "").strip():
                        qids.append(str(v))
            if qids and self.supabase_client and todo_id:
                entries = read_batch_responses(
                    self.supabase_client,
                    todo_id,
                    qids,
                    cp.get("wait_started_at"),
                )
                answered = [q for q in qids if entries.get(q)]
                # 물어본 질문 개수만큼 답변이 모두 채워졌으면 재개 (후보 개수와 무관).
                if len(answered) == len(qids):
                    return _log_resume(
                        f"HITL 응답 완료(폴백 재진입, {len(answered)}/{len(qids)} 질문)"
                    )
                logger.info(
                    "[HITL][RESUME] 대기 — todo_id=%s 답변 %d/%d (미응답=%s)",
                    todo_id, len(answered), len(qids),
                    [q for q in qids if not entries.get(q)],
                )
        return None

    async def _apply_unified_hitl_to_process(
        self,
        *,
        proc_json: Dict[str, Any],
        process_name: str,
        process_skill_metas: List[Dict[str, Any]],
        agent_candidates_for_process: List[Dict[str, Any]],
        approved_skill_keys: Optional[Set[str]],
        approved_agent_ids: Optional[Set[str]],
        state: Dict[str, Any],
        tenant_id: str,
        agent_user_ids_for_skill_sync: Set[str],
        agent_skill_names_for_sync: Dict[str, Set[str]],
    ) -> List[Dict[str, Any]]:
        """HITL 응답을 proc_json / 후보 목록에 반영."""
        if approved_skill_keys is not None:
            removed = self._remove_rejected_skills_from_proc_json(
                proc_json=proc_json,
                approved_skill_keys=approved_skill_keys,
            )
            if removed:
                logger.info(f"[HITL][SKILL] {removed}개 스킬 거부 → proc_json 정리")
            state.setdefault("__hitl_approved_skill_keys", set()).update(approved_skill_keys)

        if agent_candidates_for_process and approved_agent_ids is not None:
            try:
                applied = await self._apply_approved_agents(
                    proc_json=proc_json,
                    tenant_id=tenant_id,
                    candidates=agent_candidates_for_process,
                    approved_ids=approved_agent_ids,
                )
                for aid, sk_set in (applied or {}).items():
                    agent_user_ids_for_skill_sync.add(aid)
                    agent_skill_names_for_sync.setdefault(aid, set()).update(sk_set)
            except Exception as exc:
                logger.warning(f"[HITL][AGENT] apply 실패: {exc}")

        if process_skill_metas and approved_skill_keys is not None:
            process_skill_metas = [
                m for m in process_skill_metas
                if isinstance(m, dict)
                and self._normalize_skill_key(str(m.get("safe_name") or m.get("name") or ""))
                in approved_skill_keys
            ]
        return process_skill_metas

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
        
        temp_paths_to_cleanup: Set[str] = set()
        # finally 에서 AGE 그래프를 정리하기 위한 핸들. try 진입 전에 초기화하여
        # 부분 실패 상황에서도 NameError 없이 안전하게 drop 시도할 수 있도록 한다.
        workflow: Optional[Any] = None  # type: ignore[name-defined]
        age_graph_name_for_cleanup: str = ""
        hitl_paused_only: bool = False
        
        try:
            # 2. 작업 시작 이벤트
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[START] PDF2BPMN 변환 작업을 시작합니다...",
                "task_started", 0
            )
            
            # 3. Query 파싱 (PDF 정보 추출)
            parsed = self._parse_query(user_input or "")
            input_files = parsed.get("input_files") or []
            # 컨설팅 기반 생성 모드: 파일 대신 컨설팅 내용으로 프로세스를 생성한다.
            consulting_payload = (
                parsed.get("consulting_payload")
                if isinstance(parsed.get("consulting_payload"), dict)
                else None
            )
            parsed_room_id = str(parsed.get("room_id") or "").strip()
            parsed_tenant_id = str(parsed.get("tenant_id") or "").strip() or str(tenant_id or "")
            effective_tenant_id = parsed_tenant_id or str(tenant_id or "")
            from src.pdf2bpmn.graph.neo4j_client import Neo4jClient  # type: ignore
            age_graph_name = Neo4jClient.build_graph_name(
                tenant_id=effective_tenant_id,
                todo_id=str(task_id or ""),
            )
            logger.info(
                f"[GRAPH] AGE graph scope selected: tenant='{effective_tenant_id}', "
                f"todo='{task_id}', graph='{age_graph_name}'"
            )
            age_graph_name_for_cleanup = str(age_graph_name or "")
            if not input_files:
                pdf_url_fallback = (parsed.get("pdf_url") or "").strip()
                if pdf_url_fallback:
                    input_files = [{"url": pdf_url_fallback, "name": parsed.get("pdf_name") or ""}]

            pdf_name = (
                (input_files[0].get("name") or "").strip()
                if input_files
                else parsed.get("pdf_name", "document.pdf")
            )
            logger.info(f"[INFO] Input file count: {len(input_files)}")
            
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
            
            fresh_output = await self._load_fresh_todolist_output(str(task_id))
            resume_cp = self._get_hitl_resume_checkpoint_if_fb_requested(row, fresh_output)
            main_loop = asyncio.get_running_loop()
            request_graph_run_id = ""
            input_file_names: List[str] = []
            extracted_by_proc_id: Dict[str, Dict[str, Any]] = {}
            state: Dict[str, Any] = {}

            if resume_cp:
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[RESUME] HITL 응답을 반영해 프로세스 생성을 이어갑니다...",
                    "tool_usage_started", 73,
                )
                effective_tenant_id = str(resume_cp.get("effective_tenant_id") or effective_tenant_id)
                age_graph_name = str(resume_cp.get("age_graph_name") or age_graph_name)
                age_graph_name_for_cleanup = age_graph_name
                user_input = str(resume_cp.get("user_input") or user_input or "")
                pdf_name = str(resume_cp.get("pdf_name") or pdf_name)
                input_file_names = list(resume_cp.get("input_file_names") or [])
                request_graph_run_id = str(resume_cp.get("request_graph_run_id") or f"{task_id}-resume")
                ws = resume_cp.get("workflow_state") if isinstance(resume_cp.get("workflow_state"), dict) else {}
                state = {
                    "dmn_decisions": list(ws.get("dmn_decisions") or []),
                    "dmn_rules": list(ws.get("dmn_rules") or []),
                    "skill_docs": dict(ws.get("skill_docs") or {}),
                    "__hitl_dmn_decided": ws.get("__hitl_dmn_decided"),
                    "__hitl_approved_dmn_gateways": set(ws.get("__hitl_approved_dmn_gateways") or []),
                    "__hitl_approved_skill_keys": set(ws.get("__hitl_approved_skill_keys") or []),
                    "processes": [],
                }
                # resume: checkpoint 의 prepared_processes(proc_json/extracted_payload 포함)를
                # 그대로 복원하므로 Neo4j 재조회 불필요. extracted_by_proc_id 를 비워두면
                # PASS1 for 루프가 0회 실행되어 자동으로 건너뛴다.
                extracted_by_proc_id = {}

            # 컨설팅 모드일 때 PASS1 생성 LLM 에 넘길 컨설팅 초안(파일 모드는 빈 값)
            consulting_outline_for_gen = ""

            if (not resume_cp) and consulting_payload:
                # =================================================================
                # 4'. 컨설팅 기반 생성 모드
                #    - 업로드 문서가 없으므로 메멘토 청크/섹션 분리/Neo4j 그래프 추출을
                #      모두 건너뛴다.
                #    - 컨설팅 내용(=consulting_outline 등) 자체를 LLM 입력으로 그대로 사용하고,
                #      JSON 생성 규칙은 파일 모드와 동일한 것을 그대로 적용한다.
                #      → 별도의 "consulting → extracted" 중간 변환 단계를 두지 않는다.
                #    - PASS1(JSON 생성) → 스킬/에이전트/DMN HITL → 저장/검증 흐름은
                #      파일 모드와 100% 동일하게 재사용된다.
                # =================================================================
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[CONSULTING] 컨설팅 내용으로 프로세스 정의를 생성합니다...",
                    "tool_usage_started", 20,
                )
                request_graph_run_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
                input_file_names = []
                pdf_name = "컨설팅 기반 프로세스"
                # 컨설팅 입력 텍스트(요청 + 초안 + 사용자 답변 + 이미지 분석)를 하나로 합쳐
                # 그대로 LLM 의 user 메시지로 사용한다. (별도 helper 없이 인라인으로 합친다.)
                _cp = consulting_payload or {}
                _consulting_parts: List[str] = []
                if str(_cp.get("user_request") or "").strip():
                    _consulting_parts.append(f"[사용자 요청]\n{_cp['user_request']}")
                if str(_cp.get("consulting_outline") or "").strip():
                    _consulting_parts.append(f"[컨설팅 초안]\n{_cp['consulting_outline']}")
                if str(_cp.get("user_answer") or "").strip():
                    _consulting_parts.append(f"[사용자 답변]\n{_cp['user_answer']}")
                if str(_cp.get("image_analysis") or "").strip():
                    _consulting_parts.append(f"[이미지 분석]\n{_cp['image_analysis']}")
                consulting_outline_for_gen = "\n\n".join(_consulting_parts)
                # PASS1 루프가 한 번 도는 단일 프로세스 단위만 만든다 — extracted 는 비워둔다.
                extracted_by_proc_id = {
                    f"consulting_{uuid.uuid4().hex[:12]}": {
                        "detail": {},
                        "graph_elements": {},
                        "process_name": "",
                    }
                }
                # PASS1/PASS2/HITL 에서 참조하는 최소 state 초기화 (파일 모드 resume 경로와 동일 키)
                state = {
                    "dmn_decisions": [],
                    "dmn_rules": [],
                    "skill_docs": {},
                    "processes": [],
                }
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[CONSULTING] 컨설팅 내용 준비 완료",
                    "tool_usage_finished", 40,
                )

            if (not resume_cp) and (not consulting_payload):

                # =================================================================
                # 4. 메멘토(process-gpt-memento)에서 사전 처리된 청크/임베딩 로드
                #    - 메인 채팅 → 메멘토 → 메인 에이전트 → pdf2bpmn 흐름이 고정이므로
                #      pdf2bpmn 시점에는 메멘토가 이미 다음을 끝낸 상태이다:
                #        · Storage 업로드 (PDF 변환 포함)
                #        · 페이지/문서 텍스트 추출 + chunking
                #        · 임베딩(Chroma + Supabase documents)
                #    - 따라서 pdf2bpmn은 더 이상 다운로드/변환/Synap/HWP 분기 처리를 하지 않고,
                #      파일 형식과 무관하게 동일하게 메멘토의 청크를 받아 섹션 분할/노드 추출에
                #      이어 사용한다.
                # =================================================================
                if not input_files:
                    raise Exception("파일 URL이 제공되지 않았습니다. query의 [InputData]에 file/files를 포함해주세요.")

                pdf_paths_for_workflow: List[str] = []  # 더 이상 사용하지 않음 (호환용 placeholder)
                input_file_names: List[str] = []
                fetch_errors: List[str] = []
                preloaded_documents: List[PdfDocument] = []
                preloaded_sections: List[Section] = []
                preloaded_chunks: List[ReferenceChunk] = []
                total_reused_embeddings = 0

                # 사용자 [도구 설정] pdf2bpmnLevel 을 SOP 분할 단계에서 사용하기 위해 미리 추출.
                # 이후 workflow.set_dedup_level 호출은 dedup 임계 적용용으로 동일 값을 다시 사용한다.
                try:
                    _ts_for_section = parsed.get("tool_settings") or {}
                    _seg_level = (_ts_for_section.get("pdf2bpmnLevel") or "standard").strip().lower()
                    if _seg_level not in {"concise", "standard", "detailed"}:
                        _seg_level = "standard"
                except Exception:
                    _seg_level = "standard"
                logger.info(f"[SECTION-LEVEL] pdf2bpmnLevel='{_seg_level}' applied to SOP segmentation")

                for idx, file_info in enumerate(input_files, start=1):
                    file_url = (file_info.get("url") or "").strip()
                    display_name = (file_info.get("name") or f"document_{idx}").strip() or f"document_{idx}"
                    file_path = (file_info.get("path") or "").strip().rstrip("?")
                    file_room_id = (file_info.get("room_id") or parsed_room_id or "").strip()
                    file_tenant_id = (file_info.get("tenant_id") or effective_tenant_id or "").strip()

                    if not file_url and not file_path and not display_name:
                        continue

                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[MEMENTO] 청크/임베딩 조회 중 ({idx}/{len(input_files)}): {display_name}",
                        "tool_usage_started", 8
                    )

                    try:
                        memento_chunks = await self._fetch_memento_chunks(
                            tenant_id=file_tenant_id,
                            file_path=file_path,
                            file_name=display_name,
                            room_id=file_room_id,
                            include_embeddings=True,
                        )
                        if not memento_chunks:
                            raise Exception(
                                f"메멘토에 사전 처리된 청크가 없습니다 (tenant={file_tenant_id}, "
                                f"file_path={file_path or 'N/A'}, file_name={display_name})"
                            )

                        docs, secs, chs = self._build_state_from_memento_chunks(
                            display_name=display_name,
                            source=f"memento://{file_path or display_name}",
                            memento_chunks=memento_chunks,
                            pdf2bpmn_level=_seg_level,
                        )
                        if not docs or not chs:
                            raise Exception(
                                f"메멘토 청크로부터 문서 상태를 구성하지 못했습니다: {display_name}"
                            )

                        reused = sum(1 for c in chs if isinstance(c.embedding, list) and len(c.embedding) > 0)
                        total_reused_embeddings += reused

                        preloaded_documents.extend(docs)
                        preloaded_sections.extend(secs)
                        preloaded_chunks.extend(chs)
                        input_file_names.append(display_name)

                        await self._send_progress_event(
                            event_queue, context_id, task_id, job_id,
                            f"[MEMENTO] 재사용 완료 ({idx}/{len(input_files)}): {display_name} "
                            f"(chunks={len(chs)}, embeddings={reused}/{len(chs)})",
                            "tool_usage_started", 10
                        )
                    except Exception as e:
                        fetch_errors.append(f"{display_name}: {e}")
                        logger.warning(
                            f"[WARN] 메멘토 청크 로드 실패({idx}/{len(input_files)}): {display_name} - {e}"
                        )

                if not preloaded_chunks:
                    raise Exception(
                        "메멘토에서 사전 처리된 청크를 가져오지 못했습니다: "
                        + ("; ".join(fetch_errors) if fetch_errors else "유효한 파일이 없습니다.")
                    )

                pdf_name = input_file_names[0] if input_file_names else (pdf_name or "document.pdf")
                logger.info(
                    "[MEMENTO] preload summary: files=%d sections=%d chunks=%d reused_embeddings=%d",
                    len(preloaded_documents), len(preloaded_sections),
                    len(preloaded_chunks), total_reused_embeddings,
                )

                # =================================================================
                # 5. 선행 정리: 기존 프로세스 핵심 라벨만 삭제 (교차 실행 데이터 혼합 방지)
                # =================================================================
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[CLEANUP] 기존 프로세스/태스크 그래프를 정리합니다...",
                    "tool_usage_started", 12
                )

                try:
                    def _clear_process_core_labels_sync() -> Dict[str, Any]:
                        client = Neo4jClient(graph_name=age_graph_name)
                        try:
                            return client.clear_process_core_labels()
                        finally:
                            client.close()

                    cleanup_result = await asyncio.to_thread(_clear_process_core_labels_sync)
                    deleted_nodes = int(cleanup_result.get("deleted_nodes", 0) or 0)
                    logger.info(
                        "[CLEANUP] Process-core labels cleared before run: "
                        f"deleted_nodes={deleted_nodes}, labels={cleanup_result.get('labels', [])}"
                    )
                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[CLEANUP] 기존 그래프 정리 완료 (삭제 노드: {deleted_nodes})",
                        "tool_usage_finished", 14,
                        {"cleanup": cleanup_result}
                    )
                except Exception as e:
                    logger.error(f"[CLEANUP] Failed to clear process-core labels: {e}")
                    # Fail fast by design: run should not continue with mixed legacy graph data.
                    raise Exception(
                        f"Neo4j 선삭제 실패로 작업을 중단합니다: {e}"
                    ) from e

                # =================================================================
                # 6. PDF2BPMN 워크플로우를 "직접 호출"로 실행 (FastAPI BackgroundTasks 제거)
                # =================================================================
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[PROCESSING] PDF 분석 및 엔티티 추출을 시작합니다...",
                    "tool_usage_started", 15
                )

                # Import here to keep agent startup light
                from src.pdf2bpmn.workflow.graph import PDF2BPMNWorkflow  # type: ignore

                # IMPORTANT:
                # - 일부 단계는 asyncio.to_thread(...)에서 실행되며 progress_callback도 워커 스레드에서 호출됩니다.
                # - ProcessGPT SDK의 event_queue.enqueue_event()는 내부적으로 asyncio.create_task(...)를 사용하므로
                #   "실행 중인 이벤트 루프가 있는 스레드"에서만 호출되어야 합니다.
                # - 따라서 스레드에서 콜백이 오더라도 메인 루프 스레드로 안전하게 마샬링합니다.
                main_loop = asyncio.get_running_loop()

                def _enqueue_progress(msg: str, progress: int, extra: Optional[Dict[str, Any]] = None):
                    # 워크플로우 sub-progress(0–100) → 전체 파이프라인 12–66% 로 매핑
                    local_p = max(0, min(100, int(progress)))
                    global_p = 12 + int(local_p * 0.54)
                    event_data = {
                        "message": msg,
                        "status": "tool_usage_started",
                        "progress": global_p,
                        "job_id": job_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if extra:
                        event_data.update(extra)

                    evt = TaskStatusUpdateEvent(
                        status={
                            "state": TaskState.working,
                            "message": new_agent_text_message(
                                json.dumps(event_data, ensure_ascii=False),
                                context_id, task_id,
                            ),
                        },
                        final=False,
                        contextId=context_id,
                        taskId=task_id,
                        metadata={
                            "crew_type": "pdf2bpmn",
                            "event_type": "tool_usage_started",
                            "job_id": job_id,
                            "progress": int(progress),
                        },
                    )

                    # Always marshal to main loop thread (safe for both same-thread and worker-thread callers)
                    try:
                        main_loop.call_soon_threadsafe(event_queue.enqueue_event, evt)
                    except Exception:
                        # Extremely defensive fallback: if loop is unavailable, try direct enqueue
                        event_queue.enqueue_event(evt)

                workflow = PDF2BPMNWorkflow(graph_name=age_graph_name)

                # 사용자 [도구 설정] 다이얼로그에서 선택한 dedup 강도 적용.
                # 메인 에이전트가 [InputData].tool_settings 로 넣어 준 값을 _parse_query 가 보존했고,
                # 여기서 workflow 의 임계값을 일괄 오버라이드한다.
                #   - "concise"  : 임계 ↓ → 간결한 결과
                #   - "standard" : Config 기본값 (안 넣어도 동일)
                #   - "detailed" : 임계 ↑ → 원문에 가까운 자세한 결과
                try:
                    tool_settings_in = parsed.get("tool_settings") or {}
                    pdf2bpmn_level = (tool_settings_in.get("pdf2bpmnLevel") or "standard").strip().lower()
                    workflow.set_dedup_level(pdf2bpmn_level)
                    logger.info(
                        f"[DEDUP] User pdf2bpmnLevel='{pdf2bpmn_level}' applied "
                        f"(tool_settings={tool_settings_in})"
                    )
                except Exception as exc:
                    logger.warning(f"[DEDUP] tool_settings 적용 실패 → standard 유지: {exc}")
                    workflow.set_dedup_level("standard")

                state: Dict[str, Any] = {
                    # pdf_paths 는 메멘토 재사용 흐름에서는 사용하지 않지만, 워크플로우 state 스키마 호환을 위해 빈 리스트로 둔다.
                    "pdf_paths": [],
                    "documents": list(preloaded_documents),
                    "sections": list(preloaded_sections),
                    "reference_chunks": list(preloaded_chunks),
                    "processes": [],
                    "tasks": [],
                    "roles": [],
                    "gateways": [],
                    "events": [],
                    "skills": [],
                    "evidences": [],
                    "agent_generation_policy": "existing_only",
                    "confidence_threshold": 0.8,
                    "current_step": "ingest_pdf",
                    "error": None,
                    "bpmn_xml": None,
                    "bpmn_xmls": {},
                    "bpmn_files": {},
                    "skill_docs": {},
                    "dmn_xml": None,
                }

                try:
                    # Neo4j schema init (same as API)
                    await asyncio.to_thread(workflow.neo4j.init_schema)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 1: ingest_pdf 는 더 이상 호출하지 않는다.
                    # - 메멘토에서 받아온 청크/임베딩으로 이미 documents/sections/reference_chunks 를 채웠으므로
                    #   재파싱 없이 Neo4j에만 문서/섹션 노드를 등록한다.
                    if preloaded_documents:
                        _enqueue_progress("[STEP] 메멘토 청크 기반 문서/섹션 그래프 등록 중...", 20)
                        for doc in preloaded_documents:
                            workflow.neo4j.create_document(doc)
                        for sec in preloaded_sections:
                            workflow.neo4j.create_section(sec)
                    page_count = 0
                    try:
                        docs = state.get("documents") or []
                        if docs:
                            page_count = sum(int(getattr(d, "page_count", 0) or 0) for d in docs)
                    except Exception:
                        page_count = 0
                    chunk_count = len(state.get("reference_chunks") or [])
                    parsed_file_count = len(preloaded_documents)
                    _enqueue_progress(
                        f"[STEP] 메멘토 재사용 완료: 파일 {parsed_file_count}개, "
                        f"총 {page_count}페이지, {chunk_count}개 청크 "
                        f"(재사용 임베딩 {total_reused_embeddings}/{chunk_count})",
                        28
                    )

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 2: segment_sections
                    _enqueue_progress("[STEP] 섹션 분석 및 임베딩 생성 중...", 32)
                    state.update(await asyncio.to_thread(workflow.segment_sections, state))
                    section_count = len(state.get("sections") or [])
                    _enqueue_progress(f"[STEP] 섹션 분석 완료: {section_count}개 섹션", 38)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 3: extract_candidates_with_progress (LLM-heavy)
                    total_sections = len([s for s in (state.get("sections") or []) if getattr(s, "content", None) and len((s.content or "").strip()) >= 50])
                    _enqueue_progress(f"[STEP] 엔티티 추출 시작: {total_sections}개 섹션", 40, {"chunk_info": {"current": 0, "total": total_sections}})

                    def _progress_callback(current: int, total: int, msg: str):
                        # Map to 40~55
                        mapped = 40 + int((current / max(total, 1)) * 15)
                        _enqueue_progress(f"[EXTRACT] {msg}", mapped, {"chunk_info": {"current": current, "total": total}})

                    state.update(await asyncio.to_thread(workflow.extract_candidates_with_progress, state, _progress_callback))
                    process_count = len(state.get("processes") or [])
                    task_count = len(state.get("tasks") or [])
                    role_count = len(state.get("roles") or [])
                    _enqueue_progress(f"[STEP] 추출 완료: 프로세스 {process_count}, 태스크 {task_count}, 역할 {role_count}", 58)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 4: normalize_entities
                    _enqueue_progress("[STEP] 엔티티 정규화 및 중복 제거 중...", 62)
                    state.update(await asyncio.to_thread(workflow.normalize_entities, state))
                    _enqueue_progress("[STEP] 정규화 완료", 70)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 5: ontology skill generation is disabled by policy.
                    # Skill creation now happens from runtime task instructions per process definition.
                    state["skills"] = []
                    state["skill_docs"] = {}
                    _enqueue_progress("[STEP] 온톨로지 스킬 생성 스킵(지침 기반 후처리 사용)", 80)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 6: generate_dmn — 자동 생성.
                    # 사용자가 process loop 안 통합 elicit 에서 어떤 gateway 를 DMN 으로 만들지 선택하므로,
                    # 여기서는 일단 모든 gateway 의 후보를 만들고, process loop 끝난 직후에
                    # state["__hitl_approved_dmn_gateways"] 로 필터링한다.
                    _enqueue_progress("[STEP] DMN 의사결정 테이블 후보 생성 중...", 84)
                    state.update(await asyncio.to_thread(workflow.generate_dmn, state))
                    _enqueue_progress("[STEP] DMN 후보 생성 완료 (사용자 선택은 process loop 통합 elicit 에서)", 88)

                    if self.is_cancelled:
                        raise Exception("작업이 취소되었습니다.")

                    # Step 7: export_artifacts
                    _enqueue_progress("[STEP] 결과물 저장 중...", 92)
                    state.update(await asyncio.to_thread(workflow.export_artifacts, state))
                    _enqueue_progress("[STEP] PDF2BPMN 워크플로우 완료", 95)

                finally:
                    try:
                        workflow.neo4j.close()
                    except Exception:
                        pass

                # =================================================================
                # 7. 이번 작업에서 생성된 process_id 목록을 state에서 직접 수집 + Neo4j에서 상세 조회
                # =================================================================
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    "[GENERATING] 이번 작업의 추출 정보(Neo4j)로 ProcessGPT 프로세스 정의/유저 매핑을 생성합니다...",
                    "tool_usage_started", 58
                )
                job_process_ids: List[str] = []
                process_names_by_id: Dict[str, str] = {}
                processes_state = state.get("processes", []) or []
                for p in processes_state:
                    try:
                        pid = getattr(p, "proc_id", None) or getattr(p, "process_id", None) or getattr(p, "id", None)
                        pname = getattr(p, "name", None)
                        if pid:
                            job_process_ids.append(str(pid))
                            if pname:
                                process_names_by_id[str(pid)] = str(pname)
                    except Exception:
                        continue

                # 요청 단위 그래프 스냅샷 식별자
                request_graph_run_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
                extracted_by_proc_id: Dict[str, Dict[str, Any]] = {}
                if not job_process_ids:
                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        "[NOTICE] 문서에서 추출된 프로세스가 없어 생성할 BPMN이 없습니다. (이미지/슬라이드 위주 문서일 수 있습니다.)",
                        "tool_usage_finished", 100,
                        {"process_count": 0, "reason": "no_process_extracted"},
                    )
                else:
                    logger.info(f"[INFO] 이번 작업 기준 추출 프로세스: {len(job_process_ids)}개")

                    # Re-open Neo4j client for detail queries (workflow.neo4j was closed)
                    neo4j = Neo4jClient(graph_name=age_graph_name)
                    try:
                        for proc_id in job_process_ids:
                            try:
                                detail = await asyncio.to_thread(neo4j.get_process_with_details, proc_id)
                                if not detail:
                                    continue
                                flows = await asyncio.to_thread(neo4j.get_sequence_flows, proc_id)
                                if isinstance(flows, list):
                                    detail["sequence_flows"] = flows
                                graph_elements = await asyncio.to_thread(neo4j.get_process_graph_elements, proc_id)
                                detail = self._enrich_tasks_with_role_from_graph(
                                    detail=detail,
                                    graph_elements=(graph_elements or {}),
                                )
                                extracted_by_proc_id[proc_id] = {
                                    "detail": detail,
                                    "graph_elements": graph_elements or {},
                                    "process_name": (detail.get("process", {}) or {}).get("name")
                                    or process_names_by_id.get(proc_id)
                                    or "",
                                }
                            except Exception as e:
                                logger.warning(f"[WARN] process detail 조회 중 예외: proc_id={proc_id}, err={e}")
                    finally:
                        try:
                            neo4j.close()
                        except Exception:
                            pass

                extracted_count2 = len(extracted_by_proc_id)
                logger.info(f"[INFO] 이번 작업 기준 추출 프로세스: {extracted_count2}개")

                # -----------------------------------------------------------------
                # 동일/유사 이름 프로세스가 중복 추출된 경우, 저장 직전에 보수적으로 병합
                # - 여러 파일이 하나의 프로세스를 나눠 설명하는 케이스에서 중복 저장 방지
                # -----------------------------------------------------------------
                def _norm_proc_name(name: Any) -> str:
                    text = str(name or "").strip().lower()
                    text = re.sub(r"\s+", " ", text)
                    return text

                def _dedup_list(items: Any, key_builder):
                    if not isinstance(items, list):
                        return []
                    seen: Set[str] = set()
                    out: List[Any] = []
                    for item in items:
                        try:
                            key = key_builder(item)
                        except Exception:
                            key = ""
                        key = str(key or "").strip()
                        if not key:
                            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(item)
                    return out

                def _merge_graph_elements(g1: Any, g2: Any) -> Dict[str, Any]:
                    base = g1 if isinstance(g1, dict) else {}
                    inc = g2 if isinstance(g2, dict) else {}
                    e1 = base.get("elements") if isinstance(base.get("elements"), list) else []
                    e2 = inc.get("elements") if isinstance(inc.get("elements"), list) else []
                    merged: List[Dict[str, Any]] = []
                    seen_ids: Set[str] = set()
                    for el in (e1 + e2):
                        if not isinstance(el, dict):
                            continue
                        data = el.get("data") or {}
                        eid = str(data.get("id") or "").strip()
                        if not eid:
                            continue
                        if eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                        merged.append(el)
                    counts = dict(base.get("counts") or {})
                    counts["elements"] = len(merged)
                    return {
                        **base,
                        "elements": merged,
                        "counts": counts,
                    }

                merged_by_name: Dict[str, Dict[str, Any]] = {}
                for proc_id, pinfo in extracted_by_proc_id.items():
                    process_name = (pinfo.get("process_name") or "").strip()
                    norm_name = _norm_proc_name(process_name)
                    detail = pinfo.get("detail") or {}
                    graph_elements = pinfo.get("graph_elements") or {}

                    # 이름이 비어있으면 병합하지 않고 독립 유지
                    merge_key = norm_name if norm_name else f"__{proc_id}"
                    if merge_key not in merged_by_name:
                        pinfo_copy = dict(pinfo)
                        pinfo_copy["detail"] = dict(detail) if isinstance(detail, dict) else {}
                        pinfo_copy["graph_elements"] = dict(graph_elements) if isinstance(graph_elements, dict) else {}
                        pinfo_copy["_source_proc_ids"] = [proc_id]
                        merged_by_name[merge_key] = pinfo_copy
                        continue

                    target = merged_by_name[merge_key]
                    target_detail = target.get("detail") or {}
                    target_detail = target_detail if isinstance(target_detail, dict) else {}
                    incoming_detail = detail if isinstance(detail, dict) else {}

                    # process 본문은 설명이 더 긴 쪽을 우선
                    t_proc = target_detail.get("process") if isinstance(target_detail.get("process"), dict) else {}
                    i_proc = incoming_detail.get("process") if isinstance(incoming_detail.get("process"), dict) else {}
                    t_desc = str(t_proc.get("description") or "")
                    i_desc = str(i_proc.get("description") or "")
                    if len(i_desc) > len(t_desc):
                        target_detail["process"] = i_proc
                        if i_proc.get("name"):
                            target["process_name"] = str(i_proc.get("name"))

                    target_detail["tasks"] = _dedup_list(
                        (target_detail.get("tasks") or []) + (incoming_detail.get("tasks") or []),
                        lambda x: (x or {}).get("task_id") or (x or {}).get("id") or (x or {}).get("name") or "",
                    )
                    target_detail["roles"] = _dedup_list(
                        (target_detail.get("roles") or []) + (incoming_detail.get("roles") or []),
                        lambda x: (x or {}).get("role_id") or (x or {}).get("id") or (x or {}).get("name") or "",
                    )
                    target_detail["gateways"] = _dedup_list(
                        (target_detail.get("gateways") or []) + (incoming_detail.get("gateways") or []),
                        lambda x: (x or {}).get("gateway_id") or (x or {}).get("id") or (x or {}).get("name") or "",
                    )
                    target_detail["events"] = _dedup_list(
                        (target_detail.get("events") or []) + (incoming_detail.get("events") or []),
                        lambda x: (x or {}).get("event_id") or (x or {}).get("id") or (x or {}).get("name") or "",
                    )
                    target_detail["sequence_flows"] = _dedup_list(
                        (target_detail.get("sequence_flows") or target_detail.get("flows") or [])
                        + (incoming_detail.get("sequence_flows") or incoming_detail.get("flows") or []),
                        lambda x: f"{(x or {}).get('source')}>{(x or {}).get('target')}|{(x or {}).get('condition') or ''}",
                    )
                    target["detail"] = target_detail
                    target["graph_elements"] = _merge_graph_elements(target.get("graph_elements"), graph_elements)
                    src = target.get("_source_proc_ids") if isinstance(target.get("_source_proc_ids"), list) else []
                    src.append(proc_id)
                    target["_source_proc_ids"] = src

                if len(merged_by_name) != len(extracted_by_proc_id):
                    logger.info(
                        f"[MERGE] duplicate-name merge applied: {len(extracted_by_proc_id)} -> {len(merged_by_name)}"
                    )
                extracted_by_proc_id = {
                    (v.get("_source_proc_ids")[0] if isinstance(v.get("_source_proc_ids"), list) and v.get("_source_proc_ids") else k): v
                    for k, v in merged_by_name.items()
                }

                # -----------------------------------------------------------------
                # 요청 단위 통합 그래프 + 최종 프로세스 그래프 스냅샷 저장
                # -----------------------------------------------------------------
                try:
                    process_graphs: Dict[str, Dict[str, Any]] = {}
                    integrated_elements: List[Dict[str, Any]] = []
                    seen_element_ids: Set[str] = set()
                    for pid, pinfo in extracted_by_proc_id.items():
                        g = pinfo.get("graph_elements") or {}
                        if isinstance(g, dict):
                            process_graphs[pid] = g
                            for el in g.get("elements") or []:
                                if not isinstance(el, dict):
                                    continue
                                data = el.get("data") or {}
                                eid = str(data.get("id") or "").strip()
                                if not eid or eid in seen_element_ids:
                                    continue
                                seen_element_ids.add(eid)
                                integrated_elements.append(el)

                    integrated_graph = {
                        "run_id": request_graph_run_id,
                        "task_id": str(task_id or ""),
                        "graph_name": age_graph_name,
                        "process_ids": list(extracted_by_proc_id.keys()),
                        "elements": integrated_elements,
                        "counts": {
                            "elements": len(integrated_elements),
                            "processes": len(process_graphs),
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                    def _save_graph_snapshots_sync():
                        client = Neo4jClient(graph_name=age_graph_name)
                        try:
                            return client.save_request_graph_snapshots(
                                run_id=request_graph_run_id,
                                integrated_graph=integrated_graph,
                                process_graphs=process_graphs,
                                metadata={
                                    "task_id": str(task_id or ""),
                                    "tenant_id": str(effective_tenant_id or ""),
                                    "graph_name": age_graph_name,
                                    "process_count": len(process_graphs),
                                },
                            )
                        finally:
                            client.close()

                    graph_snapshot_result = await asyncio.to_thread(_save_graph_snapshots_sync)
                    logger.info(
                        "[GRAPH] request graph snapshots saved: "
                        f"run_id={request_graph_run_id}, result={graph_snapshot_result}"
                    )
                except Exception as e:
                    logger.warning(f"[WARN] request graph snapshot save failed: {e}")
            

            # --- end skip workflow when HITL resume ---
            # 9. 각 추출 프로세스에 대해 ProcessGPT 정의/유저매핑 → XML 생성 → DB 저장
            saved_processes: List[Dict[str, Any]] = []
            all_bpmn_xmls = {}  # proc_def_id -> bpmn_xml 매핑
            total_bpmn = len(extracted_by_proc_id)
            agent_user_ids_for_skill_sync: Set[str] = set()
            agent_skill_names_for_sync: Dict[str, Set[str]] = {}
            assigned_agent_user_ids: Set[str] = set()
            initial_agent_user_ids: Set[str] = {
                str(a.get("id"))
                for a in (self._agents or [])
                if isinstance(a, dict) and str(a.get("id") or "").strip()
            }
            uploaded_skill_names: List[str] = []

            # 프로세스 후처리(지침 기반)에서 생성된 스킬 메타 누적
            generated_skill_metas: List[Dict[str, Any]] = []
            if resume_cp:
                saved_processes = list(resume_cp.get("completed_processes") or [])
                generated_skill_metas = list(resume_cp.get("generated_skill_metas") or [])
                agent_user_ids_for_skill_sync = {
                    str(x) for x in (resume_cp.get("agent_user_ids_for_skill_sync") or []) if str(x).strip()
                }
                assigned_agent_user_ids = {
                    str(x) for x in (resume_cp.get("assigned_agent_user_ids") or []) if str(x).strip()
                }
                for aid, names in (resume_cp.get("agent_skill_names_for_sync") or {}).items():
                    if not str(aid or "").strip():
                        continue
                    agent_skill_names_for_sync[str(aid)] = {
                        str(n) for n in (names or []) if str(n or "").strip()
                    }

            def _sync_agent_graph_for_process_sync(
                neo4j_proc_id: str,
                role_agent_pairs: List[tuple[str, str]],
                agent_skill_names: Dict[str, Set[str]],
            ) -> None:
                """Runtime assignment 결과를 Neo4j 그래프(Agent 노드/엣지)에 반영."""
                if not neo4j_proc_id:
                    return
                users_by_id = {
                    str(u.get("id")): u
                    for u in (self._users or [])
                    if isinstance(u, dict) and u.get("id")
                }
                neo4j = Neo4jClient(graph_name=age_graph_name)
                try:
                    # 1) Agent node + Role -> Agent
                    for role_name, agent_id in role_agent_pairs:
                        aid = str(agent_id or "").strip()
                        if not aid:
                            continue
                        u = users_by_id.get(aid, {})
                        neo4j.create_agent(
                            agent_id=aid,
                            name=str(u.get("username") or u.get("name") or aid),
                            role=str(u.get("role") or ""),
                            tenant_id=str(u.get("tenant_id") or ""),
                        )
                        if role_name:
                            neo4j.link_role_to_agent_in_process(neo4j_proc_id, role_name, aid)

                    # 2) Agent -> Skill (by skill name)
                    for agent_id, skill_names in (agent_skill_names or {}).items():
                        aid = str(agent_id or "").strip()
                        if not aid:
                            continue
                        for sk in sorted({str(x).strip() for x in (skill_names or set()) if str(x).strip()}):
                            neo4j.link_agent_to_skill_by_name(aid, sk)
                finally:
                    neo4j.close()

            # =================================================================
            # PASS 1 — 모든 프로세스의 proc_json 생성 + HITL 후보 수집 (DB 저장/HITL 없음)
            #   resume 시: checkpoint 의 prepared_processes 를 복원하고 PASS1 을 건너뛴다
            #   (resume 시 extracted_by_proc_id 가 비어 있어 PASS1 for 루프가 자동 skip 됨)
            # =================================================================
            prepared_processes: List[Dict[str, Any]] = (
                list(resume_cp.get("prepared_processes") or []) if resume_cp else []
            )
            
            logger.info(f"[DEBUG] extracted_by_proc_id keys: {list(extracted_by_proc_id.keys())}")
            
            for idx, (proc_id, pinfo) in enumerate(extracted_by_proc_id.items()):
                process_name = pinfo.get("process_name") or f"Process {idx + 1}"
                detail = pinfo.get("detail") or {}
                
                logger.info(f"[DEBUG] Processing extracted process {idx+1}/{total_bpmn}: {process_name}")

                # extracted info -> ProcessGPT definition + BPMN XML
                # NOTE: Neo4j collect(DISTINCT t {.*}) does NOT preserve order; tasks come back arbitrary.
                # 게다가 추출 LLM 이 task.order 를 잘못 부여할 수 있어 (예: 종결 task 가 order=1)
                # task_order 단순 정렬은 시작 task 를 오선택한다.
                # → detail.sequence_flows 의 topological order 로 정렬 (cycle/누락은 task_order fallback).
                #   LLM/deterministic generator 모두 "extracted.tasks 의 순서" 를 신뢰하므로
                #   이 단계의 정렬이 BPMN 시작 task 와 흐름을 결정한다.
                raw_tasks = detail.get("tasks") or []
                raw_flows = detail.get("sequence_flows") or detail.get("flows") or []
                if isinstance(raw_tasks, list):
                    sorted_tasks_payload = self._topological_sort_tasks(
                        raw_tasks,
                        raw_flows if isinstance(raw_flows, list) else [],
                    )
                    # task_order backbone 으로 isolated task 보강.
                    # 이때 sorted_tasks_payload 가 task_order asc 순으로 (I) GLOBAL-ORDER 결과를
                    # 반영하므로, source text 위치 순서대로 backbone 이 만들어진다.
                    augmented_flows = self._augment_sequence_flows_for_isolated_tasks(
                        sorted_tasks_payload,
                        raw_flows if isinstance(raw_flows, list) else [],
                    )
                    try:
                        sample_names = [
                            str(t.get("name") or "")
                            for t in sorted_tasks_payload[:5]
                            if isinstance(t, dict)
                        ]
                        logger.info(
                            f"[PROCDEF][TOPO-SORT] proc={process_name!r} tasks={len(sorted_tasks_payload)} "
                            f"flows_in={len(raw_flows) if isinstance(raw_flows, list) else 0} "
                            f"flows_out={len(augmented_flows)} "
                            f"head={sample_names}"
                        )
                    except Exception:
                        pass
                else:
                    sorted_tasks_payload = []
                    augmented_flows = raw_flows if isinstance(raw_flows, list) else []
                extracted_payload = {
                    "process": detail.get("process") or {},
                    "tasks": sorted_tasks_payload,
                    "roles": detail.get("roles") or [],
                    "gateways": detail.get("gateways") or [],
                    "events": detail.get("events") or [],
                    "sequence_flows": augmented_flows,
                }

                # legacy flow is intentionally removed

                try:
                    generated = await self._generate_processgpt_definition_and_bpmn(
                        tenant_id=tenant_id,
                        process_name=process_name,
                        extracted=extracted_payload,
                        user_request=(user_input or ""),
                        consulting_outline=(consulting_outline_for_gen or None),
                    )
                except Exception as e:
                    logger.exception(
                        f"[PROCDEF][ERROR] process generation crashed; skipping process "
                        f"(index={idx+1}/{total_bpmn}, process={process_name!r}): {type(e).__name__}: {e}"
                    )
                    continue
                if not generated:
                    logger.warning(f"[WARN] ProcessGPT generation failed: {process_name}")
                    continue

                elements_model = generated.get("elements_model") or {}
                proc_json = generated.get("definition") or {}
                # 컨설팅 모드: LLM 이 채운 processDefinitionName 으로 PASS2 의 process_name 갱신
                _llm_pname = str(
                    proc_json.get("processDefinitionName")
                    or elements_model.get("processDefinitionName")
                    or ""
                ).strip()
                if _llm_pname and (not pinfo.get("process_name")):
                    process_name = _llm_pname
                proc_json = self._ensure_end_event_connectivity(
                    proc_json,
                    process_name=process_name,
                )

                # NEW: proc_def.definition에 "추출에 사용된 Neo4j proc_id/그래프"를 저장
                # - 프론트에서 실제 Neo4j 그래프(노드/관계)를 조회할 때 사용
                # - tenant_id/todo_id 도 함께 저장하여 프론트가 그래프 식별자를
                #   재구성할 수 있도록 한다 (graph_name 도 동일 값을 가짐)
                # - 컨설팅 모드(consulting_payload 가 있는 경우)는 Neo4j 그래프가 없으므로
                #   source 만 "consulting" 으로 표시하고 나머지 그래프 키는 생략한다.
                try:
                    ex = proc_json.get("extraction")
                    if not isinstance(ex, dict):
                        ex = {}
                    if consulting_payload:
                        ex["source"] = "consulting"
                        ex["task_id"] = str(task_id or "")
                        ex["tenant_id"] = str(effective_tenant_id or "")
                    else:
                        ex["source"] = "pdf2bpmn"
                        ex["neo4j_proc_id"] = str(proc_id)
                        ex["neo4j_graph_name"] = age_graph_name
                        ex["tenant_id"] = str(effective_tenant_id or "")
                        ex["todo_id"] = str(task_id or "")
                        ex["task_id"] = str(task_id or "")
                        ex["graph_run_id"] = request_graph_run_id
                        ex["graph_snapshot_ref"] = {
                            "run_id": request_graph_run_id,
                            "snapshot_type": "process",
                            "proc_id": str(proc_id),
                        }
                        ex["integrated_graph_ref"] = {
                            "run_id": request_graph_run_id,
                            "snapshot_type": "integrated",
                            "tenant_id": str(effective_tenant_id or ""),
                            "task_id": str(task_id or ""),
                        }
                        # 가벼운 임베드(디버깅/복구용): 실제 조회는 GraphSnapshot API 권장
                        if isinstance(pinfo.get("graph_elements"), dict):
                            ex["process_graph_preview"] = {
                                "process_id": str(proc_id),
                                "counts": (pinfo.get("graph_elements") or {}).get("counts") or {},
                            }
                    proc_json["extraction"] = ex
                except Exception:
                    pass
                
                # proc_def_id: UUID (already forced inside _generate_processgpt_definition_and_bpmn)
                proc_def_id = str(proc_json.get("processDefinitionId") or elements_model.get("processDefinitionId") or "").strip()
                if not proc_def_id:
                    # extremely defensive fallback (should not happen)
                    proc_def_id = str(uuid.uuid4())
                    proc_json["processDefinitionId"] = proc_def_id

                # === [HITL] 후보 추출 → 통합 elicit → 응답 반영 ===
                # 사용자에게 "스킬/에이전트/DMN 을 어떻게 생성할지" 를 한 번에 묻고, 응답대로만 실제 생성.
                # 단일 process 시나리오에서 사용자 개입 1회. 다중 process 면 process 마다 발생.

                # 1) 스킬 후보 추출 — _postprocess_skills_and_tasks 는 proc_json 을 mutate 하지만
                #    그 결과 자체가 후보 카드이므로, 거부된 항목은 _remove_rejected_skills_from_proc_json
                #    로 사후 정리한다.
                process_skill_metas = await self._postprocess_skills_and_tasks(
                    proc_json=proc_json,
                    process_name=process_name,
                )

                # 2) 에이전트 후보 수집 (INSERT 없음, LLM 프로필만 미리 만들어 미리보기 제공)
                agent_candidates_for_process: List[Dict[str, Any]] = []
                try:
                    agent_candidates_for_process = await self._collect_agent_candidates(
                        proc_json=proc_json,
                        tenant_id=tenant_id,
                        process_name=process_name,
                    )
                except Exception as exc:
                    logger.warning(f"[HITL][CAND][AGENT] 수집 실패: {exc}")

                # 3) DMN 후보 — 실제 생성될 proc_json 게이트웨이 기준으로 수집
                #    (parallel/분기<2 게이트웨이 제외 → 실제 DMN 으로 만들 수 있는 것만 노출)
                dmn_candidates = self._collect_dmn_candidates_from_proc_json(
                    proc_json=proc_json,
                    proc_def_id=proc_def_id,
                    process_name=process_name,
                )

                # 이 프로세스의 생성 결과 + HITL 후보를 모아둔다 (DB 저장은 PASS2 에서)
                prepared_processes.append({
                    "process_index": idx,
                    "proc_id": str(proc_id),
                    "process_name": process_name,
                    "proc_def_id": proc_def_id,
                    "proc_json": proc_json,
                    "extracted_payload": extracted_payload,
                    "process_skill_metas": process_skill_metas,
                    "agent_candidates_for_process": agent_candidates_for_process,
                    "dmn_candidates": dmn_candidates,
                })
            # --- PASS1 끝: 모든 proc_json 생성 + 스킬/에이전트/DMN 후보 수집 완료 ---

            # =================================================================
            # [HITL] 통합 elicit — 모든 프로세스의 스킬/에이전트/DMN 후보를 한 패널에
            #   묶어 1회만 질문한다 (사용자 개입 1회). 미응답이면 checkpoint 저장 후 pause.
            # =================================================================
            dmn_qid = stable_hitl_question_id(str(task_id), "dmn_apply")
            unified_questions: List[Dict[str, Any]] = []
            skill_qid_by_pdid: Dict[str, str] = {}
            agent_qid_by_pdid: Dict[str, str] = {}
            all_dmn_candidates: List[Dict[str, Any]] = []

            for prep in prepared_processes:
                p_pdid = str(prep.get("proc_def_id") or "")
                p_name = str(prep.get("process_name") or "")
                p_skill_metas = prep.get("process_skill_metas") or []
                p_agent_cands = prep.get("agent_candidates_for_process") or []
                all_dmn_candidates.extend(prep.get("dmn_candidates") or [])

                # 스킬 질문 (유효한 후보가 있을 때만)
                skill_items: List[Dict[str, Any]] = []
                for meta in p_skill_metas:
                    if not isinstance(meta, dict):
                        continue
                    safe = str(meta.get("safe_name") or "").strip()
                    name = str(meta.get("name") or "").strip()
                    if not safe or not name:
                        continue
                    summary = (
                        str(meta.get("description") or meta.get("summary") or meta.get("purpose") or "").strip()
                    )
                    if len(summary) > 140:
                        summary = summary[:140].rstrip() + "…"
                    skill_items.append({
                        "id": safe,
                        "label": name,
                        "description": summary or "반복 지침 기반 공통 스킬",
                    })
                if skill_items:
                    sk_qid = stable_hitl_question_id(str(task_id), f"skills_{p_pdid}")
                    skill_qid_by_pdid[p_pdid] = sk_qid
                    unified_questions.append(build_question_payload(
                        question=f"[{p_name}] 어떤 스킬을 생성할까요?",
                        feedback_type="select_items",
                        items=skill_items,
                        context=(
                            "선택한 스킬만 등록됩니다. 모두 해제하면 이 프로세스의 스킬 생성을 건너뜁니다.\n"
                            "직접 입력란에는 스킬 이름 변경 요청 (예: \"검토 처리 → 결재 검토\") 이나 추가 요청을 적을 수 있습니다."
                        ),
                        allow_multiple=True,
                        min_select=0,
                        allow_other=True,
                        target_type="skills_batch",
                        target_id=task_id,
                        question_id=sk_qid,
                        option_meta={
                            "tool": "pdf2bpmn", "key": "skills_approval", "stage": "skills",
                            "task_id": task_id, "proc_def_id": p_pdid,
                        },
                    ))

                # 에이전트 질문 (후보 있을 때만)
                if p_agent_cands:
                    ag_qid = stable_hitl_question_id(str(task_id), f"agents_{p_pdid}")
                    agent_qid_by_pdid[p_pdid] = ag_qid
                    agent_items = [
                        {
                            "id": c["candidate_id"],
                            "label": c["label"],
                            "description": c["description"],
                        }
                        for c in p_agent_cands
                    ]
                    unified_questions.append(build_question_payload(
                        question=f"[{p_name}] 어떤 에이전트를 생성/연결할까요?",
                        feedback_type="select_items",
                        items=agent_items,
                        context=(
                            "선택한 에이전트만 활동에 연결됩니다. '기존 에이전트 재사용' 표시가 있는 항목은 신규 생성 없이 매칭만 이뤄집니다.\n"
                            "직접 입력란에는 추가 의견을 자유롭게 적을 수 있습니다."
                        ),
                        allow_multiple=True,
                        min_select=0,
                        allow_other=True,
                        target_type="agents_batch",
                        target_id=task_id,
                        question_id=ag_qid,
                        option_meta={
                            "tool": "pdf2bpmn", "key": "agents_approval", "stage": "agents",
                            "task_id": task_id, "proc_def_id": p_pdid,
                        },
                    ))

            # DMN 질문 (전체 프로세스의 게이트웨이를 한 질문에 통합)
            if all_dmn_candidates:
                dmn_items = [
                    {
                        "id": c["candidate_id"],
                        "label": c["label"],
                        "description": c["description"],
                    }
                    for c in all_dmn_candidates
                ]
                unified_questions.append(build_question_payload(
                    question="DMN 의사결정 테이블을 어떤 게이트웨이에 만들까요?",
                    feedback_type="select_items",
                    items=dmn_items,
                    context=(
                        "선택한 게이트웨이만 DMN 의사결정 테이블로 변환됩니다.\n"
                        "선택하지 않은 게이트웨이는 BPMN 분기 조건만 사용합니다."
                    ),
                    allow_multiple=True,
                    min_select=0,
                    allow_other=True,
                    target_type="dmn_batch",
                    target_id=task_id,
                    question_id=dmn_qid,
                    option_meta={
                        "tool": "pdf2bpmn", "key": "dmn_apply", "stage": "dmn",
                        "task_id": task_id,
                    },
                ))

            # emit + pause (사용자 개입 1회). 응답이 모두 있으면 즉시 진행.
            hitl_entries_map: Dict[str, Optional[Dict[str, Any]]] = {}
            if unified_questions:
                qids_to_wait = [q["question_id"] for q in unified_questions if q.get("question_id")]
                if not self.supabase_client:
                    logger.error(
                        "[HITL][UNIFIED] supabase_client 없음 — 사용자 응답 대기 불가. "
                        "스킬/에이전트/DMN 은 기본값(미선택)으로 진행합니다."
                    )
                else:
                    out_hitl_state = await self._load_fresh_todolist_output(str(task_id))
                    prior_wait = str(out_hitl_state.get("hitl_wait_started_at") or "")
                    hitl_entries_map = await asyncio.to_thread(
                        read_batch_responses,
                        self.supabase_client,
                        str(task_id),
                        qids_to_wait,
                        prior_wait or None,
                    )
                    missing_qids = [q for q in qids_to_wait if hitl_entries_map.get(q) is None]
                    if missing_qids:
                        emit_waiting_for_user(
                            event_queue=event_queue,
                            context_id=context_id,
                            task_id=task_id,
                            job_id=job_id,
                            main_loop=main_loop,
                            questions=unified_questions,
                            progress=72,
                            message_text=(
                                f"[HITL] 사용자 확인 대기 중: {len(prepared_processes)}개 프로세스의 "
                                "스킬/에이전트/DMN 생성 결정"
                            ),
                        )
                        checkpoint = {
                            "version": 2,
                            "stage": "unified_post_procgen",
                            "prepared_processes": prepared_processes,
                            # DMN qid 는 후보가 있어 질문이 실제 생성된 경우에만 기록한다.
                            # (없는데 기록하면 재개 게이트가 묻지도 않은 답을 기다린다.)
                            "question_ids": {
                                "dmn": (dmn_qid if all_dmn_candidates else ""),
                                "skills": skill_qid_by_pdid,
                                "agents": agent_qid_by_pdid,
                            },
                            "job_process_ids": [
                                str(p.get("proc_id") or "") for p in prepared_processes
                            ],
                            "age_graph_name": age_graph_name,
                            "effective_tenant_id": effective_tenant_id,
                            "request_graph_run_id": request_graph_run_id,
                            "user_input": (user_input or "")[:8000],
                            "pdf_name": pdf_name,
                            "input_file_names": list(input_file_names),
                            "workflow_state": {
                                "dmn_decisions": state.get("dmn_decisions") or [],
                                "dmn_rules": state.get("dmn_rules") or [],
                                "skill_docs": state.get("skill_docs") or {},
                            },
                        }
                        await asyncio.to_thread(
                            pause_for_hitl,
                            self.supabase_client,
                            str(task_id),
                            checkpoint,
                            qids_to_wait,
                        )
                        logger.info(
                            "[HITL][UNIFIED] pause — todo_id=%s processes=%s (HUMAN_ASKED)",
                            task_id, len(prepared_processes),
                        )
                        raise HitlPauseException()
                    # 응답 모두 수신 — 재개 또는 즉시 진행
                    emit_human_feedback_received(
                        event_queue=event_queue,
                        context_id=context_id,
                        task_id=task_id,
                        job_id=job_id,
                        main_loop=main_loop,
                        question_id=qids_to_wait[0] if qids_to_wait else "",
                        summary="HITL 응답 수신 — 프로세스 생성 계속",
                        progress=74,
                    )
                    await asyncio.to_thread(
                        clear_hitl_checkpoint, self.supabase_client, str(task_id)
                    )
                    await asyncio.to_thread(
                        mark_hitl_process_resolved, self.supabase_client, str(task_id), 0
                    )

            # DMN 선택 파싱 — candidate_id 형식 "dmn::<proc_def_id>::<gateway_id>"
            #   approved_dmn_tokens 에는 "<proc_def_id>::<gateway_id>" 토큰을 담는다.
            approved_dmn_tokens: Set[str] = set()
            dmn_entry = hitl_entries_map.get(dmn_qid)
            if dmn_entry is not None and not hitl_is_skipped(dmn_entry):
                for x in hitl_selected_ids(dmn_entry):
                    sx = str(x or "")
                    if sx.startswith("dmn::"):
                        approved_dmn_tokens.add(sx[len("dmn::"):])
            state["__hitl_dmn_decided"] = True
            state["__hitl_approved_dmn_gateways"] = approved_dmn_tokens

            # =================================================================
            # PASS 2 — 각 프로세스에 HITL 응답을 반영하고 DB 에 저장
            # =================================================================
            total_bpmn = len(prepared_processes)
            # 프로세스별 실행 검증 리포트 누적 (최종 결과/아티팩트에 첨부)
            validation_results: Dict[str, Any] = {}
            for prepared in prepared_processes:
                idx = int(prepared.get("process_index") or 0)
                proc_id = str(prepared.get("proc_id") or "")
                process_name = str(prepared.get("process_name") or f"Process {idx + 1}")
                proc_def_id = str(prepared.get("proc_def_id") or "")
                proc_json = prepared.get("proc_json") or {}
                extracted_payload = prepared.get("extracted_payload") or {}
                # 폼 생성 단계가 실패해도 검증 호출에서 참조할 수 있도록 미리 초기화
                forms_result: Dict[str, Any] = {}
                process_skill_metas = list(prepared.get("process_skill_metas") or [])
                agent_candidates_for_process = list(prepared.get("agent_candidates_for_process") or [])

                # 이 프로세스의 스킬/에이전트 HITL 응답 파싱
                sk_qid = skill_qid_by_pdid.get(proc_def_id) or ""
                ag_qid = agent_qid_by_pdid.get(proc_def_id) or ""
                approved_skill_keys: Optional[Set[str]] = None
                approved_agent_ids: Optional[Set[str]] = None
                if sk_qid:
                    sk_entry = hitl_entries_map.get(sk_qid)
                    if sk_entry is not None and not hitl_is_skipped(sk_entry):
                        approved_skill_keys = {
                            self._normalize_skill_key(x)
                            for x in hitl_selected_ids(sk_entry)
                            if str(x or "").strip()
                        }
                    else:
                        approved_skill_keys = set()
                if ag_qid:
                    ag_entry = hitl_entries_map.get(ag_qid)
                    if ag_entry is not None and not hitl_is_skipped(ag_entry):
                        approved_agent_ids = set(hitl_selected_ids(ag_entry))
                    else:
                        approved_agent_ids = set()

                # 응답 반영 (스킬/에이전트)
                if approved_skill_keys is not None or approved_agent_ids is not None:
                    process_skill_metas = await self._apply_unified_hitl_to_process(
                        proc_json=proc_json,
                        process_name=process_name,
                        process_skill_metas=process_skill_metas,
                        agent_candidates_for_process=agent_candidates_for_process,
                        approved_skill_keys=approved_skill_keys,
                        approved_agent_ids=approved_agent_ids,
                        state=state,
                        tenant_id=tenant_id,
                        agent_user_ids_for_skill_sync=agent_user_ids_for_skill_sync,
                        agent_skill_names_for_sync=agent_skill_names_for_sync,
                    )
                generated_skill_metas.extend(process_skill_metas)

                # DMN 적용 — 사용자가 선택한 게이트웨이만 proc_json 에 DMN 테이블 보강
                # candidate_id 는 같은 이름의 게이트웨이를 '~' 로 묶을 수 있으므로 분해한다.
                this_proc_dmn_gids: Set[str] = set()
                for tok in approved_dmn_tokens:
                    if "::" not in tok:
                        continue
                    tok_pdid, tok_gids = tok.split("::", 1)
                    if tok_pdid != proc_def_id:
                        continue
                    for g in tok_gids.split("~"):
                        g = g.strip()
                        if g:
                            this_proc_dmn_gids.add(g)
                if this_proc_dmn_gids:
                    proc_json = self._augment_runtime_with_gateway_dmn(
                        runtime_def=proc_json,
                        extracted=extracted_payload,
                        approved_gateway_ids=this_proc_dmn_gids,
                    )
                    prepared["proc_json"] = proc_json
                    logger.info(
                        "[HITL][DMN] proc=%s — 게이트웨이 %d개를 DMN 의사결정 테이블로 생성",
                        process_name, len(this_proc_dmn_gids),
                    )
                
                # DB에 저장
                proc_def_data = {
                    "id": proc_def_id,
                    "name": process_name,
                    "definition": proc_json,
                    # 요청사항: XML 생성/저장 비활성화 -> bpmn은 null(None)로 저장
                    "bpmn": None,
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

                # -----------------------------------------------------------------
                # B안: proc_def 먼저 저장 → 폼 생성/저장(프론트 없이도 워커가 수행)
                # - 실패해도 폴백 폼을 만들어 form_def에 저장 시도
                # -----------------------------------------------------------------
                if save_result:
                    try:
                        await self._send_progress_event(
                            event_queue, context_id, task_id, job_id,
                            f"[FORM] 프로세스 폼 생성/저장을 시작합니다: {process_name}",
                            "tool_usage_started", 78,
                            {"proc_def_id": proc_def_id, "process_name": process_name},
                        )
                        forms_result = await self._ensure_forms_for_process(
                            proc_def_id=proc_def_id,
                            process_name=process_name,
                            proc_json=proc_json,
                            tenant_id=tenant_id,
                            event_queue=event_queue,
                            context_id=context_id,
                            task_id=task_id,
                            job_id=job_id,
                        )
                        # 폼 id를 activity.tool에 반영했으므로, proc_def.definition도 동기화 업데이트
                        await self._update_proc_def_definition_only(
                            proc_def_id=proc_def_id,
                            tenant_id=tenant_id,
                            definition=proc_json,
                        )

                        # -----------------------------------------------------------------
                        # NEW: After forms exist, expand process:
                        # - inputData wiring based on REAL saved forms (form_id + fields_json)
                        # - (re)apply agent assignment near-final
                        # - sanitize to avoid referencing future/non-existent form fields
                        # -----------------------------------------------------------------
                        try:
                            await self._expand_process_after_forms(
                                proc_def_id=proc_def_id,
                                process_name=process_name,
                                proc_json=proc_json,
                                forms_result=forms_result,
                                extracted=extracted_payload,
                                tenant_id=tenant_id,
                                event_queue=event_queue,
                                context_id=context_id,
                                task_id=task_id,
                                job_id=job_id,
                            )
                            proc_json = self._backfill_activity_content_from_extracted(
                                runtime_def=proc_json,
                                extracted=extracted_payload,
                            )
                            proc_json = self._ensure_end_event_connectivity(
                                proc_json,
                                process_name=process_name,
                            )
                            # proc_json changed (inputData/agent fields), persist definition again
                            await self._update_proc_def_definition_only(
                                proc_def_id=proc_def_id,
                                tenant_id=tenant_id,
                                definition=proc_json,
                            )
                        except Exception as e:
                            logger.warning(f"[WARN] expand(after-forms) stage failed: {e}")

                        await self._send_progress_event(
                            event_queue, context_id, task_id, job_id,
                            f"[FORM] 프로세스 폼 처리 완료: {process_name} (saved={forms_result.get('forms_saved')}/{forms_result.get('activities')})",
                            "tool_usage_finished", 82,
                            {"proc_def_id": proc_def_id, "forms_result": forms_result},
                        )
                    except Exception as e:
                        logger.warning(f"[WARN] form generation/save stage failed unexpectedly: {e}")

                    # -----------------------------------------------------------------
                    # FINAL(XML disabled): 요청사항에 따라 BPMN XML 생성/저장 비활성화
                    # -----------------------------------------------------------------
                    all_bpmn_xmls[proc_def_id] = None

                    # -----------------------------------------------------------------
                    # VALIDATION: 생성된 프로세스를 실행 엔진으로 검증/자동개선
                    #  - start→end 실제 실행 테스트 후, 결함이 있으면 proc_json 자동 교정
                    #  - 교정 시 _update_proc_def_definition_only 로 이미 재저장된 상태
                    # -----------------------------------------------------------------
                    try:
                        val_report = await self._validate_generated_process(
                            proc_def_id=proc_def_id,
                            process_name=process_name,
                            proc_json=proc_json,
                            forms_result=forms_result,
                            extracted=extracted_payload,
                            tenant_id=tenant_id,
                            event_queue=event_queue,
                            context_id=context_id,
                            task_id=task_id,
                            job_id=job_id,
                        )
                        # 검증 과정에서 proc_json 이 교정됐다면 최신본으로 갱신
                        fixed_def = val_report.get("final_definition")
                        if isinstance(fixed_def, dict) and fixed_def:
                            proc_json = fixed_def
                            prepared["proc_json"] = proc_json
                        # 최종 결과 payload 비대화 방지: 전체 정의는 proc_def 에 이미 저장돼 있으므로 제외
                        val_report.pop("final_definition", None)
                        validation_results[proc_def_id] = val_report
                    except Exception as e:
                        logger.warning(f"[VALIDATION] 검증 단계 실패(무시): {e}")
                        validation_results[proc_def_id] = {
                            "proc_def_id": proc_def_id,
                            "process_name": process_name,
                            "skipped": True,
                            "skip_reason": f"검증 단계 예외: {e}",
                            "passed": None,
                        }

                # saved_processes에 bpmn_xml 포함
                #   - neo4j_proc_id 를 함께 저장하여 프론트가 process_graphs 캐시를
                #     `neo4j_proc_id` 키로 매칭할 수 있도록 한다.
                saved_processes.append({
                    "id": proc_def_id,
                    "name": process_name,
                    "neo4j_proc_id": str(proc_id),
                    "bpmn_xml": all_bpmn_xmls.get(proc_def_id)  # XML 생성 비활성화: None
                })

                # 스킬 매핑용: activity에 매핑된 agent user_id 수집(best-effort)
                try:
                    acts = proc_json.get("activities") or []
                    proc_skill_name_by_key = {
                        str(s.get("id") or "").strip(): str(s.get("name") or "").strip()
                        for s in (proc_json.get("skills") or [])
                        if isinstance(s, dict)
                    }
                    for s in (proc_json.get("skills") or []):
                        if not isinstance(s, dict):
                            continue
                        safe_name = str(s.get("safe_name") or "").strip()
                        name = str(s.get("name") or "").strip()
                        if safe_name and name and safe_name not in proc_skill_name_by_key:
                            proc_skill_name_by_key[safe_name] = name
                    process_role_agent_pairs: List[tuple[str, str]] = []
                    process_agent_skill_names: Dict[str, Set[str]] = {}
                    process_agent_activity_texts: Dict[str, List[str]] = {}
                    process_agent_roles: Dict[str, Set[str]] = {}
                    if isinstance(acts, list):
                        for a in acts:
                            if isinstance(a, dict):
                                aid = str(a.get("agent") or "").strip()
                                role_name = str(a.get("role") or "").strip()
                                if aid:
                                    assigned_agent_user_ids.add(aid)
                                    agent_user_ids_for_skill_sync.add(aid)
                                    if role_name:
                                        process_role_agent_pairs.append((role_name, aid))
                                        process_agent_roles.setdefault(aid, set()).add(role_name)
                                    process_agent_activity_texts.setdefault(aid, []).append(
                                        " ".join(
                                            str(x or "")
                                            for x in (
                                                a.get("name"),
                                                a.get("instruction"),
                                                a.get("description"),
                                                a.get("tool"),
                                                role_name,
                                            )
                                        )
                                    )
                                    # task->skill 기준으로 에이전트 스킬 동기화 후보 누적
                                    task_skill_ids = [
                                        str(x).strip()
                                        for x in (a.get("skills") or [])
                                        if str(x).strip()
                                    ]
                                    task_skill_names = [
                                        proc_skill_name_by_key.get(sid) or sid
                                        for sid in task_skill_ids
                                    ]
                                    if task_skill_names:
                                        agent_skill_names_for_sync.setdefault(aid, set()).update(task_skill_names)
                                        process_agent_skill_names.setdefault(aid, set()).update(task_skill_names)

                    # 에이전트 프로필(역할/goal/persona/tools) 기반으로 스킬 매칭 보강
                    users_by_id = {
                        str(u.get("id")): u
                        for u in (self._users or [])
                        if isinstance(u, dict) and u.get("id")
                    }
                    for aid, texts in process_agent_activity_texts.items():
                        profile = users_by_id.get(aid, {})
                        activity_text = " ".join(texts[:20])
                        role_hints = process_agent_roles.get(aid, set())
                        scored: List[tuple[float, str]] = []
                        for sm in generated_skill_metas:
                            name = str(sm.get("name") or "").strip()
                            if not name:
                                continue
                            score = self._match_skill_score_for_agent(
                                agent_profile=profile,
                                skill_meta=sm,
                                activity_text=activity_text,
                                role_hints=role_hints,
                            )
                            if score >= 0.22:
                                scored.append((score, name))
                        if scored:
                            scored.sort(key=lambda x: x[0], reverse=True)
                            top_agent_skills = [n for _, n in scored[:5]]
                            agent_skill_names_for_sync.setdefault(aid, set()).update(top_agent_skills)
                            process_agent_skill_names.setdefault(aid, set()).update(top_agent_skills)

                    # Neo4j 그래프에 Agent 노드/엣지 반영 (Task->Skill 직접 연결은 사용하지 않음)
                    await asyncio.to_thread(
                        _sync_agent_graph_for_process_sync,
                        str(proc_id),
                        process_role_agent_pairs,
                        process_agent_skill_names,
                    )
                except Exception:
                    pass
                
                # 진행 이벤트 (XML 포함)
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    f"[SAVED] 프로세스 저장 완료: {process_name}",
                    "tool_usage_finished", 75 + int(8 * (idx + 1) / max(total_bpmn, 1)),
                    {
                        "process_id": proc_def_id, 
                        "process_name": process_name,
                        "bpmn_xml": all_bpmn_xmls.get(proc_def_id)  # XML 생성 비활성화: None
                    }
                )
            
            # 10. 최종 결과 구성 (saved_processes에는 이미 bpmn_xml 포함됨)
            actual_count = len(saved_processes)
            logger.info(f"[DEBUG] Actual saved process count: {actual_count}")

            # DMN 응답 반영은 PASS2 에서 proc_json 별로 이미 완료됨:
            #   사용자가 선택한 게이트웨이만 _augment_runtime_with_gateway_dmn 으로
            #   해당 proc_json 에 DMN 의사결정 테이블을 생성한 뒤 DB 저장했다.
            #   (이전의 state.dmn_decisions 필터링 방식은 추출 단계 엔티티를 대상으로 해
            #    proc_def 의 실제 게이트웨이와 ID 가 일치하지 않아 제거되었다.)

            # 10.5 NEW: 생성된 스킬을 Claude Skills 서비스에 업로드하고 Supabase에 동기화
            # - 워크플로우에서 만든 skill_docs(markdown)를 제품이 쓰는 스킬 저장소로 등록
            try:
                skill_docs: Dict[str, str] = state.get("skill_docs") or {}
                if (not isinstance(skill_docs, dict) or not skill_docs) and generated_skill_metas:
                    # LLM 으로 enrich 된 스킬 카드를 풍부한 SKILL.md 로 직렬화해 업로드 대상 구성.
                    # (개요/사용 시점/입력/산출물/절차/예시/주의사항/출처 섹션 포함)
                    skill_docs = {}
                    for sm in generated_skill_metas:
                        if not isinstance(sm, dict):
                            continue
                        sname = str(sm.get("name") or "").strip()
                        if not sname:
                            continue
                        key = self._normalize_skill_key(sname)
                        try:
                            md = render_skill_markdown(sm)
                        except Exception as e:
                            logger.warning(
                                "[SKILL] render_skill_markdown failed for %r: %s → minimal fallback",
                                sname, e,
                            )
                            summary = str(sm.get("summary") or "").strip()
                            procedure_text = str(sm.get("procedure_text") or "").strip()
                            source_ids = sm.get("source_activity_ids") or []
                            src_line = ", ".join(str(x) for x in source_ids if str(x).strip())
                            md = (
                                f"---\nname: {sname}\n---\n\n"
                                f"# {sname}\n\n"
                                f"## 개요\n{summary or '반복 지침 기반 공통 스킬'}\n\n"
                                f"## 절차\n{procedure_text or summary or sname}\n\n"
                                f"## 출처\n{src_line or '-'}\n"
                            )
                        skill_docs[key] = md
                if isinstance(skill_docs, dict) and skill_docs:
                    # 사용자가 process loop 안의 통합 elicit 에서 이미 결정함.
                    # state["__hitl_approved_skill_keys"] 에 승인된 스킬 키가 있으면 그대로 사용,
                    # 없으면 (단일 process 시나리오 아니거나 응답 없었음) 모두 자동 승인.
                    approved_keys: Optional[Set[str]] = state.get("__hitl_approved_skill_keys")
                    if approved_keys is not None and not isinstance(approved_keys, set):
                        approved_keys = set(approved_keys) if approved_keys else set()

                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[SKILL] 생성된 스킬({len(skill_docs)}) 업로드/동기화를 시작합니다...",
                        "tool_usage_started", 88,
                        {"skill_count": len(skill_docs), "approved_count": (
                            len(approved_keys) if approved_keys is not None else len(skill_docs)
                        )},
                    )

                    uploaded: List[str] = []
                    for key, md in skill_docs.items():
                        if not isinstance(md, str) or not md.strip():
                            continue
                        # 사용자가 선택한 키만 업로드. None 이면 폴백(전체 자동 승인).
                        if approved_keys is not None and key not in approved_keys:
                            continue
                        sname = self._extract_skill_name_from_markdown(md) or "generated-skill"
                        safe = self._normalize_skill_key(sname) or "generated-skill"
                        ok = await self._upload_skill_to_claude_skills(
                            tenant_id=tenant_id,
                            skill_name=safe,
                            file_name="SKILL.md",
                            content=md,
                        )
                        if ok:
                            uploaded.append(safe)
                    uploaded_skill_names = list(uploaded)

                    # Supabase sync (best-effort)
                    uploaded_set = {self._normalize_skill_key(x) for x in uploaded if str(x or "").strip()}
                    if agent_skill_names_for_sync:
                        for agent_id, desired_skills in agent_skill_names_for_sync.items():
                            if not desired_skills:
                                continue
                            assigned = [
                                self._normalize_skill_key(s)
                                for s in desired_skills
                                if self._normalize_skill_key(s) in uploaded_set
                            ]
                            if not assigned:
                                # 매칭되는 스킬이 없으면 스킵 (요구사항)
                                continue
                            await self._sync_skills_to_supabase(
                                tenant_id=tenant_id,
                                skill_names=assigned,
                                agent_user_ids={agent_id},
                            )

                    await self._send_progress_event(
                        event_queue, context_id, task_id, job_id,
                        f"[SKILL] 스킬 업로드/동기화 완료: {len(uploaded)}/{len(skill_docs)}",
                        "tool_usage_finished", 94,
                        {
                            "skills_uploaded": uploaded,
                            "agents_updated": len(agent_skill_names_for_sync),
                            "agents_seen": len(agent_user_ids_for_skill_sync),
                        },
                    )
            except Exception as e:
                logger.warning(f"[WARN] skill upload/sync stage failed: {e}")
            
            completed_message = (
                "[COMPLETED] PDF2BPMN 변환 완료: 문서에서 프로세스를 추출하지 못해 생성할 BPMN이 없습니다."
                if actual_count == 0
                else f"[COMPLETED] PDF2BPMN 변환 완료: {actual_count}개의 프로세스가 생성되었습니다."
            )

            # 최종 요약: 스킬/에이전트 결과도 프론트에서 바로 렌더링 가능한 형태로 제공
            uploaded_skill_set = {str(x or "").strip() for x in (uploaded_skill_names or []) if str(x or "").strip()}
            saved_skills_summary: List[Dict[str, Any]] = []
            seen_skill = set()
            for sm in generated_skill_metas:
                if not isinstance(sm, dict):
                    continue
                name = str(sm.get("name") or "").strip()
                safe_name = str(sm.get("safe_name") or self._normalize_skill_key(name) or "").strip()
                if not name or not safe_name:
                    continue
                key = safe_name.lower()
                if key in seen_skill:
                    continue
                seen_skill.add(key)
                saved_skills_summary.append(
                    {
                        "name": name,
                        "safe_name": safe_name,
                        "url_path": f"/skills/{quote(name, safe='')}",
                        "uploaded": safe_name in uploaded_skill_set,
                    }
                )

            users_by_id_now = {
                str(u.get("id")): u
                for u in (self._users or [])
                if isinstance(u, dict) and str(u.get("id") or "").strip()
            }
            saved_agents_summary: List[Dict[str, Any]] = []
            for aid in sorted({str(x).strip() for x in assigned_agent_user_ids if str(x).strip()}):
                row = users_by_id_now.get(aid, {})
                saved_agents_summary.append(
                    {
                        "id": aid,
                        "name": str(row.get("username") or row.get("name") or aid),
                        "role": str(row.get("role") or ""),
                        "created": aid not in initial_agent_user_ids,
                    }
                )

            # ----------------------------------------------------------------
            # AGE 그래프 elements 추출
            #   - ScaledJob 워커는 처리 후 Pod 가 종료되므로 외부에서 AGE 를 쿼리할
            #     수단(API 서버)이 없다. 따라서 그래프 미리보기에 필요한 데이터를
            #     워커가 처리 종료 직전에 모두 추출해 결과 메시지에 함께 첨부한다.
            #   - 프론트는 이 첨부 데이터를 그대로 렌더링하므로 별도 API 호출 불필요.
            #   - 추출 후 finally 블록에서 AGE 의 그래프 자체는 drop 한다 (정리).
            # ----------------------------------------------------------------
            integrated_graph_full: Dict[str, Any] = {"elements": [], "counts": {"nodes": 0, "edges": 0}}
            process_graphs: Dict[str, Any] = {}
            neo4j_for_graph: Any = None
            neo4j_graph_owned = False
            try:
                if workflow is not None and getattr(workflow, "neo4j", None):
                    neo4j_for_graph = workflow.neo4j
                elif age_graph_name:
                    neo4j_for_graph = Neo4jClient(graph_name=age_graph_name)
                    neo4j_graph_owned = True
                if neo4j_for_graph is not None:
                    integrated_graph_full = neo4j_for_graph.get_full_graph_elements(max_nodes=3000) or {
                        "elements": [], "counts": {"nodes": 0, "edges": 0}
                    }
                    for p in saved_processes:
                        pid = str(p.get("neo4j_proc_id") or p.get("id") or "").strip()
                        if not pid:
                            continue
                        try:
                            process_graphs[pid] = neo4j_for_graph.get_process_graph_elements(pid) or {
                                "elements": [], "counts": {"nodes": 0, "edges": 0}
                            }
                        except Exception as exc_proc:
                            logger.warning(f"[GRAPH] process({pid}) elements 추출 실패: {exc_proc}")
                    logger.info(
                        f"[GRAPH] elements 추출 완료 — full: nodes={integrated_graph_full.get('counts', {}).get('nodes', 0)}, "
                        f"edges={integrated_graph_full.get('counts', {}).get('edges', 0)}, "
                        f"per_process: {len(process_graphs)} 개"
                    )
            except Exception as exc_full:
                logger.warning(f"[GRAPH] AGE elements 추출 실패 (그래프 미리보기 누락 가능): {exc_full}")
            finally:
                if neo4j_graph_owned and neo4j_for_graph is not None:
                    try:
                        neo4j_for_graph.close()
                    except Exception:
                        pass

            final_result = {
                "message": completed_message,
                "status": "completed",
                "job_id": job_id,
                "task_id": str(task_id or ""),
                "graph_run_id": request_graph_run_id,
                "pdf_name": pdf_name,
                "pdf_names": input_file_names,
                "file_count": len(input_file_names),
                "process_count": actual_count,
                "saved_processes": saved_processes,  # bpmn_xml 포함
                "saved_skills": saved_skills_summary,
                "saved_agents": saved_agents_summary,
                # 프로세스별 실행 검증 리포트 (start→end 실행 테스트 + 자동개선 결과)
                "validation": validation_results,
                # 프론트 그래프 미리보기 전용 payload (final_artifact 와 동일 내용)
                #   - 이 데이터로 프론트는 외부 API 호출 없이 그래프를 즉시 렌더링한다.
                #   - integrated_graph: 통합(전체) 그래프 elements (showIntegratedGraphByTask 용)
                #   - process_graphs: { neo4j_proc_id: { elements, counts } }
                "integrated_graph": integrated_graph_full,
                "process_graphs": process_graphs,
                "graph_name": str(
                    age_graph_name
                    or (getattr(workflow, "neo4j", None) and workflow.neo4j.graph_name)
                    or ""
                ),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # 11. 최종 결과 아티팩트 이벤트 (browser_use와 동일한 패턴)
            # 이 이벤트가 프론트엔드에서 최종 결과로 사용됨
            # saved_processes에서 요약 정보만 추출 (draft 크기 제한 고려)
            saved_processes_summary = [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "neo4j_proc_id": p.get("neo4j_proc_id") or "",
                }
                for p in saved_processes
            ]

            # 검증 리포트 요약 (아티팩트 크기 제한 고려 — history/trace 등 상세는 제외)
            validation_summary: Dict[str, Any] = {}
            for _pdid, _rep in (validation_results or {}).items():
                if not isinstance(_rep, dict):
                    continue
                validation_summary[_pdid] = {
                    "process_name": _rep.get("process_name"),
                    "passed": _rep.get("passed"),
                    "skipped": _rep.get("skipped"),
                    "skip_reason": _rep.get("skip_reason"),
                    "iterations": _rep.get("iterations"),
                    "repaired": _rep.get("repaired"),
                    "remaining_defects": _rep.get("remaining_defects") or [],
                    "note": _rep.get("note"),
                }

            final_artifact_data = {
                "type": "pdf2bpmn_result",
                "task_id": str(task_id or ""),
                "graph_run_id": request_graph_run_id,
                "pdf_name": pdf_name,
                "pdf_names": input_file_names,
                "file_count": len(input_file_names),
                "process_count": actual_count,
                "saved_processes": saved_processes_summary,  # 요약만
                "saved_skills": saved_skills_summary,
                "saved_agents": saved_agents_summary,
                "validation": validation_summary,  # 프로세스별 검증 결과 요약
                "bpmn_xmls": all_bpmn_xmls,  # 모든 XML 내용
                # 프론트 그래프 미리보기 전용 payload.
                #  - integrated_graph: 통합(전체) 그래프 elements (showIntegratedGraphByTask 용)
                #  - process_graphs: { neo4j_proc_id: { elements, counts } } (프로세스별 그래프 미리보기용)
                # 프론트는 우선 이 데이터를 사용하며, 없을 때만 외부 API 로 fallback 한다.
                "integrated_graph": integrated_graph_full,
                "process_graphs": process_graphs,
                "graph_name": str(
                    age_graph_name
                    or (getattr(workflow, "neo4j", None) and workflow.neo4j.graph_name)
                    or ""
                ),
                "success": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "task_type": "pdf2bpmn"
            }
            
            event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    artifact=new_text_artifact(
                        name="PDF2BPMN Result",
                        description=f"PDF2BPMN 변환 결과: {actual_count}개 프로세스 생성",
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
                        "process_count": actual_count
                    }
                )
            )
            
            logger.info(f"[DONE] Task completed: {job_id} ({actual_count} processes)")
            
        except HitlPauseException:
            hitl_paused_only = True
            logger.info(
                "[HITL] execute paused — todo HUMAN_ASKED, consumer released; "
                "will resume on FB_REQUESTED"
            )
            return

        except httpx.ConnectError as e:
            # 보통은 _download_file에서 ConnectError를 Exception으로 감싸 올리지만,
            # 방어적으로 남겨둡니다(네트워크 계층 오류).
            logger.error(f"[ERROR] Network connection error: {e}")
            error_msg = f"네트워크 연결 오류가 발생했습니다: {str(e)}"
            await self._send_error_event(event_queue, context_id, task_id, job_id, error_msg, "connection_error")

        except Exception as e:
            logger.error(f"[ERROR] Task execution error: {e}")
            logger.error(traceback.format_exc())
            await self._send_error_event(event_queue, context_id, task_id, job_id, str(e), type(e).__name__)
        
        finally:
            # 임시 파일 정리
            for p in list(temp_paths_to_cleanup):
                if not p:
                    continue
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                        logger.info(f"[CLEANUP] Removed temp file: {p}")
                except Exception as e:
                    logger.warning(f"[WARN] Failed to remove temp file: {e}")

            # AGE 그래프 정리
            #   - 처리 결과(노드/엣지 elements)는 위에서 final_artifact 에 첨부했고,
            #     프론트는 그 첨부 데이터로 그래프를 렌더링한다.
            #   - 따라서 이 ScaledJob 워커가 종료되기 전 AGE 인스턴스에 남은 그래프를
            #     drop 하여 데이터가 무한정 부풀지 않도록 정리한다.
            try:
                if hitl_paused_only:
                    logger.info(
                        "[CLEANUP] AGE graph drop skipped (HITL pause — resume needs graph): %s",
                        age_graph_name_for_cleanup,
                    )
                elif age_graph_name_for_cleanup:
                    dropped = False
                    if workflow is not None and getattr(workflow, "neo4j", None):
                        dropped = workflow.neo4j.drop_graph(age_graph_name_for_cleanup)
                    else:
                        nc_drop = Neo4jClient(graph_name=age_graph_name_for_cleanup)
                        try:
                            dropped = nc_drop.drop_graph(age_graph_name_for_cleanup)
                        finally:
                            try:
                                nc_drop.close()
                            except Exception:
                                pass
                    if dropped:
                        logger.info(f"[CLEANUP] AGE graph dropped: {age_graph_name_for_cleanup}")
                    else:
                        logger.info(
                            f"[CLEANUP] AGE graph not dropped (missing or skipped): {age_graph_name_for_cleanup}"
                        )
            except Exception as e:
                logger.warning(f"[WARN] AGE graph drop 실패: {e}")
            
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
