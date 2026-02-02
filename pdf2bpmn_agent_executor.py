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
from typing import Any, Dict, Optional, List, Set, Tuple
import traceback
import xml.etree.ElementTree as ET
import sys
from html.parser import HTMLParser
from urllib.parse import urlparse, urlunparse

from src.pdf2bpmn.processgpt.bpmn_xml_generator import ProcessGPTBPMNXmlGenerator
from src.pdf2bpmn.processgpt.process_definition_prompt import build_system_prompt_processgpt
from src.pdf2bpmn.processgpt.process_consulting_prompt import get_process_consulting_system_prompt
from src.pdf2bpmn.processgpt.process_generation_messages import build_process_definition_messages

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
        
        # Supabase 설정
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SERVICE_ROLE_KEY')
        self.supabase_client: Optional[Client] = None

        # OpenAI client (for form generation)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        # Default to gpt-4.1 for longer, more stable structured outputs.
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        # Separate models (requested: user/agent creation vs process creation)
        self.user_mapping_model = os.getenv("USER_MAPPING_MODEL", self.openai_model)
        self.process_definition_model = os.getenv("PROCESS_DEF_MODEL", self.openai_model)
        self.openai_client: Optional[OpenAI] = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
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
        if total > max_forms:
            activities = activities[:max_forms]
            total = len(activities)

        for idx, a in enumerate(activities):
            if not isinstance(a, dict):
                continue

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

            html = ""
            # 1) LLM 시도
            try:
                per_form_timeout = float(os.getenv("FORM_LLM_TIMEOUT_SEC", "120"))
                html = await asyncio.wait_for(self._call_openai_for_form_html(request_text), timeout=per_form_timeout)
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
            if ok:
                forms_saved += 1

            # Always keep an index for post-processing (even if save failed).
            forms_by_activity_id[activity_id] = {
                "form_id": form_def_id,
                "fields_json": fields_json,
            }

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
        - Ensure agent fields are consistent (agentMode/orchestration)
        - Set inputData using real form_id + fields_json from earlier tasks
        """
        # 0) Final-stage assignee mapping:
        # - Do it AFTER forms exist and process is enriched, so we can use:
        #   (a) generated process info (activities/roles/tools)
        #   (b) extracted info (from PDF/Neo4j) if provided
        #   (c) organization chart + agent profiles
        try:
            await self._apply_assignment_and_maybe_create_agents(
                proc_json=proc_json,
                tenant_id=tenant_id,
                process_name=process_name,
                extracted=extracted,
            )
        except Exception as e:
            logger.warning(f"[WARN] assignment apply failed in expand stage: {e}")

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
                agent_id = str(a.get("agent") or "").strip()
                if agent_id:
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
            "llm_used": bool(isinstance(chosen, dict)),
        }

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

        elems = em.get("elements")
        if not isinstance(elems, list):
            # The generator can accept dict-shaped elements too, but this backend path uses list.
            return em

        for e in elems:
            if not isinstance(e, dict):
                continue
            if e.get("elementType") != "Activity":
                continue
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
                    "customProperties": a.get("customProperties") or [],
                }
            )
            e["properties"] = props

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

        # Docker 환경에서 로컬 Supabase(Storage) URL이 localhost/127.0.0.1로 들어오는 경우 보정
        # - 컨테이너 내부에서 127.0.0.1은 컨테이너 자신이므로, 호스트의 Supabase에 접근하려면 host.docker.internal로 바꿔야 함
        if result.get("pdf_url") and self._is_running_in_docker():
            rewritten = self._rewrite_localhost_url(result["pdf_url"], localhost_target="host.docker.internal")
            if rewritten != result["pdf_url"]:
                logger.info(f"[PARSE] Rewrote pdf_url for Docker: {result['pdf_url']} -> {rewritten}")
                result["pdf_url"] = rewritten
        
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

    async def _download_pdf(self, url: str, filename: str = None) -> Tuple[str, str, Optional[str]]:
        """(deprecated) 파일 다운로드 후 (임시 경로, 추정 파일명, content-type) 반환"""
        return await self._download_file(url, filename)

    async def _download_file(self, url: str, filename: str = None) -> Tuple[str, str, Optional[str]]:
        """
        파일 다운로드 후 (임시 파일 경로, 추정 파일명, content-type)을 반환합니다.
        - docx2pdf 프로젝트처럼 Content-Disposition / URL / Content-Type으로 파일명/확장자를 최대한 추정합니다.
        - 확장자가 불명확하면 `.bin`으로 저장합니다(이 경우 변환이 실패할 수 있음).
        """
        client = await self._get_http_client()

        download_url = url
        if self._is_running_in_docker():
            download_url = self._rewrite_localhost_url(download_url, localhost_target="host.docker.internal")
            if download_url != url:
                logger.info(f"[DOWNLOAD] Rewrote URL for Docker: {url} -> {download_url}")

        logger.info(f"[DOWNLOAD] Downloading file from: {download_url}")

        try:
            response = await client.get(download_url, follow_redirects=True)
        except httpx.ConnectError as e:
            # ConnectError는 "PDF2BPMN API(8001)" 뿐 아니라 "첨부파일 다운로드 URL"에서도 발생할 수 있음
            raise Exception(f"파일 다운로드 연결 실패: {download_url} ({e})")
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")

        def _sanitize_filename(name: str) -> str:
            name = (name or "").replace("\\", "/").split("/")[-1]
            name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" ._")
            name = re.sub(r"\s+", " ", name).strip()
            return name[:150] if name else ""

        def _guess_filename_from_headers() -> str:
            cd = response.headers.get("content-disposition") or response.headers.get("Content-Disposition") or ""
            if cd:
                m = re.search(r"filename\*=UTF-8''([^;]+)", cd, flags=re.IGNORECASE)
                if m:
                    try:
                        from urllib.parse import unquote

                        v = _sanitize_filename(unquote(m.group(1)))
                        if v:
                            return v
                    except Exception:
                        pass
                m = re.search(r'filename="([^"]+)"', cd, flags=re.IGNORECASE)
                if m:
                    v = _sanitize_filename(m.group(1))
                    if v:
                        return v
                m = re.search(r"filename=([^;]+)", cd, flags=re.IGNORECASE)
                if m:
                    v = _sanitize_filename(m.group(1).strip().strip('"'))
                    if v:
                        return v

            try:
                from urllib.parse import urlparse, unquote

                parsed = urlparse(str(response.url))
                base = _sanitize_filename(unquote(Path(parsed.path).name))
                if base:
                    return base
            except Exception:
                pass

            return ""

        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower() or None
        content_type_to_ext = {
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "text/plain": ".txt",
            "text/csv": ".csv",
            "text/html": ".html",
            "application/rtf": ".rtf",
            "application/vnd.oasis.opendocument.text": ".odt",
            "application/vnd.oasis.opendocument.spreadsheet": ".ods",
            "application/vnd.oasis.opendocument.presentation": ".odp",
        }

        inferred_name = _sanitize_filename(filename) if filename else ""
        if not inferred_name:
            inferred_name = _guess_filename_from_headers() or "input"

        ext = Path(inferred_name).suffix.lower()
        body_head = response.content[:6] if response.content else b""
        is_pdf_by_magic = body_head.startswith(b"%PDF-")
        if (not ext) or (ext == ".pdf" and (not is_pdf_by_magic) and content_type and content_type != "application/pdf"):
            mapped = content_type_to_ext.get(content_type or "")
            if mapped:
                inferred_name = str(Path(inferred_name).with_suffix(mapped))
                ext = mapped

        suffix = ext or ".bin"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()

        logger.info(f"[DOWNLOAD] File saved to: {temp_file.name} (inferred={inferred_name}, ct={content_type})")
        return temp_file.name, inferred_name, content_type

    def _normalize_text_key(self, s: str) -> str:
        return re.sub(r"\s+", "", (s or "").strip().lower())

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

            resp = await asyncio.to_thread(_run)
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return None
            # strip code fences if present
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if m:
                content = m.group(1).strip()
            return json.loads(content)
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

    async def _call_openai_process_definition(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int = 3500,
    ) -> Optional[Dict[str, Any]]:
        """프로세스 정의 생성용: JSON-only 출력 강제 + 파싱 실패 시 재시도."""
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
        temperature = float(
            os.getenv(
                "LLM_PROCESS_DEFINITION_TEMPERATURE",
                os.getenv("LLM_PROCESS_TEMPERATURE", "0.0"),
            )
        )

        # Retry a few times because a single invalid character breaks JSON parsing.
        for attempt in range(1, 4):
            attempt_messages = messages
            if attempt > 1:
                attempt_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "이전 응답은 JSON 파싱에 실패했습니다.\n"
                            "이번에는 **단 하나의 JSON 객체만** 출력하세요.\n"
                            "- 마크다운/설명/코드블록/백틱/주석 금지\n"
                            "- JSON 외 텍스트가 1글자라도 있으면 실패\n"
                        ),
                    }
                ]
            try:
                def _run():
                    # Prefer JSON mode when supported; fallback gracefully if SDK/model doesn't support it.
                    try:
                        return self.openai_client.chat.completions.create(
                            model=self.process_definition_model,
                            messages=attempt_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                        )
                    except TypeError:
                        return self.openai_client.chat.completions.create(
                            model=self.process_definition_model,
                            messages=attempt_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                resp = await asyncio.to_thread(_run)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    logger.warning(
                        f"[PROCDEF][LLM] empty content returned (attempt={attempt}/3, model={self.process_definition_model})"
                    )
                    continue

                # First try: content itself is JSON (when response_format=json_object worked)
                try:
                    parsed = _loads_with_newlines_removed(content)
                    if isinstance(parsed, dict):
                        elems = parsed.get("elements")
                        elems_len = len(elems) if isinstance(elems, list) else None
                        logger.info(
                            f"[PROCDEF][LLM] parsed ok (attempt={attempt}/3, keys={list(parsed.keys())}, elements_len={elems_len})"
                        )
                    else:
                        logger.warning(
                            f"[PROCDEF][LLM] parsed non-dict JSON (attempt={attempt}/3, type={type(parsed).__name__})"
                        )
                    return parsed
                except json.JSONDecodeError:
                    # Fallback: attempt to recover JSON from markdown/codefence responses
                    json_block = self._extract_json_block_from_markdown(content) or ""
                    if not json_block:
                        logger.warning(
                            "[PROCDEF][LLM] JSON decode failed and no JSON block recovered "
                            f"(attempt={attempt}/3, content_preview={content[:300]!r})"
                        )
                        continue
                    try:
                        parsed = _loads_with_newlines_removed(json_block)
                        if isinstance(parsed, dict):
                            elems = parsed.get("elements")
                            elems_len = len(elems) if isinstance(elems, list) else None
                            logger.info(
                                f"[PROCDEF][LLM] parsed ok from recovered block (attempt={attempt}/3, keys={list(parsed.keys())}, elements_len={elems_len})"
                            )
                        else:
                            logger.warning(
                                f"[PROCDEF][LLM] parsed non-dict JSON from recovered block (attempt={attempt}/3, type={type(parsed).__name__})"
                            )
                        return parsed
                    except json.JSONDecodeError:
                        logger.warning(
                            "[PROCDEF][LLM] JSON decode failed even after block recovery "
                            f"(attempt={attempt}/3, block_preview={json_block[:300]!r}, content_preview={content[:300]!r})"
                        )
                        continue
            except Exception as e:
                logger.warning(f"[WARN] OpenAI process-definition call failed (attempt {attempt}/3): {e}")
                continue

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

            resp = await asyncio.to_thread(_run)
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return None
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if m:
                content = m.group(1).strip()
            return json.loads(content)
        except Exception as e:
            logger.warning(f"[WARN] OpenAI JSON(messages) call failed: {e}")
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

        # Connectivity repair: ensure every consecutive node is connected (fallback chain)
        for i in range(len(node_order) - 1):
            s = node_order[i]
            t = node_order[i + 1]
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

        # Gateway branching: ensure conditions exist when a gateway has multiple outgoing.
        # Also: remove degenerate gateways (<=1 outgoing) by collapsing them into straight sequences.
        outgoing_by_source: Dict[str, List[Dict[str, Any]]] = {}
        incoming_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for s in seqs:
            outgoing_by_source.setdefault(str(s.get("source")), []).append(s)
            incoming_by_target.setdefault(str(s.get("target")), []).append(s)

        gateway_ids = {str(e.get("id")) for e in elems if e.get("elementType") == "Gateway" and e.get("id")}

        removed_gateway_ids: Set[str] = set()
        if gateway_ids:
            # collapse single-branch gateways: connect incoming.source -> outgoing.target and remove gateway node/sequences
            for gid in list(gateway_ids):
                outs = outgoing_by_source.get(gid) or []
                if len(outs) >= 2:
                    continue
                ins = incoming_by_target.get(gid) or []
                out_target = str(outs[0].get("target")) if outs else ""
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
            for j, s in enumerate(outs, start=1):
                cond = str(s.get("condition") or "").strip()
                if not cond:
                    s["condition"] = f"조건 {j}"

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

    async def _generate_processgpt_definition_and_bpmn(
        self,
        *,
        tenant_id: str,
        process_name: str,
        extracted: Dict[str, Any],
        user_request: str,
    ) -> Optional[Dict[str, Any]]:
        """
        (프로세스 생성 LLM) 단계:
        - 프론트 ProcessDefinitionGenerator 프롬프트와 동일한 규칙을 사용해 elements 모델 생성
        - proc_def.definition(런타임) 구조로 변환하여 함께 반환

        IMPORTANT:
        - BPMN XML은 폼 생성/참조정보(inputData) 확장 이후에 최종값으로 생성/저장해야 합니다.
          (초기 생성 후 확장 단계에서 tool/form id 등이 변경되므로, XML을 먼저 만들면 stale 됩니다.)
        """
        if not self.openai_client:
            return None

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

        # 1) (프롬프트 개선) ProcessConsultingGenerator 프롬프트로 "말로 된 프로세스 초안"을 먼저 생성
        # NOTE: 담당자 매핑은 "마지막 확장 단계(after forms)"에서 수행한다.
        consulting_outline = await self._generate_process_outline_via_consulting_prompt(
            process_name=process_name,
            user_request=user_request,
            extracted=extracted,
            hints_simplified={},
        )

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

        elements_model = await self._call_openai_process_definition(messages=messages)
        if not isinstance(elements_model, dict):
            logger.warning(f"[PROCDEF][LLM] elements_model is not dict -> generation failed (process={process_name!r})")
            return None

        # LLM raw shape summary
        try:
            elems = elements_model.get("elements")
            elems_len = len(elems) if isinstance(elems, list) else None
            logger.info(
                f"[PROCDEF][LLM] elements_model received: keys={list(elements_model.keys())} elements_len={elems_len} "
                f"(process={process_name!r})"
            )
        except Exception:
            pass

        # 5) Strict validate/normalize elements model (connectivity + ids + required fields)
        elements_model = self._validate_and_normalize_elements_model(elements_model, process_name=process_name)

        # 5.5) Force proc_def id to UUID (avoid collisions on save)
        # NOTE:
        # - We intentionally IGNORE model-provided processDefinitionId to prevent accidental reuse.
        # - BPMN XML generator does not use this id for <bpmn:process id="..."> (it uses Process_1),
        #   so UUID starting with digits is safe.
        forced_proc_def_id = str(uuid.uuid4())
        elements_model["processDefinitionId"] = forced_proc_def_id

        # 6) Convert to runtime definition + enrich + assignment(again, as safety)
        runtime_def = self._elements_model_to_runtime_definition(elements_model)
        runtime_def = self._enrich_process_definition(
            runtime_def,
            process_name=str(runtime_def.get("processDefinitionName") or process_name),
            process_definition_id=str(elements_model.get("processDefinitionId") or runtime_def.get("processDefinitionId")),
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
        if not isinstance(obj.get("decisions"), list):
            return None
        return obj

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
        """
        proc_json.roles / proc_json.activities에 대해:
        - 기존 룰 기반 매핑(사용자/에이전트/팀)
        - LLM 기반 추천(선택)
        - 필요 시 에이전트 자동 생성(users insert) + 조직도 반영(선택)
        """
        await self._load_org_and_agents(tenant_id)
        roles = proc_json.get("roles") or []
        activities = proc_json.get("activities") or []
        if not isinstance(roles, list):
            roles = []
        if not isinstance(activities, list):
            activities = []

        # -------------------------------------------------------------------
        # NEW (요구사항): 프로세스 "단건" 맥락에서 자동화 가능 부분을 판단하고
        # - 기존 에이전트 매핑
        # - 없으면 생성(create_agent)까지
        # 를 한 번에 계획/적용한다. (키워드 기반 human_required는 폴백으로만 남김)
        # -------------------------------------------------------------------
        plan = await self._llm_plan_assignments_for_process(
            tenant_id=tenant_id,
            process_name=process_name,
            proc_json=proc_json,
            extracted_context=extracted,
        )

        users_by_id = {str(u.get("id")): u for u in (self._users or []) if isinstance(u, dict) and u.get("id")}

        def _clean_text(s: Any) -> str:
            return " ".join(str(s or "").split())

        def _is_agent_user_id(user_id: str) -> bool:
            u = users_by_id.get(str(user_id) or "")
            return bool(u and u.get("is_agent") is True and u.get("id"))

        default_agent_mode = "draft"  # fixed

        if isinstance(plan, dict) and isinstance(plan.get("decisions"), list):
            decisions = plan.get("decisions") or []
            by_aid: Dict[str, Dict[str, Any]] = {}
            for d in decisions:
                if isinstance(d, dict):
                    aid = str(d.get("activity_id") or "").strip()
                    if aid:
                        by_aid[aid] = d

            for a in activities:
                if not isinstance(a, dict):
                    continue
                aid = _clean_text(a.get("id"))
                aname = _clean_text(a.get("name"))
                rn = _clean_text(a.get("role"))

                d = by_aid.get(aid)
                if not isinstance(d, dict):
                    # no decision => leave unassigned (fallback later)
                    continue

                action = str(d.get("action") or "none").strip()
                conf = d.get("confidence")
                reason = str(d.get("reason") or "")[:200]

                if action == "existing_user":
                    target_user_id = str(d.get("target_user_id") or "").strip()
                    if target_user_id and _is_agent_user_id(target_user_id):
                        a["agent"] = target_user_id
                        a["agentMode"] = default_agent_mode
                        a["orchestration"] = "crewai-action"
                        logger.info(
                            f"[ASSIGN][PLAN] activity id={aid!r} name={aname!r} role={rn!r} -> existing_user id={target_user_id!r} conf={conf} reason={reason!r}"
                        )
                        continue
                    # invalid => treat as none
                    action = "none"

                if action == "create_agent":
                    create_agent = d.get("create_agent") or {}
                    if not isinstance(create_agent, dict):
                        create_agent = {}
                    team_id_for_new = str(create_agent.get("team_id") or "").strip()
                    team_name = self._org_team_name_by_id.get(team_id_for_new) or ""
                    if not team_name:
                        team_name = "미분류"
                    user_input = str(create_agent.get("user_input") or "").strip()
                    if not user_input:
                        user_input = (
                            f"다음 태스크를 자동화할 에이전트를 생성해주세요.\n"
                            f"- 프로세스: {process_name}\n"
                            f"- 역할: {rn}\n"
                            f"- 태스크: {aname}\n"
                            f"- 지침: {_clean_text(a.get('instruction'))}\n"
                            "자동화 기준/입력/출력/검증 방법을 포함해 설계해 주세요."
                        )
                    mcp_tools = self._safe_json_loads(os.getenv("MCP_TOOLS_JSON", "")) or {}
                    agent_profile = await self._llm_generate_agent_profile(team_name=team_name, user_input=user_input, mcp_tools=mcp_tools)
                    if agent_profile:
                        new_agent_type = str(create_agent.get("agent_type") or "agent").strip() or "agent"
                        created = await self._insert_agent_user(tenant_id=tenant_id, agent_profile=agent_profile, agent_type=new_agent_type)
                        if created and created.get("id"):
                            if team_id_for_new:
                                await self._update_org_chart_add_member(
                                    tenant_id=tenant_id, team_id=team_id_for_new, member_user=created
                                )
                            a["agent"] = created.get("id")
                            a["agentMode"] = default_agent_mode
                            a["orchestration"] = "crewai-action"
                            logger.info(
                                f"[ASSIGN][PLAN] activity id={aid!r} name={aname!r} role={rn!r} -> create_agent id={created.get('id')!r} conf={conf} reason={reason!r}"
                            )
                            continue
                    # creation failed => none
                    action = "none"

                # none/default
                a["agent"] = None
                a["agentMode"] = "none"
                a["orchestration"] = None
                logger.info(
                    f"[ASSIGN][PLAN] activity id={aid!r} name={aname!r} role={rn!r} -> none conf={conf} reason={reason!r}"
                )

            proc_json["activities"] = activities
            return

        # -------------------------------------------------------------------
        # Fallback (legacy): per-activity heuristics + LLM
        # -------------------------------------------------------------------
        # Build per-role activity context (LLM 입력으로 활용)
        role_ctx: Dict[str, List[Dict[str, Any]]] = {}
        for a in activities:
            if not isinstance(a, dict):
                continue
            rn = str(a.get("role") or "").strip()
            if not rn:
                continue
            role_ctx.setdefault(rn, [])
            role_ctx[rn].append(
                {
                    "activityId": str(a.get("id") or ""),
                    "activityName": str(a.get("name") or ""),
                    "instruction": str(a.get("instruction") or ""),
                    "description": str(a.get("description") or ""),
                    "tool": str(a.get("tool") or ""),
                }
            )

        # -------------------------------------------------------------------
        # Activity-based assignment (요구사항):
        # - roles.endpoint/default는 '기준'으로 쓰지 않고, activities를 기준으로 agent를 채운다.
        # - 역할명이 추상적이거나(신청자 등) 역할만으로는 판단이 어려운 경우에도
        #   activity name/instruction/tool 컨텍스트로 매핑/생성을 시도한다.
        # -------------------------------------------------------------------
        # users_by_id/_clean_text/_is_agent_user_id/default_agent_mode already defined above

        # 사람이 직접 수행해야 하는 태스크를 매우 보수적으로 걸러서
        # "모든 태스크에 에이전트가 붙는" 현상을 방지한다.
        # (LLM 프롬프트에도 동일 규칙이 있지만, 휴리스틱이 먼저 붙이는 케이스를 막기 위함)
        _HUMAN_REQUIRED_KWS = [
            "신청", "등록", "접수", "제출", "결재", "결제", "입금", "납부", "승인", "서명",
            "대면", "회의", "면담", "전화", "방문", "출석", "참석", "수령", "발급", "실물", "현장",
        ]
        _AUTOMATION_HINT_KWS = [
            # NOTE: "작성"은 신청/제출 같은 인간업무에도 자주 등장하므로 자동화 힌트로 쓰지 않는다.
            "자동", "에이전트", "봇", "생성", "요약", "정리", "분석", "검증", "추출", "조회", "검색",
            "분류", "추천", "채점", "퀴즈", "문항", "리포트", "보고서", "취합", "집계",
        ]

        def _looks_strongly_human_required(activity: Dict[str, Any]) -> bool:
            text = f"{activity.get('name') or ''} {activity.get('instruction') or ''} {activity.get('description') or ''}"
            key = self._normalize_text_key(text)
            if not key:
                return False
            # 자동화 힌트가 있으면 사람 업무로 단정하지 않는다(예: '신청서 자동 작성')
            for kw in _AUTOMATION_HINT_KWS:
                if self._normalize_text_key(kw) in key:
                    return False
            return any(self._normalize_text_key(kw) in key for kw in _HUMAN_REQUIRED_KWS)

        def _heuristic_pick_agent(activity: Dict[str, Any], *, allow_on_human_required: bool = False) -> Optional[Dict[str, Any]]:
            human_required = _looks_strongly_human_required(activity)
            if human_required and not allow_on_human_required:
                return None
            # IMPORTANT:
            # - human_required 태스크에서는 role명(팀/신청자 등)으로 매칭하면 오탐이 많으므로 role 기반 매칭을 건너뛴다.
            if not human_required:
                rn = _clean_text(activity.get("role"))
                if rn:
                    hit = self._pick_user_for_role(rn)
                    if hit and hit.get("is_agent") is True:
                        return hit
            # match by activity name/instruction/description keywords vs agent username/role/alias
            key = self._normalize_text_key(f"{activity.get('name') or ''} {activity.get('instruction') or ''} {activity.get('description') or ''}")
            if not key:
                return None
            for a in (self._agents or []):
                if not isinstance(a, dict) or not a.get("id"):
                    continue
                for field in ("username", "role", "alias"):
                    cand = self._normalize_text_key(a.get(field)) or ""
                    if cand and (cand in key or key in cand):
                        return a
            return None

        default_agent_mode = "draft"  # fixed
        for a in activities:
            if not isinstance(a, dict):
                continue
            aid = _clean_text(a.get("id"))
            aname = _clean_text(a.get("name"))
            rn = _clean_text(a.get("role"))

            # already assigned and valid
            if a.get("agent") and _is_agent_user_id(str(a.get("agent"))):
                a["agentMode"] = default_agent_mode
                a["orchestration"] = "crewai-action"
                continue

            human_required = _looks_strongly_human_required(a)
            # human_required: 신규 생성은 금지하되, 기존에 조직도/users에 있는 적절한 에이전트는 찾아서 매핑할 수 있어야 한다.
            if human_required:
                hit = _heuristic_pick_agent(a, allow_on_human_required=True)
                if hit and hit.get("id"):
                    a["agent"] = hit.get("id")
                    a["agentMode"] = default_agent_mode
                    a["orchestration"] = "crewai-action"
                    logger.info(f"[ASSIGN] activity id={aid!r} name={aname!r} role={rn!r} -> existing_agent(human_required) id={hit.get('id')}")
                    continue
                # LLM may still find a better existing agent, but MUST NOT create a new one for human_required tasks.
                activity_ctx = [
                    {
                        "activityId": aid,
                        "activityName": aname,
                        "role": rn,
                        "instruction": _clean_text(a.get("instruction")),
                        "description": _clean_text(a.get("description")),
                        "tool": _clean_text(a.get("tool")),
                    }
                ]
                role_query = _clean_text(f"{rn} {aname} {a.get('instruction') or ''} {a.get('description') or ''}") or aid
                rec = await self._llm_recommend_assignee(
                    tenant_id=tenant_id,
                    process_name=process_name,
                    role_name=role_query,
                    activities_context=activity_ctx,
                    extracted_context=extracted,
                    allow_create_agent=False,
                )
                if isinstance(rec, dict) and str(rec.get("action") or "") == "existing_user":
                    target_user_id = str(rec.get("target_user_id") or "").strip()
                    if target_user_id and _is_agent_user_id(target_user_id):
                        a["agent"] = target_user_id
                        a["agentMode"] = default_agent_mode
                        a["orchestration"] = "crewai-action"
                        logger.info(f"[ASSIGN] activity id={aid!r} name={aname!r} role={rn!r} -> existing_user(human_required) id={target_user_id}")
                        continue
                # default: keep unassigned (no creation)
                a["agent"] = None
                a["agentMode"] = "none"
                a["orchestration"] = None
                logger.info(f"[ASSIGN] activity id={aid!r} name={aname!r} role={rn!r} -> human_required => no agent (creation disabled)")
                continue

            # heuristic pick first
            hit = _heuristic_pick_agent(a, allow_on_human_required=False)
            if hit and hit.get("id"):
                a["agent"] = hit.get("id")
                a["agentMode"] = default_agent_mode
                a["orchestration"] = "crewai-action"
                logger.info(f"[ASSIGN] activity id={aid!r} name={aname!r} role={rn!r} -> existing_agent id={hit.get('id')}")
                continue

            # LLM-based for this specific activity (more context than role-only)
            activity_ctx = [
                {
                    "activityId": aid,
                    "activityName": aname,
                    "role": rn,
                    "instruction": _clean_text(a.get("instruction")),
                    "description": _clean_text(a.get("description")),
                    "tool": _clean_text(a.get("tool")),
                }
            ]
            # IMPORTANT: do not use role name only. Include activity name/instruction so that
            # candidate filtering can find agents like "인터뷰 검증 에이전트".
            role_query = _clean_text(f"{rn} {aname} {a.get('instruction') or ''} {a.get('description') or ''}") or aid
            rec = await self._llm_recommend_assignee(
                tenant_id=tenant_id,
                process_name=process_name,
                role_name=role_query,
                activities_context=activity_ctx,
                extracted_context=extracted,
                allow_create_agent=True,
            )
            if not isinstance(rec, dict):
                a["agent"] = None
                a["agentMode"] = "none"
                a["orchestration"] = None
                continue

            action = str(rec.get("action") or "none").strip()
            target_uid_dbg = str(rec.get("target_user_id") or "").strip()
            target_tid_dbg = str(rec.get("target_team_id") or "").strip()
            logger.info(
                f"[ASSIGN] activity id={aid!r} name={aname!r} role={rn!r} "
                f"LLM action={action} target_user_id={target_uid_dbg!r} target_team_id={target_tid_dbg!r} "
                f"conf={rec.get('confidence')} reason={str(rec.get('reason') or '')[:120]!r}"
            )

            if action == "existing_user":
                target_user_id = str(rec.get("target_user_id") or "").strip()
                if target_user_id and _is_agent_user_id(target_user_id):
                    a["agent"] = target_user_id
                    a["agentMode"] = default_agent_mode
                    a["orchestration"] = "crewai-action"
                    continue

            if action == "create_agent":
                create_agent = rec.get("create_agent") or {}
                if not isinstance(create_agent, dict):
                    create_agent = {}
                team_id_for_new = str(create_agent.get("team_id") or rec.get("target_team_id") or "").strip()
                team_name = self._org_team_name_by_id.get(team_id_for_new) or ""
                if not team_name:
                    team_name = "미분류"
                user_input = str(create_agent.get("user_input") or "").strip()
                if not user_input:
                    user_input = (
                        f"다음 태스크를 자동화할 에이전트를 생성해주세요.\n"
                        f"- 역할: {rn}\n"
                        f"- 태스크: {aname}\n"
                        f"- 지침: {_clean_text(a.get('instruction'))}\n"
                        "필요 시 사용자에게 확인을 요청하고 결과를 정리해 주세요."
                    )
                mcp_tools = self._safe_json_loads(os.getenv("MCP_TOOLS_JSON", "")) or {}
                agent_profile = await self._llm_generate_agent_profile(team_name=team_name, user_input=user_input, mcp_tools=mcp_tools)
                if agent_profile:
                    new_agent_type = str(create_agent.get("agent_type") or "agent").strip() or "agent"
                    created = await self._insert_agent_user(tenant_id=tenant_id, agent_profile=agent_profile, agent_type=new_agent_type)
                    if created and created.get("id"):
                        if team_id_for_new:
                            await self._update_org_chart_add_member(tenant_id=tenant_id, team_id=team_id_for_new, member_user=created)
                        a["agent"] = created.get("id")
                        a["agentMode"] = default_agent_mode
                        a["orchestration"] = "crewai-action"
                        continue

            # default: no agent
            a["agent"] = None
            a["agentMode"] = "none"
            a["orchestration"] = None

        proc_json["activities"] = activities

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
            a.setdefault("instruction", a.get("instruction") or a.get("description") or "")
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
        
        temp_download_path: Optional[str] = None
        temp_pdf_path: Optional[str] = None
        
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
            
            # =================================================================
            # 4. PDF URL 다운로드 및 (필요 시) PDF 변환
            #    - 내부 FastAPI(/api/*) 호출은 제거하고, 워크플로우를 직접 실행합니다.
            # =================================================================
            pdf_url = parsed.get("pdf_url", "")
            if not pdf_url:
                raise Exception("PDF URL이 제공되지 않았습니다. query에 pdf_url을 포함해주세요.")

            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                f"[DOWNLOAD] 파일 다운로드 중: {pdf_name}",
                "tool_usage_started", 8
            )

            # 파일 다운로드(헤더/URL/Content-Type 기반 파일명 추정 포함)
            temp_download_path, inferred_name, inferred_ct = await self._download_file(pdf_url, pdf_name)
            pdf_name = inferred_name or pdf_name

            # PDF 여부 판별(확장자보다 매직바이트 우선)
            is_pdf = False
            try:
                with open(temp_download_path, "rb") as _f:
                    is_pdf = (_f.read(6) or b"").startswith(b"%PDF-")
            except Exception:
                is_pdf = False

            pdf_path_for_workflow = temp_download_path

            # PDF가 아니면 로컬에서 PDF로 변환
            if not is_pdf:
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    f"[CONVERT] PDF가 아닌 파일을 PDF로 변환 중: {pdf_name}",
                    "tool_usage_started", 9
                )

                # 확장자가 ".pdf"인데 실제 PDF가 아닌 경우(잘못된 힌트/URL) → Content-Type/파일명으로 보정 후 변환
                try:
                    src_p = Path(temp_download_path)
                    name_ext = Path(pdf_name).suffix.lower()
                    if src_p.suffix.lower() == ".pdf":
                        fixed_ext = ""
                        if name_ext and name_ext != ".pdf":
                            fixed_ext = name_ext
                        else:
                            ct_map = {
                                "application/msword": ".doc",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                                "application/vnd.ms-excel": ".xls",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                                "application/vnd.ms-powerpoint": ".ppt",
                                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                                "text/plain": ".txt",
                                "text/csv": ".csv",
                                "text/html": ".html",
                                "application/rtf": ".rtf",
                                "application/vnd.oasis.opendocument.text": ".odt",
                                "application/vnd.oasis.opendocument.spreadsheet": ".ods",
                                "application/vnd.oasis.opendocument.presentation": ".odp",
                            }
                            fixed_ext = ct_map.get((inferred_ct or "").lower(), "")
                        if fixed_ext and fixed_ext != ".pdf":
                            new_path = str(src_p.with_suffix(fixed_ext))
                            os.replace(temp_download_path, new_path)
                            temp_download_path = new_path
                            pdf_name = str(Path(pdf_name).with_suffix(fixed_ext))
                except Exception:
                    pass

                try:
                    from src.pdf2bpmn.converters.file_to_pdf import convert_to_pdf, FileToPdfError  # type: ignore

                    converted_pdf = convert_to_pdf(str(temp_download_path), str(Path(str(temp_download_path)).parent))
                    pdf_path_for_workflow = converted_pdf
                    temp_pdf_path = converted_pdf
                    pdf_name = Path(converted_pdf).name
                except FileToPdfError as e:
                    raise Exception(f"파일을 PDF로 변환하지 못했습니다: {e}")
                finally:
                    # 다운로드 원본은 가능하면 정리(변환 산출물과 경로가 다르면)
                    try:
                        if temp_download_path and temp_pdf_path and os.path.abspath(temp_download_path) != os.path.abspath(temp_pdf_path):
                            if os.path.exists(temp_download_path):
                                os.unlink(temp_download_path)
                    except Exception:
                        pass
            else:
                # PDF인데 확장자가 PDF가 아니면 표시용 이름만 보정
                if not str(pdf_name).lower().endswith(".pdf"):
                    pdf_name = str(Path(pdf_name).with_suffix(".pdf"))

            # =================================================================
            # 5. PDF2BPMN 워크플로우를 "직접 호출"로 실행 (FastAPI BackgroundTasks 제거)
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
                event_data = {
                    "message": msg,
                    "status": "tool_usage_started",
                    "progress": int(progress),
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

            workflow = PDF2BPMNWorkflow()
            state: Dict[str, Any] = {
                "pdf_paths": [str(pdf_path_for_workflow)],
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
                "open_questions": [],
                "resolved_questions": [],
                "current_question": None,
                "user_answer": None,
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

                # Step 1: ingest_pdf
                _enqueue_progress("[STEP] PDF 파싱 중...", 20)
                state.update(await asyncio.to_thread(workflow.ingest_pdf, state))
                page_count = 0
                try:
                    docs = state.get("documents") or []
                    if docs:
                        page_count = int(getattr(docs[0], "page_count", 0) or 0)
                except Exception:
                    page_count = 0
                chunk_count = len(state.get("reference_chunks") or [])
                _enqueue_progress(f"[STEP] PDF 파싱 완료: {page_count}페이지, {chunk_count}개 청크", 28)

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

                # Step 5: generate_skills
                _enqueue_progress("[STEP] Agent Skill 문서 생성 중...", 74)
                state.update(await asyncio.to_thread(workflow.generate_skills, state))
                _enqueue_progress("[STEP] Agent Skill 문서 생성 완료", 80)

                if self.is_cancelled:
                    raise Exception("작업이 취소되었습니다.")

                # Step 6: generate_dmn
                _enqueue_progress("[STEP] DMN 의사결정 테이블 생성 중...", 84)
                state.update(await asyncio.to_thread(workflow.generate_dmn, state))
                _enqueue_progress("[STEP] DMN 생성 완료", 88)

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
            # 6. 이번 작업에서 생성된 process_id 목록을 state에서 직접 수집 + Neo4j에서 상세 조회
            # =================================================================
            await self._send_progress_event(
                event_queue, context_id, task_id, job_id,
                "[GENERATING] 이번 작업의 추출 정보(Neo4j)로 ProcessGPT 프로세스 정의/유저 매핑을 생성합니다...",
                "tool_usage_started", 88
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
                from src.pdf2bpmn.graph.neo4j_client import Neo4jClient  # type: ignore

                neo4j = Neo4jClient()
                try:
                    for proc_id in job_process_ids:
                        try:
                            detail = await asyncio.to_thread(neo4j.get_process_with_details, proc_id)
                            if not detail:
                                continue
                            flows = await asyncio.to_thread(neo4j.get_sequence_flows, proc_id)
                            if isinstance(flows, list):
                                detail["sequence_flows"] = flows
                            extracted_by_proc_id[proc_id] = {
                                "detail": detail,
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
            
            # 9. 각 추출 프로세스에 대해 ProcessGPT 정의/유저매핑 → XML 생성 → DB 저장
            saved_processes = []
            all_bpmn_xmls = {}  # proc_def_id -> bpmn_xml 매핑
            total_bpmn = len(extracted_by_proc_id)
            
            logger.info(f"[DEBUG] extracted_by_proc_id keys: {list(extracted_by_proc_id.keys())}")
            
            for idx, (proc_id, pinfo) in enumerate(extracted_by_proc_id.items()):
                process_name = pinfo.get("process_name") or f"Process {idx + 1}"
                detail = pinfo.get("detail") or {}
                
                logger.info(f"[DEBUG] Processing extracted process {idx+1}/{total_bpmn}: {process_name}")

                # extracted info -> ProcessGPT definition + BPMN XML
                extracted_payload = {
                    "process": detail.get("process") or {},
                    "tasks": detail.get("tasks") or [],
                    "roles": detail.get("roles") or [],
                    "gateways": detail.get("gateways") or [],
                    "events": detail.get("events") or [],
                    "sequence_flows": detail.get("sequence_flows") or detail.get("flows") or [],
                }

                # legacy flow is intentionally removed

                generated = await self._generate_processgpt_definition_and_bpmn(
                    tenant_id=tenant_id,
                    process_name=process_name,
                    extracted=extracted_payload,
                    user_request=user_input or "",
                )
                if not generated:
                    logger.warning(f"[WARN] ProcessGPT generation failed: {process_name}")
                    continue

                elements_model = generated.get("elements_model") or {}
                proc_json = generated.get("definition") or {}
                
                # proc_def_id: UUID (already forced inside _generate_processgpt_definition_and_bpmn)
                proc_def_id = str(proc_json.get("processDefinitionId") or elements_model.get("processDefinitionId") or "").strip()
                if not proc_def_id:
                    # extremely defensive fallback (should not happen)
                    proc_def_id = str(uuid.uuid4())
                    proc_json["processDefinitionId"] = proc_def_id

                # NOTE: 담당자/에이전트 매핑은 forms + 참조정보(inputData) 확장 이후 마지막 단계에서 수행됨.
                
                # DB에 저장
                proc_def_data = {
                    "id": proc_def_id,
                    "name": process_name,
                    "definition": proc_json,
                    # BPMN XML은 "확장 완료 후" 최종본으로 생성/업데이트 한다.
                    "bpmn": "",
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
                            "tool_usage_started", 91,
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
                            "tool_usage_finished", 96,
                            {"proc_def_id": proc_def_id, "forms_result": forms_result},
                        )
                    except Exception as e:
                        logger.warning(f"[WARN] form generation/save stage failed unexpectedly: {e}")

                    # -----------------------------------------------------------------
                    # FINAL: 확장 단계 이후 최종 elements_model로 BPMN XML 생성 + DB 반영
                    # -----------------------------------------------------------------
                    final_bpmn_xml = ""
                    try:
                        # runtime_def(proc_json) -> elements_model sync (tool/inputData/agent etc)
                        final_elements_model = self._apply_runtime_definition_to_elements_model(
                            elements_model=elements_model,
                            runtime_def=proc_json,
                        )
                        final_bpmn_xml = await asyncio.to_thread(
                            self._generate_bpmn_xml_backend,
                            model=final_elements_model,
                            horizontal=final_elements_model.get("isHorizontal"),
                        )
                    except Exception as e:
                        logger.warning(f"[WARN] final BPMN XML generation failed: proc_def_id={proc_def_id} err={e}")

                    if final_bpmn_xml:
                        all_bpmn_xmls[proc_def_id] = final_bpmn_xml
                        try:
                            await self._update_proc_def_bpmn_only(proc_def_id=proc_def_id, tenant_id=tenant_id, bpmn_xml=final_bpmn_xml)
                        except Exception as e:
                            logger.warning(f"[WARN] proc_def.bpmn update failed after final xml: {e}")
                    else:
                        # keep empty bpmn; still allow completion
                        all_bpmn_xmls.setdefault(proc_def_id, "")
                
                # saved_processes에 bpmn_xml 포함
                saved_processes.append({
                    "id": proc_def_id,
                    "name": process_name,
                    "bpmn_xml": all_bpmn_xmls.get(proc_def_id, "")  # 최종 XML(없으면 빈 문자열)
                })
                
                # 진행 이벤트 (XML 포함)
                await self._send_progress_event(
                    event_queue, context_id, task_id, job_id,
                    f"[SAVED] 프로세스 저장 완료: {process_name}",
                    "tool_usage_finished", 90 + int(10 * (idx + 1) / total_bpmn),
                    {
                        "process_id": proc_def_id, 
                        "process_name": process_name,
                        "bpmn_xml": all_bpmn_xmls.get(proc_def_id, "")  # 이벤트에도 최종 XML 포함
                    }
                )
            
            # 10. 최종 결과 구성 (saved_processes에는 이미 bpmn_xml 포함됨)
            actual_count = len(saved_processes)
            logger.info(f"[DEBUG] Actual saved process count: {actual_count}")
            
            completed_message = (
                "[COMPLETED] PDF2BPMN 변환 완료: 문서에서 프로세스를 추출하지 못해 생성할 BPMN이 없습니다."
                if actual_count == 0
                else f"[COMPLETED] PDF2BPMN 변환 완료: {actual_count}개의 프로세스가 생성되었습니다."
            )

            final_result = {
                "message": completed_message,
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
            for p in [temp_pdf_path, temp_download_path]:
                if not p:
                    continue
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                        logger.info(f"[CLEANUP] Removed temp file: {p}")
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
