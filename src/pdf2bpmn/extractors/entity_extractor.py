"""LLM-based entity extraction from text."""
import json
import os
import re
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..config import Config
from ..models.entities import (
    Process, Task, Role, Gateway, Event,
    DMNDecision, DMNRule, Skill, TaskType, GatewayType, EventType,
    generate_id
)


class ExtractedEntities(BaseModel):
    """Extracted entities from text."""
    processes: list[dict] = Field(default_factory=list)
    tasks: list[dict] = Field(default_factory=list)
    roles: list[dict] = Field(default_factory=list)
    gateways: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    rules: list[dict] = Field(default_factory=list)
    skills: list[dict] = Field(default_factory=list)
    # Relationships
    task_role_mappings: list[dict] = Field(default_factory=list)
    task_process_mappings: list[dict] = Field(default_factory=list)
    role_skill_mappings: list[dict] = Field(default_factory=list)
    # Sequence flows between tasks
    sequence_flows: list[dict] = Field(default_factory=list)


EXTRACTION_PROMPT = """You are an expert at extracting business process elements from Korean business documents.
Your goal is to extract ALL relevant process elements from ANY type of business document, regardless of format.

Use the following explicit rules:
- Create or split entities only when the document provides a clear business reason.
- Do NOT create or split entities because of minor wording differences, sentence style differences, repeated explanation, or page change alone.
- Prefer stable business structure over creative inference.
- If the text clearly describes the same business purpose, keep it under the same process.
- If the text clearly describes the same business action by the same role, keep it under the same task.

{existing_context}

Analyze the following text and extract:
1. **Processes**: Business processes or procedures (절차, 업무 흐름, 처리 단계, 프로세스)
2. **Tasks/Activities**: Individual activities or actions (행위: ~한다, ~해야 한다, 점검, 승인, 검토, 접수, 등록, 통보, 보고)
3. **Roles**: Actors or performers (담당자, 승인권자, 검토자, 부서명, 직책, 시스템, 외부기관)
4. **Gateways**: Decision/branching POINTS only (분기점 - where flow splits or merges)
5. **Events**: Start/End triggers (요청 접수 시, 신청서 제출 후, 정기적으로, 완료 시)
6. **Decisions**: Business rules or decision logic (if-then rules)
7. **Rules**: Specific decision rules (조건-결과 pairs)

IMPORTANT: Also extract RELATIONSHIPS between entities:
8. **task_role_mappings**: Which role performs which task
9. **task_process_mappings**: Which process contains which task
10. **sequence_flows**: The order/sequence between tasks WITH CONDITIONS (VERY IMPORTANT!)
    - Identify which task comes BEFORE and AFTER another
    - **CONDITIONS GO HERE, NOT IN GATEWAYS**: "승인인 경우", "거부인 경우", "예산 부족 시", "금액 100만원 이상" etc.
    - Look for conditional keywords: "~인 경우", "~면", "~시", "아니면", "그렇지 않으면"
    - Look for sequential keywords: "다음", "이후", "후에", "그 다음", "완료 후"
    - Look for numbered steps: 1단계, 2단계, Step 1, Step 2

For each entity, provide:
- name: Clear, concise name
- description: Brief description from the text
- evidence: The exact text span that supports this extraction
- confidence: Your confidence level (0.0 to 1.0)
- order: Sequential order number if identifiable (1, 2, 3...)

PROCESS DECISION RULES:
- Start a NEW process only when at least one of these is clearly true:
  1. business purpose changes
  2. trigger/start condition changes
  3. final outcome changes
  4. the document explicitly introduces a separate procedure/process/workflow
- Do NOT start a new process when:
  1. the same process is described in more detail
  2. the same process continues on a later page
  3. wording/style changes but business purpose stays the same
  4. supporting notes/examples/exceptions are added

TASK DECISION RULES:
- Create a NEW task only when at least one of these is clearly true:
  1. performer role changes
  2. there is a real handoff or stage transition
  3. there is an approval/review/waiting/decision point between actions
  4. the business action itself is different
- Do NOT create a new task when:
  1. the same role continues the same purpose with more detail
  2. the text only adds explanation, checklist items, or execution notes
  3. two adjacent lines describe the same business action with slightly different wording

GATEWAY DECISION RULES:
- Create a gateway only when the text indicates a real split, merge, or decision point.
- Do NOT create a gateway for simple sequential explanation or detail expansion.
- If there is no real branching evidence, keep the flow linear.

For tasks, also identify:
- task_type: "human" (사람이 수행), "agent" (AI/자동화 가능), "system" (시스템 자동)
- performer_role: Name of the role that performs this task (IMPORTANT!)
- parent_process: Name of the process this task belongs to (IMPORTANT!)
- instruction: **How to perform this task** (태스크 수행 지침, Korean)
  - Preserve document wording as much as possible. Do NOT summarize or paraphrase when explicit instructions are present.
  - Keep line breaks/order from source text whenever possible.
  - Include inputs/outputs/checkpoints exactly as stated in the document if provided.
  - If the source does not provide explicit execution instructions, return "".
- order: Sequential order within the process (1, 2, 3...)
- next_task: Name of the task that follows this one (if identifiable)
- previous_task: Name of the task that precedes this one (if identifiable)
- Prefer one stable representative name for the same business action.
- Avoid near-duplicate task names that differ only by modifiers, suffixes, or wording order.

For gateways (IMPORTANT: Gateway is just a BRANCHING POINT, NOT the condition itself):
- gateway_type: "exclusive" (XOR - only one path taken), "parallel" (AND - all paths), "inclusive" (OR)
- name: Descriptive name for the decision point (e.g., "승인 여부 분기", "예산 확인 분기", "금액 기준 분기")
  - If description contains "~인지 여부", gateway name MUST follow "<주제> 여부 판단"
    Example: "예비타당성조사 대상 사업인지 여부..." -> "예비타당성조사 대상 사업 여부 판단"
  DO NOT put condition text here - just the name of the decision point
- parent_process: Name of the process this gateway belongs to
- incoming_task: Name of the task before this gateway (REQUIRED — what flows INTO this gateway)
- outgoing_branches: **REQUIRED** — List of branches OUT of this gateway. This is the most
  important field — without it the gateway has no purpose and will be discarded.
  Format: [ {{ "to_task": "<existing task name>", "condition": "<branch condition>" }}, ... ]
  Rules (MUST follow):
    1. MUST contain at least 2 entries (a gateway with <2 branches is not a gateway).
    2. Each "to_task" MUST refer to a task that exists in the tasks list (or in the
       provided existing_tasks context). Do NOT invent task names that are not in the
       document; do NOT use gateway names; do NOT use roles or events.
    3. Each "condition" MUST be a real, semantically distinct condition from the source
       text (e.g., "승인인 경우" / "거부인 경우"). Do NOT use placeholders like
       "조건1", "분기1", or "case1".
    4. For exclusive/inclusive gateways: conditions must be mutually meaningful (e.g.,
       a true/false pair or a clear set of exclusive cases).
    5. For parallel gateways: condition can be "" but at least 2 outgoing tasks are still
       required.
  Example for "승인권자가 승인하면 발주 처리를 진행하고, 거부하면 반려 통보한다":
    outgoing_branches = [
      {{ "to_task": "발주 처리", "condition": "승인인 경우" }},
      {{ "to_task": "반려 통보", "condition": "거부인 경우" }}
    ]
  NOTE: You MUST also list these same branches as entries in `sequence_flows` (with
  from_task = this gateway's name). The two MUST be consistent. Duplicate listing is
  intentional and required.
- description: Brief description of what decision is being made

For decisions:
- input_data: List of input data items
- output_data: List of output data items
- related_role: Role that makes this decision

For rules:
- decision_name: Name of the decision this rule belongs to (IMPORTANT)
- decision_id: Optional, only when clearly known
- when: Condition expression
- then: Result/action

IMPORTANT POLICY CHANGE:
- Do NOT extract "skills" as a separate entity.
- If the text contains professional know-how/criteria/judgment logic, map it to:
  1) task.instruction (execution guidance), and/or
  2) decisions/rules (explicit decision criteria).
- Therefore, return "skills": [] and "role_skill_mappings": [].

For task_role_mappings:
- task_name: Name of the task
- role_name: Name of the role that performs it

For task_process_mappings:
- task_name: Name of the task
- process_name: Name of the parent process

**For sequence_flows (CRITICAL - CONDITIONS ARE EXTRACTED HERE!):**
- from_task: Name of the source task (or gateway name like "승인 여부 분기")
- to_task: Name of the target task
- condition: **THE CONDITION FOR THIS SPECIFIC PATH** (VERY IMPORTANT!)
  Examples:
    - "승인인 경우" (when approved)
    - "거부인 경우" (when rejected)
    - "예산 충분" (budget sufficient)
    - "예산 부족" (budget insufficient)
    - "금액 100만원 이상" (amount >= 1M)
    - "금액 100만원 미만" (amount < 1M)
  Leave empty ("") for default/unconditional flows
- CONDITION QUALITY RULES (MANDATORY):
  - NEVER output placeholder conditions like "조건1", "조건 1", "분기1", "case1"
  - NEVER output graph edge names/labels like "HAS_TASK", "HAS_INSTRUCTION", "PERFORMED_BY"
  - If a gateway has multiple outgoing paths, each condition must be semantically distinct
  - If an exclusive gateway has exactly 2 outgoing flows, prefer true/false pair:
    "<주제>인 경우" / "<주제>이 아닌 경우"
  - Prefer source wording from document ("승인인 경우", "미승인 시", "요건 충족 시", "재검토 필요 시")
- process_name: Name of the process this flow belongs to

SELF-CHECK BEFORE OUTPUT:
- Did I create a new process only when the business purpose truly changed?
- Did I avoid creating duplicate tasks from repeated explanation?
- Did I avoid creating gateways without real branching evidence?
- Are the names stable and reusable as business labels?

Example of correct extraction:
If text says: "승인권자가 승인하면 발주 처리를 진행하고, 거부하면 구매요청자에게 반려 통보한다"
Extract:
- Gateway:
    name="승인 여부 분기",
    gateway_type="exclusive",
    incoming_task="승인 요청",
    outgoing_branches=[
      {{ "to_task": "발주 처리", "condition": "승인인 경우" }},
      {{ "to_task": "반려 통보", "condition": "거부인 경우" }}
    ]
- sequence_flows (consistent duplicate listing):
  1. from_task="승인 여부 분기", to_task="발주 처리", condition="승인인 경우"
  2. from_task="승인 여부 분기", to_task="반려 통보", condition="거부인 경우"

TEXT TO ANALYZE:
{text}

Respond with a JSON object containing arrays for each entity type.
Return ONLY valid JSON, no markdown formatting."""


# Context template for existing processes/roles/tasks
EXISTING_CONTEXT_TEMPLATE = """
**IMPORTANT - EXISTING ENTITIES (이미 추출된 엔티티들):**

{process_list}
{role_list}
{task_list}

**CRITICAL RULES FOR PROCESS IDENTIFICATION (프로세스 식별 규칙):**
1. If a task clearly belongs to an EXISTING process listed above, use that EXACT process name for parent_process.
   (태스크가 위에 나열된 기존 프로세스에 속하면, 정확히 그 프로세스 이름을 parent_process로 사용하세요)

2. Do NOT create a new process if the content describes steps/tasks of an existing process.
   (내용이 기존 프로세스의 단계/태스크를 설명하는 경우 새 프로세스를 만들지 마세요)

3. "발주 처리", "입고 검수", "대금 지급" etc. are likely TASKS within a larger process, NOT separate processes.
   ("발주 처리", "입고 검수", "대금 지급" 등은 별도 프로세스가 아니라 상위 프로세스의 태스크일 가능성이 높습니다)

4. Look for phrases like "~단계", "~절차", "제N조" which indicate sub-steps of an existing process.
   ("~단계", "~절차", "제N조" 같은 표현은 기존 프로세스의 하위 단계를 나타냅니다)

5. Only create a NEW process if the text explicitly defines a completely different business process.
   (텍스트가 완전히 다른 업무 프로세스를 명시적으로 정의하는 경우에만 새 프로세스를 생성하세요)

6. For existing roles, use the EXACT same name - do not create duplicates with slightly different names.
   (기존 역할의 경우 정확히 같은 이름을 사용하세요 - 약간 다른 이름으로 중복 생성하지 마세요)

**CRITICAL RULES FOR TASK DEDUPLICATION (태스크 중복 제거 규칙):**
7. BEFORE creating a new task, check the EXISTING TASKS list above. If a similar task already exists, DO NOT create a duplicate.
   (새 태스크를 만들기 전에 위의 기존 태스크 목록을 확인하세요. 유사한 태스크가 이미 있으면 중복 생성하지 마세요)

8. Tasks performed by the SAME ROLE in sequence WITHOUT requiring other department collaboration should be MERGED into ONE task.
   (다른 부서 협업 없이 같은 역할이 연속으로 수행하는 작업은 하나의 태스크로 통합해야 합니다)
   Examples of tasks to MERGE (통합해야 할 태스크 예시):
   - "구매요청서 접수" + "형식 검토" → "구매요청서 접수 및 형식 검토" (same role: 구매담당자)
   - "견적서 검토" + "견적 비교" → "견적서 검토 및 비교" (same role)

9. Tasks are SEPARATE only when:
   - Different roles perform them (역할이 다를 때)
   - There is a waiting/approval point between them (승인/대기 시점이 있을 때)
   - They belong to different stages requiring handoff (업무 인계가 필요한 다른 단계일 때)

10. If the current text provides MORE DETAIL about an existing task, DO NOT create a new task.
    Instead, the existing task's description should be enhanced (but this is handled in post-processing).
    (현재 텍스트가 기존 태스크에 대한 추가 설명이면 새 태스크를 만들지 마세요)

"""


class EntityExtractor:
    """Extract business process entities using LLM."""
    
    def __init__(self):
        try:
            llm_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        except Exception:
            llm_retries = 2
        llm_retries = max(0, llm_retries)

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            base_url=(Config.LLM_BASE_URL or None),
            temperature=0,
            # 원복: 응답이 늦어도 완료될 때까지 대기
            timeout=None,
            max_retries=llm_retries,
        )
        self.parser = JsonOutputParser(pydantic_object=ExtractedEntities)
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        self.chain = self.prompt | self.llm | self.parser

    def _is_placeholder_gateway_name(self, name: str) -> bool:
        key = "".join(str(name or "").lower().split())
        if not key:
            return True
        if key in {"분기", "gateway", "gw", "decision"}:
            return True
        if any(key.startswith(prefix) for prefix in ("분기", "gateway", "gw", "decision")) and any(ch.isdigit() for ch in key):
            return True
        return False

    def _extract_gateway_subject(self, text: str) -> str:
        s = " ".join(str(text or "").split()).strip()
        if not s:
            return ""
        patterns = [
            r"(.+?)(?:인지|인가)\s*여부",
            r"(.+?)\s*여부",
        ]
        for p in patterns:
            m = re.search(p, s)
            if m:
                subj = " ".join((m.group(1) or "").split()).strip(" .,-_")
                if subj:
                    return subj
        for tok in ("분기", "판단", "확인", "검토", "결정", "여부"):
            s = s.replace(tok, " ")
        s = " ".join(s.split()).strip(" .,-_")
        return s

    def _derive_gateway_name(self, raw_name: str, description: str, idx: int) -> str:
        rn = " ".join(str(raw_name or "").split()).strip()
        desc = " ".join(str(description or "").split()).strip()
        subj = self._extract_gateway_subject(desc)
        if subj:
            return f"{subj} 여부 판단"
        if rn and not self._is_placeholder_gateway_name(rn):
            return rn
        subj2 = self._extract_gateway_subject(rn)
        if subj2:
            return f"{subj2} 여부 판단"
        return f"의사결정 분기 {idx}"

    def _derive_true_false_conditions(self, gateway_name: str, gateway_description: str) -> tuple[str, str]:
        subj = self._extract_gateway_subject(gateway_description) or self._extract_gateway_subject(gateway_name)
        if subj:
            return (f"{subj}인 경우", f"{subj}이 아닌 경우")
        return ("조건 충족인 경우", "조건 미충족인 경우")

    def _build_context(
        self, 
        existing_processes: list[str] = None, 
        existing_roles: list[str] = None,
        existing_tasks: list[dict] = None
    ) -> str:
        """기존 프로세스/역할/태스크 목록으로 컨텍스트 문자열 생성
        
        Args:
            existing_processes: 기존 프로세스 이름 목록
            existing_roles: 기존 역할 이름 목록
            existing_tasks: 기존 태스크 목록 [{name, role, process, order}, ...]
        """
        if not existing_processes and not existing_roles and not existing_tasks:
            return ""
        
        process_list = ""
        if existing_processes:
            process_list = "**기존 프로세스 목록 (Existing Processes):**\n" + \
                          "\n".join(f"  - {p}" for p in existing_processes)
        
        role_list = ""
        if existing_roles:
            role_list = "**기존 역할 목록 (Existing Roles):**\n" + \
                       "\n".join(f"  - {r}" for r in existing_roles)
        
        task_list = ""
        if existing_tasks:
            task_entries = []
            for t in existing_tasks:
                entry = f"  - {t.get('name', 'Unknown')}"
                if t.get('role'):
                    entry += f" (담당: {t['role']})"
                if t.get('process'):
                    entry += f" [프로세스: {t['process']}]"
                task_entries.append(entry)
            task_list = "**기존 태스크 목록 (Existing Tasks - DO NOT DUPLICATE):**\n" + \
                       "\n".join(task_entries)
        
        return EXISTING_CONTEXT_TEMPLATE.format(
            process_list=process_list,
            role_list=role_list,
            task_list=task_list
        )
    
    def extract_from_text(
        self, 
        text: str,
        existing_processes: list[str] = None,
        existing_roles: list[str] = None,
        existing_tasks: list[dict] = None
    ) -> ExtractedEntities:
        """Extract entities from text using LLM.
        
        Args:
            text: 분석할 텍스트
            existing_processes: 이미 추출된 프로세스 이름 목록
            existing_roles: 이미 추출된 역할 이름 목록
            existing_tasks: 이미 추출된 태스크 목록 [{name, role, process, order}, ...]
        
        Returns:
            ExtractedEntities: 추출된 엔티티들
        """
        t0 = time.perf_counter()
        try:
            # 기존 컨텍스트 생성 (프로세스, 역할, 태스크 모두 포함)
            t_ctx0 = time.perf_counter()
            existing_context = self._build_context(
                existing_processes, 
                existing_roles,
                existing_tasks
            )
            t_ctx_ms = int((time.perf_counter() - t_ctx0) * 1000)
            print(
                f"   ⏱️ [EXTRACT-TIMING] context_build={t_ctx_ms}ms "
                f"text_len={len(text or '')} ctx_len={len(existing_context or '')}"
            )
            
            t_llm0 = time.perf_counter()
            result = self.chain.invoke({
                "text": text,
                "existing_context": existing_context,
            })
            t_llm_ms = int((time.perf_counter() - t_llm0) * 1000)

            t_parse0 = time.perf_counter()
            parsed = ExtractedEntities(**result)
            t_parse_ms = int((time.perf_counter() - t_parse0) * 1000)
            t_total_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"   ⏱️ [EXTRACT-TIMING] llm_invoke={t_llm_ms}ms parse={t_parse_ms}ms total={t_total_ms}ms "
                f"(tasks={len(parsed.tasks)}, roles={len(parsed.roles)}, gateways={len(parsed.gateways)}, "
                f"flows={len(parsed.sequence_flows)}, decisions={len(parsed.decisions)}, rules={len(parsed.rules)})"
            )
            return parsed
        except Exception as e:
            t_total_ms = int((time.perf_counter() - t0) * 1000)
            print(f"Extraction error: {e} (elapsed={t_total_ms}ms)")
            return ExtractedEntities()

    def _is_reusable_skill_candidate(self, skill_payload: dict) -> bool:
        """Filter out task-local instructions; keep reusable professional skills."""
        if not isinstance(skill_payload, dict):
            return False

        name = str(skill_payload.get("name", "") or "").strip().lower()
        summary = str(skill_payload.get("summary", "") or "").strip().lower()
        purpose = str(skill_payload.get("purpose", "") or "").strip().lower()
        related_tasks = skill_payload.get("related_tasks", [])
        if not isinstance(related_tasks, list):
            related_tasks = []

        if len(name) < 3:
            return False

        # Highly task-local verbs/checklist wording usually indicate instructions, not reusable skills.
        task_local_markers = [
            "접수", "제출", "등록", "전달", "통보", "요청", "확인 후", "버튼", "클릭", "입력",
            "업로드", "다운로드", "시스템", "화면", "메뉴", "양식 작성", "문서 작성", "작성 및 제출",
        ]

        # Reusable expertise markers.
        reusable_markers = [
            "평가", "분석", "판독", "산정", "예측", "검증", "분류", "판단", "기준", "규정",
            "정책", "리스크", "위험도", "모델", "전략", "방법론", "체계", "기법", "해석",
        ]

        text = " ".join([name, summary, purpose]).strip()
        if not text:
            return False

        has_task_local = any(m in text for m in task_local_markers)
        has_reusable = any(m in text for m in reusable_markers)

        # If only one very specific task is referenced and no reusable signal, likely instruction-like.
        if len(related_tasks) <= 1 and has_task_local and not has_reusable:
            return False

        # If task-local wording dominates with no reusable clue, reject.
        if has_task_local and not has_reusable:
            return False

        # Accept if reusable clues exist, or if purpose/summary are generic knowledge statements.
        return has_reusable or ("전문" in text or "노하우" in text or "역량" in text)
    
    def convert_to_entities(
        self, 
        extracted: ExtractedEntities,
        doc_id: str,
        chunk_id: str = "",
        existing_processes: dict = None,
        existing_roles: dict = None
    ) -> dict[str, Any]:
        """Convert extracted data to entity objects with relationships."""
        existing_processes = existing_processes or {}
        existing_roles = existing_roles or {}
        
        entities = {
            "processes": [],
            "tasks": [],
            "roles": [],
            "gateways": [],
            "events": [],
            "decisions": [],
            "rules": [],
            "skills": [],
            "evidences": [],
            # Relationship mappings
            "task_role_map": {},  # task_id -> role_id
            "task_process_map": {},  # task_id -> process_id
            "role_decision_map": {},  # role_id -> [decision_ids]
            "role_skill_map": {},  # role_id -> [skill_ids]
            "entity_chunk_map": {},  # entity_id -> chunk_id (for evidence)
            # Sequence flows (task ordering)
            "sequence_flows": [],  # list of {from_task_id, to_task_id, condition}
        }
        
        # Task name to ID mapping (built as we create tasks)
        task_name_to_id = {}
        
        # Create name -> id mappings for linking
        process_name_to_id = dict(existing_processes)
        role_name_to_id = dict(existing_roles)
        
        # Convert processes
        for p in extracted.processes:
            proc_id = generate_id()
            proc = Process(
                proc_id=proc_id,
                name=p.get("name", "Unknown Process"),
                purpose=p.get("description", ""),
                description=p.get("description", "")
            )
            entities["processes"].append(proc)
            process_name_to_id[proc.name.lower()] = proc_id
            
            # Link to source chunk
            if chunk_id:
                entities["entity_chunk_map"][proc_id] = chunk_id
        
        # Convert roles
        for r in extracted.roles:
            role_name = r.get("name", "Unknown Role")
            role_key = role_name.lower()
            
            # Check if role already exists
            if role_key not in role_name_to_id:
                role_id = generate_id()
                role = Role(
                    role_id=role_id,
                    name=role_name,
                    org_unit=r.get("org_unit", ""),
                    persona_hint=r.get("persona_hint", r.get("description", ""))
                )
                entities["roles"].append(role)
                role_name_to_id[role_key] = role_id
                
                if chunk_id:
                    entities["entity_chunk_map"][role_id] = chunk_id
        
        # Convert tasks with relationships
        for i, t in enumerate(extracted.tasks):
            task_type_str = (t.get("task_type") or "human").lower()
            task_type = TaskType.HUMAN
            if task_type_str == "agent":
                task_type = TaskType.AGENT
            elif task_type_str == "system":
                task_type = TaskType.SYSTEM
            
            task_id = generate_id()
            
            # Find parent process
            parent_process_name = (t.get("parent_process") or "").lower()
            process_id = ""
            if parent_process_name:
                process_id = process_name_to_id.get(parent_process_name, "")
            # If not found, use first extracted process
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id
            
            # Get order from extracted data or use index
            task_order = t.get("order")
            if task_order is None:
                task_order = i
            elif isinstance(task_order, (int, float)):
                task_order = int(task_order)
            elif isinstance(task_order, str):
                try:
                    task_order = int(float(task_order))
                except:
                    task_order = i
            else:
                task_order = i
            
            task_name = t.get("name", f"Task {i+1}")
            
            task = Task(
                task_id=task_id,
                process_id=process_id,
                name=task_name,
                task_type=task_type,
                description=t.get("description", ""),
                instruction=t.get("instruction", "") or "",
                order=task_order
            )
            entities["tasks"].append(task)
            
            # Store task name -> id mapping for sequence flow resolution
            task_name_to_id[task_name.lower()] = task_id
            
            # Map task to process
            if process_id:
                entities["task_process_map"][task_id] = process_id
            
            # Find performer role
            performer_role = (t.get("performer_role") or "").lower()
            if performer_role and performer_role in role_name_to_id:
                entities["task_role_map"][task_id] = role_name_to_id[performer_role]
            
            if chunk_id:
                entities["entity_chunk_map"][task_id] = chunk_id
            
            # Store next/previous task info for later sequence flow creation
            if t.get("next_task"):
                task._next_task_name = t.get("next_task")
            if t.get("previous_task"):
                task._previous_task_name = t.get("previous_task")
        
        # Process explicit task-role mappings
        for mapping in extracted.task_role_mappings:
            task_name = (mapping.get("task_name") or "").lower()
            role_name = (mapping.get("role_name") or "").lower()
            
            # Find matching task and role
            for task in entities["tasks"]:
                if task.name.lower() == task_name or task_name in task.name.lower():
                    if role_name in role_name_to_id:
                        entities["task_role_map"][task.task_id] = role_name_to_id[role_name]
                        break
        
        # Process explicit task-process mappings
        for mapping in extracted.task_process_mappings:
            task_name = (mapping.get("task_name") or "").lower()
            process_name = (mapping.get("process_name") or "").lower()
            
            for task in entities["tasks"]:
                if task.name.lower() == task_name or task_name in task.name.lower():
                    if process_name in process_name_to_id:
                        task.process_id = process_name_to_id[process_name]
                        entities["task_process_map"][task.task_id] = process_name_to_id[process_name]
                        break
        
        # Convert gateways
        # gateway_id → raw extracted dict (for outgoing_branches / incoming_task post-processing below)
        gateway_raw_meta: Dict[str, Dict[str, Any]] = {}
        for g in extracted.gateways:
            gw_type_str = (g.get("gateway_type") or "exclusive").lower()
            gw_type = GatewayType.EXCLUSIVE
            if gw_type_str == "parallel":
                gw_type = GatewayType.PARALLEL
            elif gw_type_str == "inclusive":
                gw_type = GatewayType.INCLUSIVE
            
            gateway_id = generate_id()
            
            # Find parent process
            parent_process_name = (g.get("parent_process") or "").lower()
            process_id = process_name_to_id.get(parent_process_name, "")
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id

            gw_desc = str(g.get("description", "") or "")
            gw_name = self._derive_gateway_name(
                str(g.get("name", "") or ""),
                gw_desc,
                idx=(len(entities["gateways"]) + 1),
            )
            
            gateway = Gateway(
                gateway_id=gateway_id,
                process_id=process_id,
                name=gw_name,
                gateway_type=gw_type,
                condition=g.get("condition", ""),
                description=gw_desc
            )
            entities["gateways"].append(gateway)
            gateway_raw_meta[gateway_id] = {
                "incoming_task": str(g.get("incoming_task") or "").strip(),
                "outgoing_branches": g.get("outgoing_branches") or [],
            }
            
            if chunk_id:
                entities["entity_chunk_map"][gateway_id] = chunk_id
        
        # Convert events
        for e in extracted.events:
            event_type_str = (e.get("event_type") or "start").lower()
            event_type = EventType.START
            if event_type_str == "end":
                event_type = EventType.END
            elif event_type_str == "intermediate":
                event_type = EventType.INTERMEDIATE
            
            event_id = generate_id()
            
            # Find parent process
            parent_process_name = (e.get("parent_process") or "").lower()
            process_id = process_name_to_id.get(parent_process_name, "")
            if not process_id and entities["processes"]:
                process_id = entities["processes"][0].proc_id
            
            event = Event(
                event_id=event_id,
                process_id=process_id,
                event_type=event_type,
                name=e.get("name", "Event"),
                trigger=e.get("trigger", "")
            )
            entities["events"].append(event)
            
            if chunk_id:
                entities["entity_chunk_map"][event_id] = chunk_id
        
        # Convert decisions and rules with role linkage
        decision_name_to_id = {}
        for d in extracted.decisions:
            decision_id = generate_id()
            dname = d.get("name", "Decision")
            decision = DMNDecision(
                decision_id=decision_id,
                name=dname,
                description=d.get("description", ""),
                input_data=d.get("input_data", []),
                output_data=d.get("output_data", [])
            )
            entities["decisions"].append(decision)
            if dname:
                decision_name_to_id[str(dname).lower().strip()] = decision_id
            
            # Link decision to role
            related_role = (d.get("related_role") or "").lower()
            if related_role and related_role in role_name_to_id:
                role_id = role_name_to_id[related_role]
                if role_id not in entities["role_decision_map"]:
                    entities["role_decision_map"][role_id] = []
                entities["role_decision_map"][role_id].append(decision_id)
            
            if chunk_id:
                entities["entity_chunk_map"][decision_id] = chunk_id
        
        for r in extracted.rules:
            rule_id = generate_id()
            mapped_decision_id = str(r.get("decision_id") or "").strip()
            if not mapped_decision_id:
                decision_name = str(r.get("decision_name") or r.get("decision") or "").lower().strip()
                if decision_name:
                    mapped_decision_id = decision_name_to_id.get(decision_name, "")
            if not mapped_decision_id and entities["decisions"]:
                # deterministic fallback: attach to first decision when only rules exist without explicit mapping
                mapped_decision_id = entities["decisions"][0].decision_id
            rule = DMNRule(
                rule_id=rule_id,
                decision_id=mapped_decision_id,
                when=r.get("when", r.get("condition", "")),
                then=r.get("then", r.get("result", "")),
                confidence=r.get("confidence", 1.0)
            )
            entities["rules"].append(rule)
            
            if chunk_id:
                entities["entity_chunk_map"][rule_id] = chunk_id

        # Policy change: skills are no longer first-class outputs.
        # Fold skill-like know-how into task instruction and DMN rules deterministically.
        decision_name_to_obj = {d.name.lower().strip(): d for d in entities["decisions"] if getattr(d, "name", None)}
        for s in extracted.skills:
            if not isinstance(s, dict):
                continue
            summary = str(s.get("summary") or s.get("description") or "").strip()
            purpose = str(s.get("purpose") or "").strip()
            proc = s.get("procedure") if isinstance(s.get("procedure"), list) else []
            proc_lines = [str(x).strip() for x in proc if str(x).strip()]
            knowhow_text = "\n".join([x for x in [summary, purpose, *proc_lines] if x]).strip()
            if not knowhow_text:
                continue

            related_tasks = s.get("related_tasks") if isinstance(s.get("related_tasks"), list) else []
            if related_tasks:
                for tname in related_tasks:
                    tn = str(tname or "").lower().strip()
                    if not tn:
                        continue
                    for task in entities["tasks"]:
                        task_name = str(task.name or "").lower().strip()
                        if task_name and (task_name == tn or tn in task_name or task_name in tn):
                            current_inst = str(task.instruction or "").strip()
                            if knowhow_text not in current_inst:
                                task.instruction = (current_inst + "\n" + knowhow_text).strip() if current_inst else knowhow_text
                            break

            # Also materialize as DMN rule (criteria/judgment logic) when possible.
            dname = str(s.get("decision_name") or s.get("decision") or "").strip()
            related_role = str(s.get("related_role") or "").strip()
            if not dname:
                dname = f"{related_role} 판단 기준" if related_role else "업무 판단 기준"
            dkey = dname.lower().strip()
            decision = decision_name_to_obj.get(dkey)
            if not decision:
                decision = DMNDecision(
                    decision_id=generate_id(),
                    name=dname,
                    description="문서 내 전문지식/판단기준에서 자동 보강",
                    input_data=[],
                    output_data=["판단 결과"],
                )
                entities["decisions"].append(decision)
                decision_name_to_obj[dkey] = decision
                # Link decision to role when resolvable
                rr = related_role.lower().strip()
                if rr and rr in role_name_to_id:
                    rid = role_name_to_id[rr]
                    entities["role_decision_map"].setdefault(rid, []).append(decision.decision_id)

            entities["rules"].append(
                DMNRule(
                    rule_id=generate_id(),
                    decision_id=decision.decision_id,
                    when="업무 수행 시",
                    then=knowhow_text,
                    confidence=0.8,
                )
            )

        # Convert skills with optional role linkage (disabled by policy)
        skill_name_to_id = {}
        for s in []:
            if not self._is_reusable_skill_candidate(s):
                continue
            skill_id = generate_id()
            name = s.get("name", "Skill")
            procedure = s.get("procedure", [])
            if not isinstance(procedure, list):
                procedure = []
            procedure = [str(x).strip() for x in procedure if str(x).strip()]

            skill = {
                "skill_id": skill_id,
                "name": name,
                "summary": s.get("summary", s.get("description", "")) or "",
                "purpose": s.get("purpose", s.get("summary", "")) or "",
                "inputs": s.get("inputs", {}) if isinstance(s.get("inputs"), dict) else {},
                "outputs": s.get("outputs", {}) if isinstance(s.get("outputs"), dict) else {},
                "preconditions": s.get("preconditions", []) if isinstance(s.get("preconditions"), list) else [],
                "procedure": procedure,
                "exceptions": s.get("exceptions", []) if isinstance(s.get("exceptions"), list) else [],
                "tools": s.get("tools", []) if isinstance(s.get("tools"), list) else [],
                "md_path": "",
            }
            entities["skills"].append(Skill(**skill))
            skill_name_to_id[name.lower().strip()] = skill_id

            # Optional direct role link from skill payload
            related_role = (s.get("related_role") or "").lower().strip()
            if related_role and related_role in role_name_to_id:
                role_id = role_name_to_id[related_role]
                entities["role_skill_map"].setdefault(role_id, []).append(skill_id)

            if chunk_id:
                entities["entity_chunk_map"][skill_id] = chunk_id

        # Explicit role-skill mappings (disabled by policy)
        for mapping in []:
            role_name = (mapping.get("role_name") or "").lower().strip()
            skill_name = (mapping.get("skill_name") or "").lower().strip()
            if not role_name or not skill_name:
                continue
            role_id = role_name_to_id.get(role_name)
            skill_id = skill_name_to_id.get(skill_name)
            if role_id and skill_id:
                entities["role_skill_map"].setdefault(role_id, []).append(skill_id)
        
        # Build gateway name -> id mapping
        gateway_name_to_id = {}
        for gw in entities["gateways"]:
            gname = str(gw.name or "").lower().strip()
            if gname:
                gateway_name_to_id[gname] = gw.gateway_id
        
        # Process sequence flows from extracted data (Task->Task, Task->Gateway, Gateway->Task)
        _BAD_CONDITION_KEYS = {
            "조건1", "조건2", "조건3", "조건4", "조건5",
            "분기1", "분기2", "분기3",
            "case1", "case2", "case3",
            "hastask", "hasinstruction", "performedby", "hasgateway", "hasevent",
        }
        for flow in extracted.sequence_flows:
            from_name = (flow.get("from_task") or "").lower()
            to_name = (flow.get("to_task") or "").lower()
            condition = flow.get("condition", "") or ""
            cond_norm = "".join(str(condition).lower().split())
            if cond_norm in _BAD_CONDITION_KEYS:
                condition = ""
            
            from_id = None
            from_type = None
            to_id = None
            to_type = None
            
            # Find source: check tasks first, then gateways
            for task in entities["tasks"]:
                task_lower = task.name.lower()
                if from_name and (task_lower == from_name or from_name in task_lower or task_lower in from_name):
                    from_id = task.task_id
                    from_type = "task"
                    break
            
            if not from_id:
                # Check gateways
                for gw_name, gw_id in gateway_name_to_id.items():
                    if from_name and (gw_name == from_name or from_name in gw_name or gw_name in from_name):
                        from_id = gw_id
                        from_type = "gateway"
                        break
            
            # Find target: check tasks first, then gateways
            for task in entities["tasks"]:
                task_lower = task.name.lower()
                if to_name and (task_lower == to_name or to_name in task_lower or task_lower in to_name):
                    to_id = task.task_id
                    to_type = "task"
                    break
            
            if not to_id:
                # Check gateways
                for gw_name, gw_id in gateway_name_to_id.items():
                    if to_name and (gw_name == to_name or to_name in gw_name or gw_name in to_name):
                        to_id = gw_id
                        to_type = "gateway"
                        break
            
            if from_id and to_id:
                entities["sequence_flows"].append({
                    "from_id": from_id,
                    "from_type": from_type,
                    "to_id": to_id,
                    "to_type": to_type,
                    "condition": condition.strip() if condition else ""
                })
                
                # Log for debugging
                if condition:
                    print(f"   📍 Sequence flow with condition: {from_name} → {to_name} [{condition}]")

        # ─────────────────────────────────────────────────────────────────────
        # (Option A) Gateway.outgoing_branches → sequence_flows 자동 보강
        #   LLM 이 sequence_flows 에 게이트웨이 분기를 빠뜨려도, gateway 객체에
        #   명시한 outgoing_branches 만 채워뒀다면 그 정보로 sequence_flow 를
        #   복원한다. 추출 단계의 안전망.
        # ─────────────────────────────────────────────────────────────────────
        if gateway_raw_meta:
            # 기존 sequence_flows 에서 이미 등록된 (from_id, to_id) 쌍을 skip 대상으로 모음
            existing_pairs: set = set()
            for sf in entities["sequence_flows"]:
                _f = sf.get("from_id") or sf.get("from_task_id")
                _t = sf.get("to_id") or sf.get("to_task_id")
                if _f and _t:
                    existing_pairs.add((str(_f), str(_t)))

            task_name_to_id_lc = {
                (t.name or "").lower().strip(): t.task_id
                for t in entities["tasks"]
                if t.name
            }

            augmented = 0
            for gw_id, meta in gateway_raw_meta.items():
                branches = meta.get("outgoing_branches") or []
                if not isinstance(branches, list):
                    continue
                # incoming_task → gateway 흐름도 누락됐다면 같이 보강
                in_name = (meta.get("incoming_task") or "").lower().strip()
                if in_name:
                    in_task_id = None
                    if in_name in task_name_to_id_lc:
                        in_task_id = task_name_to_id_lc[in_name]
                    else:
                        for tname_lc, tid in task_name_to_id_lc.items():
                            if tname_lc and (tname_lc in in_name or in_name in tname_lc):
                                in_task_id = tid
                                break
                    if in_task_id and (in_task_id, gw_id) not in existing_pairs:
                        entities["sequence_flows"].append({
                            "from_id": in_task_id,
                            "from_type": "task",
                            "to_id": gw_id,
                            "to_type": "gateway",
                            "condition": "",
                        })
                        existing_pairs.add((in_task_id, gw_id))
                        augmented += 1

                for br in branches:
                    if not isinstance(br, dict):
                        continue
                    to_name = (br.get("to_task") or "").lower().strip()
                    cond = str(br.get("condition") or "").strip()
                    if not to_name:
                        continue
                    cond_norm = "".join(cond.lower().split())
                    if cond_norm in _BAD_CONDITION_KEYS:
                        cond = ""

                    to_id = None
                    to_type = None
                    if to_name in task_name_to_id_lc:
                        to_id = task_name_to_id_lc[to_name]
                        to_type = "task"
                    else:
                        # substring match (LLM 이름 표기 흔들림 흡수)
                        for tname_lc, tid in task_name_to_id_lc.items():
                            if tname_lc and (tname_lc in to_name or to_name in tname_lc):
                                to_id = tid
                                to_type = "task"
                                break
                    # task 매칭 실패 시 gateway 도 후보로 (게이트웨이 체이닝 케이스)
                    if not to_id:
                        for gw_name_lc, other_gw_id in gateway_name_to_id.items():
                            if other_gw_id == gw_id:
                                continue
                            if gw_name_lc and (gw_name_lc in to_name or to_name in gw_name_lc):
                                to_id = other_gw_id
                                to_type = "gateway"
                                break
                    if not to_id:
                        continue
                    if (gw_id, to_id) in existing_pairs:
                        continue
                    entities["sequence_flows"].append({
                        "from_id": gw_id,
                        "from_type": "gateway",
                        "to_id": to_id,
                        "to_type": to_type,
                        "condition": cond,
                    })
                    existing_pairs.add((gw_id, to_id))
                    augmented += 1

            if augmented:
                print(f"   🛟 gateway outgoing_branches 로부터 sequence_flow {augmented} 개 자동 보강")

        # Heuristic gateway synthesis:
        # If no gateways were extracted but a source task has multiple conditional outgoing flows,
        # synthesize an ExclusiveGateway to preserve branching semantics deterministically.
        if not entities["gateways"] and extracted.sequence_flows:
            # Build helper maps
            task_name_to_id = {t.name.lower().strip(): t.task_id for t in entities["tasks"] if t.name}
            task_id_to_proc = {t.task_id: t.process_id for t in entities["tasks"]}
            grouped = {}
            for flow in extracted.sequence_flows:
                from_name = str(flow.get("from_task") or "").lower().strip()
                to_name = str(flow.get("to_task") or "").lower().strip()
                cond = str(flow.get("condition") or "").strip()
                if not from_name or not to_name:
                    continue
                grouped.setdefault(from_name, []).append((to_name, cond))

            synthesized_pairs = set()
            for from_name, items in grouped.items():
                # Need at least 2 distinct targets for real branching
                uniq_targets = []
                for to_name, _ in items:
                    if to_name not in uniq_targets:
                        uniq_targets.append(to_name)
                if len(uniq_targets) < 2:
                    continue
                # At least one conditional cue should exist to treat this as a branch.
                if not any(str(cond or "").strip() for _, cond in items):
                    continue

                src_task_id = task_name_to_id.get(from_name)
                if not src_task_id:
                    continue
                proc_id = task_id_to_proc.get(src_task_id, "")
                gw_id = generate_id()
                gw_name = self._derive_gateway_name(
                    raw_name=f"{from_name} 분기",
                    description=f"{from_name}인지 여부를 판단한다.",
                    idx=(len(entities["gateways"]) + 1),
                )
                gw = Gateway(
                    gateway_id=gw_id,
                    process_id=proc_id,
                    name=gw_name,
                    gateway_type=GatewayType.EXCLUSIVE,
                    condition="",
                    description="조건 분기(자동 보강)"
                )
                entities["gateways"].append(gw)
                gateway_name_to_id[gw.name.lower()] = gw_id
                if chunk_id:
                    entities["entity_chunk_map"][gw_id] = chunk_id

                # Replace direct src->target conditional links with src->gw and gw->target(condition)
                entities["sequence_flows"] = [
                    sf for sf in entities["sequence_flows"]
                    if not (
                        (sf.get("from_id") == src_task_id or sf.get("from_task_id") == src_task_id)
                        and ((sf.get("to_id") in task_name_to_id.values()) or sf.get("to_task_id"))
                        and str(sf.get("condition") or "").strip()
                    )
                ]
                entities["sequence_flows"].append(
                    {
                        "from_id": src_task_id,
                        "from_type": "task",
                        "to_id": gw_id,
                        "to_type": "gateway",
                        "condition": "",
                    }
                )
                for to_name, cond in items:
                    tgt_task_id = task_name_to_id.get(to_name)
                    if not tgt_task_id:
                        continue
                    cond_norm = str(cond or "").strip()
                    key = (gw_id, tgt_task_id, cond_norm)
                    if key in synthesized_pairs:
                        continue
                    synthesized_pairs.add(key)
                    entities["sequence_flows"].append(
                        {
                            "from_id": gw_id,
                            "from_type": "gateway",
                            "to_id": tgt_task_id,
                            "to_type": "task",
                            "condition": cond_norm,
                        }
                    )

        # Normalize gateway branch conditions from the beginning.
        outgoing_by_gateway: dict[str, list[dict]] = {}
        for sf in entities["sequence_flows"]:
            if not isinstance(sf, dict):
                continue
            src = str(sf.get("from_id") or sf.get("from_task_id") or "").strip()
            if not src:
                continue
            outgoing_by_gateway.setdefault(src, []).append(sf)

        _bad_cond_norm = {
            "조건1", "조건2", "조건3", "조건4", "조건5",
            "분기1", "분기2", "분기3",
            "case1", "case2", "case3",
            "hastask", "hasinstruction", "performedby", "hasgateway", "hasevent",
        }
        for gw in entities["gateways"]:
            if not isinstance(gw, Gateway):
                continue
            if gw.gateway_type != GatewayType.EXCLUSIVE:
                continue
            outs = outgoing_by_gateway.get(str(gw.gateway_id)) or []
            if len(outs) < 2:
                continue

            for sf in outs:
                cond = str(sf.get("condition") or "").strip()
                cnorm = "".join(cond.lower().split())
                if cnorm in _bad_cond_norm:
                    sf["condition"] = ""

            if len(outs) == 2:
                c_true, c_false = self._derive_true_false_conditions(
                    gateway_name=str(gw.name or ""),
                    gateway_description=str(gw.description or ""),
                )
                if not str(outs[0].get("condition") or "").strip():
                    outs[0]["condition"] = c_true
                if not str(outs[1].get("condition") or "").strip():
                    outs[1]["condition"] = c_false

        # Also create sequence flows from next_task/previous_task attributes
        for task in entities["tasks"]:
            if hasattr(task, '_next_task_name') and task._next_task_name:
                next_name = task._next_task_name.lower()
                for other_task in entities["tasks"]:
                    if other_task.name.lower() == next_name or next_name in other_task.name.lower():
                        entities["sequence_flows"].append({
                            "from_task_id": task.task_id,
                            "to_task_id": other_task.task_id,
                            "condition": ""
                        })
                        break
        
        # Create default sequence flows based on order (within same process)
        # Group tasks by process
        tasks_by_process = {}
        for task in entities["tasks"]:
            proc_id = task.process_id or "default"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        # Create sequence flows for tasks without explicit flows
        existing_flows = set()
        for flow in entities["sequence_flows"]:
            # Handle both formats: from_task_id/to_task_id or from_id/to_id
            from_id = flow.get("from_task_id") or flow.get("from_id")
            to_id = flow.get("to_task_id") or flow.get("to_id")
            if from_id and to_id:
                existing_flows.add((from_id, to_id))
        
        for proc_id, proc_tasks in tasks_by_process.items():
            # Sort by order
            sorted_tasks = sorted(proc_tasks, key=lambda t: t.order)
            
            for i in range(len(sorted_tasks) - 1):
                from_task = sorted_tasks[i]
                to_task = sorted_tasks[i + 1]
                
                # Only add if not already exists
                if (from_task.task_id, to_task.task_id) not in existing_flows:
                    entities["sequence_flows"].append({
                        "from_task_id": from_task.task_id,
                        "to_task_id": to_task.task_id,
                        "condition": ""
                    })
        
        return entities
