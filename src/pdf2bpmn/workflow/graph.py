"""LangGraph workflow definition for PDF to BPMN conversion."""
import re
import time
from langgraph.graph import StateGraph, END

from ..models.state import GraphState
from ..models.entities import (
    Process, Task, Role, Gateway, Event, 
    Skill, DMNDecision, DMNRule, Evidence
)
from ..extractors.pdf_extractor import PDFExtractor
from ..extractors.entity_extractor import EntityExtractor
from ..graph.neo4j_client import Neo4jClient
from ..graph.vector_search import VectorSearch
from ..generators.bpmn_generator import BPMNGenerator
from ..generators.dmn_generator import DMNGenerator
from ..generators.skill_generator import SkillGenerator
from ..config import Config


# ---------------------------------------------------------------------------
# Role / Task 정규화용 휴리스틱 사전
# ---------------------------------------------------------------------------
# - 의미상 동일하지만 표현만 다른 Role / Task 가 별개로 추출되는 문제를 줄이기
#   위한 deterministic 한 규칙 기반 정규화.
# - LLM 임계값을 낮추면 잘못된 병합이 늘어나기 때문에, 문서 도메인을 가리지
#   않는 일반적인 한국어 절차서 어휘 위주로 매핑한다.
# ---------------------------------------------------------------------------

# Role 이름 정규화 시 제거할 노이즈 패턴 (괄호/구분기호 등).
# 직급 접미사는 의도적으로 보존하여 "기획처장" / "기획본부장" 등 직급이 다른
# 역할은 분리된 채로 두고, 책임 task signature 가 거의 동일한 경우에만 합친다.
# (사용자 피드백: "직급은 너무 다양하니까 추출된 대로 사용하자")
_ROLE_NOISE_PATTERNS: tuple[str, ...] = (
    r"\([^)]*\)",       # ()
    r"\[[^\]]*\]",      # []
    r"<[^>]*>",         # <>
    r"[·,/]",           # 구분 기호
)

# Task 동사 동의어 클래스: 같은 "행위 유형"으로 묶기 위한 사전
_TASK_VERB_SYNONYMS: dict[str, set[str]] = {
    "share":     {"제공", "제출", "송부", "배부", "발송", "통보", "전달", "안내", "회람", "공유"},
    "review":    {"검토", "심사", "심의", "검증", "확인", "점검", "확정"},
    "draft":     {"작성", "기안", "등록", "입력", "수정", "보완"},
    "approve":   {"승인", "결재", "결정", "의결", "표결", "재가"},
    "open":      {"개최", "진행", "주관", "운영", "실시"},
    "attend":    {"참석", "출석", "참여", "참가"},
    "explain":   {"설명", "발표", "보고", "진술", "공시"},
    "select":    {"선정", "지명", "위촉", "임명", "구성", "선임"},
    "release":   {"해촉", "해임", "면직", "제외", "철회", "해지"},
    "request":   {"요청", "신청", "의뢰", "요구", "건의"},
    "receive":   {"접수", "수령", "수신", "수집"},
    "pay":       {"지급", "정산", "환급", "납부", "결제"},
    "publish":   {"공고", "공시", "게시"},
    "register":  {"등록", "기록", "기재", "보관"},
    "preserve":  {"보관", "보존", "관리"},
}

# 위 동사 사전을 평탄화 (set-of-strings)
_TASK_ALL_VERB_TOKENS: set[str] = {v for vs in _TASK_VERB_SYNONYMS.values() for v in vs}

# Task 핵심 명사 추출 시 제거할 stop tokens (의미 약한 일반 어휘)
_TASK_NOUN_STOP_TOKENS: set[str] = {
    "및", "의", "을", "를", "이", "가", "에", "로", "으로", "와", "과",
    "에서", "부터", "까지", "또는", "혹은", "or", "and",
    "추가", "사전", "사후", "결과", "관련", "대상", "기타", "이상", "이하",
    "최종", "최초", "초안", "안", "여부", "수정", "변경", "내용", "사항",
    "단계", "절차", "건", "회", "차", "상정", "협조",
}


# Dedup 강도 프리셋. 사용자가 [도구 설정] 다이얼로그에서 선택한 값에 따라
# normalize_entities 안의 임계값들이 일괄 오버라이드된다.
#   - concise  : 임계 ↓ → 더 잘 합쳐짐 (프로세스 간소화)
#   - standard : 기본값 그대로 사용 (Config.* 값)
#   - detailed : 임계 ↑ → 거의 안 합쳐짐 (원문 절차 유지)
# 각 프리셋은 partial dict — 명시되지 않은 키는 Config 기본값을 그대로 사용한다.
_DEDUP_LEVEL_PROFILES: dict[str, dict[str, float]] = {
    "concise": {
        "TASK_SEMANTIC_COSINE_MIN": 0.78,
        "TASK_SEMANTIC_HIGH_COSINE": 0.86,
        "TASK_NOUN_JACCARD_MIN": 0.40,
        "ROLE_SEMANTIC_COSINE_MIN": 0.86,
        "TASK_SAME_NAME_INSTR_COSINE_SAME_ROLE": 0.65,
        "TASK_SAME_NAME_INSTR_COSINE_DIFF_ROLE": 0.72,
    },
    "standard": {},  # Config 기본값 사용
    "detailed": {
        "TASK_SEMANTIC_COSINE_MIN": 0.93,
        "TASK_SEMANTIC_HIGH_COSINE": 0.97,
        "TASK_NOUN_JACCARD_MIN": 0.80,
        "ROLE_SEMANTIC_COSINE_MIN": 0.97,
        "TASK_SAME_NAME_INSTR_COSINE_SAME_ROLE": 0.92,
        "TASK_SAME_NAME_INSTR_COSINE_DIFF_ROLE": 0.97,
    },
}


class PDF2BPMNWorkflow:
    """Orchestrates the PDF to BPMN conversion workflow."""
    
    def __init__(self, graph_name: str | None = None):
        self.pdf_extractor = PDFExtractor()
        self.entity_extractor = EntityExtractor()
        self.neo4j = Neo4jClient(graph_name=graph_name)
        self.vector_search = VectorSearch(self.neo4j)
        self.bpmn_generator = BPMNGenerator()
        self.dmn_generator = DMNGenerator()
        self.skill_generator = SkillGenerator()
        
        # Accumulated relationship maps
        self.task_role_map = {}  # task_id -> role_id
        self.task_process_map = {}  # task_id -> process_id
        self.role_decision_map = {}  # role_id -> [decision_ids]
        self.entity_chunk_map = {}  # entity_id -> chunk_id
        self.role_skill_map = {}  # role_id -> [skill_ids]
        self.sequence_flows = []  # list of {from_id, to_id, from_type, to_type, condition}
        self.all_gateways = []  # list of Gateway objects
        
        # Name -> ID mappings for lookup
        self.process_name_to_id = {}
        self.role_name_to_id = {}
        self.task_name_to_id = {}

        # 이름이 같은 task 페어의 instruction/description 임베딩 cosine 캐시.
        # key = sorted (task_id_a, task_id_b). value = float | None (None = 비교 불가).
        # 같은 워크플로우 인스턴스 안에서만 유효 (normalize 시작 시 reset).
        self._instr_sim_cache: dict[tuple[str, str], float | None] = {}

        # Dedup 강도 (사용자 선택값). agent_executor 가 set_dedup_level() 로 설정한다.
        # _dedup_overrides 는 _cfg() 가 Config 보다 우선해서 사용한다.
        self._dedup_level: str = "standard"
        self._dedup_overrides: dict[str, float] = {}

        # ----------------------------------------------------------------
        # Global task order tracking (회귀 방지: SOP segmentation 으로 인한 order 충돌)
        # ----------------------------------------------------------------
        # SOP segmentation 이 강화된 이후 (commit 5662a2e) 문서가 여러 section 으로
        # 분할되어 entity_extractor 가 section 단위로 호출된다. 각 section 안에서
        # task.order 가 독립적으로 1..N 으로 부여되므로, 다른 section 의 첫 task 와
        # order 가 충돌하여 시작 task 가 비결정적으로 뽑힌다.
        # → 각 task 마다 (section_index, section.content 내 첫 등장 offset) 를 기록하고,
        #   모든 section 처리 후 그 키로 글로벌 정렬하여 task.order 를 재할당한다.
        # key: task_id -> (section_index, byte_offset_in_section)
        self._task_global_order_key: dict[str, tuple[int, int]] = {}
    
    def ingest_pdf(self, state: GraphState) -> GraphState:
        """Node: Ingest PDF and extract document structure."""
        print("📄 Ingesting PDF documents...")
        
        documents = []
        sections = []
        chunks = []
        
        for pdf_path in state.get("pdf_paths", []):
            doc, doc_sections, doc_chunks = self.pdf_extractor.extract_document(pdf_path)
            documents.append(doc)
            sections.extend(doc_sections)
            chunks.extend(doc_chunks)
            
            # Store in Neo4j
            self.neo4j.create_document(doc)
            for section in doc_sections:
                self.neo4j.create_section(section)
        
        return {
            "documents": documents,
            "sections": sections,
            "reference_chunks": chunks,
            "current_step": "segment_sections"
        }
    
    def segment_sections(self, state: GraphState) -> GraphState:
        """Node: Process and embed sections."""
        print("📑 Segmenting and embedding sections...")
        
        chunks = state.get("reference_chunks", [])
        
        # Batch embed chunks (in smaller batches to avoid rate limits)
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            self.vector_search.batch_embed_chunks(batch)
            
            # Store in Neo4j and link to document
            for chunk in batch:
                self.neo4j.create_chunk(chunk)
                if chunk.doc_id:
                    self.neo4j.link_chunk_to_document(chunk.chunk_id, chunk.doc_id)
        
        return {
            "reference_chunks": chunks,
            "current_step": "extract_candidates"
        }
    
    def _record_task_global_order_keys(
        self,
        *,
        section_index: int,
        section_content: str,
        new_tasks: list,
    ) -> None:
        """각 task 의 (section_index, section.content 내 첫 등장 offset) 을 기록.

        - section.content 안에서 task.name 이 처음 등장하는 위치를 정렬 키로 사용한다.
        - LLM 이 section 내부에서 잘못된 order 를 부여해도, source text 위치가 truth source.
        - 같은 task 이름이 여러 section 에 나타나면 normalize 단계에서 merge 되므로
          여기서는 별도 처리 없이 첫 등장 위치만 저장한다.
        """
        if not new_tasks:
            return
        content_low = str(section_content or "").lower()
        for t in new_tasks:
            tid = getattr(t, "task_id", None)
            if not tid:
                continue
            if tid in self._task_global_order_key:
                continue
            name = str(getattr(t, "name", "") or "").strip().lower()
            pos = -1
            if name and content_low:
                pos = content_low.find(name)
                if pos < 0:
                    # 첫 단어만 잡아도 의미상 충분 (소수의 token 차이 보정)
                    first_tok = name.split()[0] if name.split() else ""
                    if first_tok and len(first_tok) >= 2:
                        pos = content_low.find(first_tok)
            if pos < 0:
                pos = 10**9
            self._task_global_order_key[tid] = (section_index, pos)

    def _reassign_global_task_order(self, all_tasks: list) -> None:
        """모든 section 처리 후 task.order 를 재할당.

        1순위: 추출된 sequence_flows 의 위상(topological) 순서.
          - source text 등장 위치만 쓰면, 문서 개요/요약이 후행 단계를 먼저 언급할 때
            (예: '최종 승인 면담' 이 도입부에 언급) 종결 단계가 order 1 로 잘못 잡힌다.
          - flow 그래프에서 in-degree 0 후보가 여럿이면 reach_count(그 노드에서 도달
            가능한 task 수)가 큰 것을 실제 시작점으로 본다. text 위치는 동률 tie-break.
        2순위/fallback: flows 가 빈약하면 (section_index, section 내 첫 등장 offset).
        """
        if not all_tasks:
            return

        def _text_key(t) -> tuple:
            tid = getattr(t, "task_id", None) or ""
            return self._task_global_order_key.get(tid, (10**9, 10**9))

        task_ids = {getattr(t, "task_id", None) for t in all_tasks
                    if getattr(t, "task_id", None)}
        flows = self.sequence_flows if isinstance(self.sequence_flows, list) else []
        use_topology = bool(task_ids) and len(flows) >= max(2, len(task_ids) // 4)

        ordered: list = []
        if use_topology:
            # --- flow DAG 빌드 (task/gateway/event 모든 노드 포함; reachability 용) ---
            adjacency: dict = {}
            in_degree: dict = {}
            all_nodes: set = set(task_ids)

            def _fid(f, *keys) -> str:
                for k in keys:
                    v = f.get(k) if isinstance(f, dict) else None
                    if v:
                        return str(v).strip()
                return ""

            for f in flows:
                if not isinstance(f, dict):
                    continue
                s = _fid(f, "source", "from_id", "from_task_id")
                d = _fid(f, "target", "to_id", "to_task_id")
                if not s or not d or s == d:
                    continue
                all_nodes.add(s)
                all_nodes.add(d)
                adjacency.setdefault(s, []).append(d)
                in_degree[d] = in_degree.get(d, 0) + 1
                in_degree.setdefault(s, in_degree.get(s, 0))
            for n in all_nodes:
                in_degree.setdefault(n, 0)

            reach_cache: dict = {}

            def _reach(start: str) -> int:
                if start in reach_cache:
                    return reach_cache[start]
                seen = {start}
                stack = [start]
                while stack:
                    cur = stack.pop()
                    for nxt in adjacency.get(cur, []):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                cnt = sum(1 for v in seen if v in task_ids)
                reach_cache[start] = cnt
                return cnt

            task_obj_by_id = {getattr(t, "task_id", None): t for t in all_tasks
                              if getattr(t, "task_id", None)}
            orig_index = {getattr(t, "task_id", None): i
                          for i, t in enumerate(all_tasks)}

            def _cand_key(nid: str, succ_of_last: set) -> tuple:
                # 1) 직전 처리 노드의 후행 우선(체인 연속성 — 끊긴 sub-chain 으로 점프 방지)
                # 2) task 우선 → 3) reach 큰 것(실제 시작점) → 4) text 위치 → 5) 원래 인덱스
                tk = _text_key(task_obj_by_id.get(nid)) if nid in task_ids else (10**9, 10**9)
                return (
                    0 if nid in succ_of_last else 1,
                    0 if nid in task_ids else 1,
                    -_reach(nid),
                    tk,
                    orig_index.get(nid, 10**9),
                    str(nid),
                )

            # Kahn — in-degree 0 후보를 우선순위로 선택
            remaining = dict(in_degree)
            seen_tasks: set = set()
            last_picked = None
            guard = len(remaining) + 8
            while remaining and guard > 0:
                guard -= 1
                zero = [n for n, d in remaining.items() if d == 0]
                if not zero:
                    zero = list(remaining.keys())   # cycle — 강제 진행
                succ_of_last = set(adjacency.get(last_picked, [])) if last_picked else set()
                zero.sort(key=lambda n: _cand_key(n, succ_of_last))
                chosen = zero[0]
                for nxt in adjacency.get(chosen, []):
                    if nxt in remaining:
                        remaining[nxt] = max(0, remaining[nxt] - 1)
                remaining.pop(chosen, None)
                last_picked = chosen
                if chosen in task_ids and chosen not in seen_tasks:
                    seen_tasks.add(chosen)
                    obj = task_obj_by_id.get(chosen)
                    if obj is not None:
                        ordered.append(obj)
            # flow 에 한 번도 안 나타난 잔여 task 는 text 위치 순으로 뒤에 붙임
            leftovers = [t for t in all_tasks
                         if getattr(t, "task_id", None) not in seen_tasks]
            leftovers.sort(key=_text_key)
            ordered.extend(leftovers)
        else:
            # flows 빈약 → 기존 방식(텍스트 위치 기반)
            indexed = [(_text_key(t), i, t) for i, t in enumerate(all_tasks)]
            indexed.sort(key=lambda x: (x[0], x[1]))
            ordered = [t for _, _, t in indexed]

        # order 부여 (동일 이름 task 는 같은 order 공유 — 기존 동작 유지)
        name_to_assigned_order: dict[str, int] = {}
        next_order = 1
        for t in ordered:
            nm = str(getattr(t, "name", "") or "").strip().lower()
            if nm and nm in name_to_assigned_order:
                t.order = name_to_assigned_order[nm]
                continue
            t.order = next_order
            if nm:
                name_to_assigned_order[nm] = next_order
            next_order += 1
        try:
            head_names = [
                f"{getattr(t, 'order', '?')}.{getattr(t, 'name', '?')!r}"
                for t in ordered[:6]
            ]
            print(
                f"   📊 [GLOBAL-ORDER] reassigned task.order for {len(all_tasks)} tasks "
                f"(flow-topology={'on' if use_topology else 'off'}). head6={head_names}"
            )
        except Exception:
            pass

    def extract_candidates(self, state: GraphState) -> GraphState:
        """Node: Extract process/task/role candidates from sections."""
        print("🔍 Extracting candidate entities...")
        
        all_processes = []
        all_tasks = []
        all_roles = []
        all_gateways = []
        all_events = []
        all_decisions = []
        all_rules = []
        all_skills = []
        
        sections = state.get("sections", [])
        chunks = state.get("reference_chunks", [])
        is_multi_doc = len({s.doc_id for s in sections if getattr(s, "doc_id", None)}) > 1
        
        # Create chunk index for linking
        chunk_by_doc_page = {}
        for chunk in chunks:
            key = (chunk.doc_id, chunk.page)
            if key not in chunk_by_doc_page:
                chunk_by_doc_page[key] = []
            chunk_by_doc_page[key].append(chunk)

        # 문서별 엔티티 컨텍스트 (다중 파일일 때 서로 다른 문서 간 과도한 병합 방지)
        doc_process_name_to_id = {}
        doc_role_name_to_id = {}

        # SOP segmentation 으로 section 순서가 비결정적으로 반환될 수 있어
        # page_from(보조: page_to) 기준으로 안정 정렬한 뒤 순회한다.
        sections_sorted = sorted(
            sections,
            key=lambda s: (
                int(getattr(s, "page_from", 0) or 0),
                int(getattr(s, "page_to", 0) or 0),
                str(getattr(s, "section_id", "") or ""),
            ),
        )

        for section_index, section in enumerate(sections_sorted):
            if not section.content or len(section.content.strip()) < 50:
                continue
            
            # Find relevant chunk for this section (for evidence linking)
            section_chunk_id = ""
            section_doc_id = section.doc_id
            section_key = (section_doc_id, section.page_from)
            if section_key in chunk_by_doc_page and chunk_by_doc_page[section_key]:
                section_chunk_id = chunk_by_doc_page[section_key][0].chunk_id
            
            # Extract entities from section content with existing context
            # 기존 프로세스/역할/태스크 목록을 LLM에 전달하여 동일 엔티티 식별 개선
            if is_multi_doc:
                existing_process_names = list(doc_process_name_to_id.get(section_doc_id, {}).keys())
                existing_role_names = list(doc_role_name_to_id.get(section_doc_id, {}).keys())
            else:
                existing_process_names = list(self.process_name_to_id.keys())
                existing_role_names = list(self.role_name_to_id.keys())
            
            # 기존 태스크 정보 수집 (이름, 역할, 프로세스)
            existing_tasks_info = []
            for task in all_tasks:
                task_info = {"name": task.name, "order": task.order}
                # 태스크의 역할 찾기
                if task.task_id in self.task_role_map:
                    role_id = self.task_role_map[task.task_id]
                    for role_name, rid in self.role_name_to_id.items():
                        if rid == role_id:
                            task_info["role"] = role_name
                            break
                # 태스크의 프로세스 찾기
                if task.process_id:
                    for proc_name, pid in self.process_name_to_id.items():
                        if pid == task.process_id:
                            task_info["process"] = proc_name
                            break
                existing_tasks_info.append(task_info)
            
            extracted = self.entity_extractor.extract_from_text(
                section.content,
                existing_processes=existing_process_names,
                existing_roles=existing_role_names,
                existing_tasks=existing_tasks_info,
            )
            
            # Convert to entity objects with relationships
            entities = self.entity_extractor.convert_to_entities(
                extracted, 
                section_doc_id,
                chunk_id=section_chunk_id,
                existing_processes=(doc_process_name_to_id.get(section_doc_id, {}) if is_multi_doc else self.process_name_to_id),
                existing_roles=(doc_role_name_to_id.get(section_doc_id, {}) if is_multi_doc else self.role_name_to_id)
            )
            
            # Collect entities
            all_processes.extend(entities["processes"])
            new_tasks_for_section = entities.get("tasks") or []
            all_tasks.extend(new_tasks_for_section)
            # 각 task 의 글로벌 정렬 키 (section_index, offset_in_section_content) 기록
            self._record_task_global_order_keys(
                section_index=section_index,
                section_content=section.content,
                new_tasks=new_tasks_for_section,
            )
            all_roles.extend(entities["roles"])
            all_gateways.extend(entities["gateways"])
            all_events.extend(entities["events"])
            all_decisions.extend(entities["decisions"])
            all_rules.extend(entities["rules"])
            all_skills.extend(entities.get("skills", []))
            
            # Accumulate relationship mappings
            self.task_role_map.update(entities.get("task_role_map", {}))
            self.task_process_map.update(entities.get("task_process_map", {}))
            self.entity_chunk_map.update(entities.get("entity_chunk_map", {}))
            
            # Accumulate sequence flows
            self.sequence_flows.extend(entities.get("sequence_flows", []))
            
            for role_id, decision_ids in entities.get("role_decision_map", {}).items():
                if role_id not in self.role_decision_map:
                    self.role_decision_map[role_id] = []
                self.role_decision_map[role_id].extend(decision_ids)

            for role_id, skill_ids in entities.get("role_skill_map", {}).items():
                if role_id not in self.role_skill_map:
                    self.role_skill_map[role_id] = []
                self.role_skill_map[role_id].extend(skill_ids)
            
            # Update name -> ID mappings
            for proc in entities["processes"]:
                self.process_name_to_id[proc.name.lower()] = proc.proc_id
                if section_doc_id not in doc_process_name_to_id:
                    doc_process_name_to_id[section_doc_id] = {}
                doc_process_name_to_id[section_doc_id][proc.name.lower()] = proc.proc_id
            for role in entities["roles"]:
                self.role_name_to_id[role.name.lower()] = role.role_id
                if section_doc_id not in doc_role_name_to_id:
                    doc_role_name_to_id[section_doc_id] = {}
                doc_role_name_to_id[section_doc_id][role.name.lower()] = role.role_id
            for task in entities["tasks"]:
                self.task_name_to_id[task.name.lower()] = task.task_id

        # SOP segmentation 으로 section 단위 호출이 끝난 후, 글로벌 task.order 재할당.
        # 각 task 의 section 내 첫 등장 offset 으로 정렬 → BPMN 흐름이 source text 와 일치.
        self._reassign_global_task_order(all_tasks)

        return {
            "processes": all_processes,
            "tasks": all_tasks,
            "roles": all_roles,
            "gateways": all_gateways,
            "events": all_events,
            "skills": all_skills,
            "dmn_decisions": all_decisions,
            "dmn_rules": all_rules,
            "current_step": "normalize_entities"
        }
    
    def extract_candidates_with_progress(self, state: GraphState, progress_callback=None) -> GraphState:
        """Extract candidates with progress callback for frontend updates."""
        print("🔍 Extracting candidate entities with progress...")
        
        all_processes = []
        all_tasks = []
        all_roles = []
        all_gateways = []
        all_events = []
        all_decisions = []
        all_rules = []
        all_skills = []
        
        sections = state.get("sections", [])
        chunks = state.get("reference_chunks", [])
        is_multi_doc = len({s.doc_id for s in sections if getattr(s, "doc_id", None)}) > 1
        
        # Filter valid sections + SOP segmentation 출력 순서 안정화 (page_from 기준)
        valid_sections = [s for s in sections if s.content and len(s.content.strip()) >= 50]
        valid_sections.sort(
            key=lambda s: (
                int(getattr(s, "page_from", 0) or 0),
                int(getattr(s, "page_to", 0) or 0),
                str(getattr(s, "section_id", "") or ""),
            )
        )
        total_sections = len(valid_sections)
        
        # Create chunk index for linking
        chunk_by_doc_page = {}
        for chunk in chunks:
            key = (chunk.doc_id, chunk.page)
            if key not in chunk_by_doc_page:
                chunk_by_doc_page[key] = []
            chunk_by_doc_page[key].append(chunk)

        # 문서별 엔티티 컨텍스트 (다중 파일일 때 서로 다른 문서 간 과도한 병합 방지)
        doc_process_name_to_id = {}
        doc_role_name_to_id = {}
        
        for i, section in enumerate(valid_sections):
            # Report progress
            if progress_callback:
                section_preview = section.content[:50].replace('\n', ' ')
                progress_callback(
                    i + 1, 
                    total_sections, 
                    f"섹션 {i+1}/{total_sections} LLM 분석 중: {section_preview}..."
                )
            
            # Find relevant chunk for this section
            section_chunk_id = ""
            section_doc_id = section.doc_id
            section_key = (section_doc_id, section.page_from)
            if section_key in chunk_by_doc_page and chunk_by_doc_page[section_key]:
                section_chunk_id = chunk_by_doc_page[section_key][0].chunk_id
            
            # Extract entities with existing context (프로세스, 역할, 태스크 모두 포함)
            if is_multi_doc:
                existing_process_names = list(doc_process_name_to_id.get(section_doc_id, {}).keys())
                existing_role_names = list(doc_role_name_to_id.get(section_doc_id, {}).keys())
            else:
                existing_process_names = list(self.process_name_to_id.keys())
                existing_role_names = list(self.role_name_to_id.keys())
            
            # 기존 태스크 정보 수집 (이름, 역할, 프로세스)
            existing_tasks_info = []
            for task in all_tasks:
                task_info = {"name": task.name, "order": task.order}
                # 태스크의 역할 찾기
                if task.task_id in self.task_role_map:
                    role_id = self.task_role_map[task.task_id]
                    for role_name, rid in self.role_name_to_id.items():
                        if rid == role_id:
                            task_info["role"] = role_name
                            break
                # 태스크의 프로세스 찾기
                if task.process_id:
                    for proc_name, pid in self.process_name_to_id.items():
                        if pid == task.process_id:
                            task_info["process"] = proc_name
                            break
                existing_tasks_info.append(task_info)
            
            try:
                t_extract0 = time.perf_counter()
                extracted = self.entity_extractor.extract_from_text(
                    section.content,
                    existing_processes=existing_process_names,
                    existing_roles=existing_role_names,
                    existing_tasks=existing_tasks_info,
                )
                t_extract_ms = int((time.perf_counter() - t_extract0) * 1000)
                
                # Convert to entity objects
                t_convert0 = time.perf_counter()
                entities = self.entity_extractor.convert_to_entities(
                    extracted, 
                    section_doc_id,
                    chunk_id=section_chunk_id,
                    existing_processes=(doc_process_name_to_id.get(section_doc_id, {}) if is_multi_doc else self.process_name_to_id),
                    existing_roles=(doc_role_name_to_id.get(section_doc_id, {}) if is_multi_doc else self.role_name_to_id)
                )
                t_convert_ms = int((time.perf_counter() - t_convert0) * 1000)
                print(
                    f"   ⏱️ [SECTION-TIMING] {i+1}/{total_sections} "
                    f"extract={t_extract_ms}ms convert={t_convert_ms}ms "
                    f"text_len={len(section.content or '')} "
                    f"entities(tasks={len(entities.get('tasks') or [])}, roles={len(entities.get('roles') or [])}, "
                    f"gateways={len(entities.get('gateways') or [])}, flows={len(entities.get('sequence_flows') or [])}, "
                    f"decisions={len(entities.get('decisions') or [])}, rules={len(entities.get('rules') or [])})"
                )
                
                # Collect entities
                all_processes.extend(entities["processes"])
                new_tasks_for_section = entities.get("tasks") or []
                all_tasks.extend(new_tasks_for_section)
                # 각 task 의 글로벌 정렬 키 (section_index, offset_in_section_content) 기록
                self._record_task_global_order_keys(
                    section_index=i,
                    section_content=section.content,
                    new_tasks=new_tasks_for_section,
                )
                all_roles.extend(entities["roles"])
                all_gateways.extend(entities["gateways"])
                all_events.extend(entities["events"])
                all_decisions.extend(entities["decisions"])
                all_rules.extend(entities["rules"])
                all_skills.extend(entities.get("skills", []))
                
                # Accumulate mappings
                self.task_role_map.update(entities.get("task_role_map", {}))
                self.task_process_map.update(entities.get("task_process_map", {}))
                self.entity_chunk_map.update(entities.get("entity_chunk_map", {}))
                self.sequence_flows.extend(entities.get("sequence_flows", []))
                
                for role_id, decision_ids in entities.get("role_decision_map", {}).items():
                    if role_id not in self.role_decision_map:
                        self.role_decision_map[role_id] = []
                    self.role_decision_map[role_id].extend(decision_ids)

                for role_id, skill_ids in entities.get("role_skill_map", {}).items():
                    if role_id not in self.role_skill_map:
                        self.role_skill_map[role_id] = []
                    self.role_skill_map[role_id].extend(skill_ids)
                
                # Update name mappings
                for proc in entities["processes"]:
                    self.process_name_to_id[proc.name.lower()] = proc.proc_id
                    if section_doc_id not in doc_process_name_to_id:
                        doc_process_name_to_id[section_doc_id] = {}
                    doc_process_name_to_id[section_doc_id][proc.name.lower()] = proc.proc_id
                for role in entities["roles"]:
                    self.role_name_to_id[role.name.lower()] = role.role_id
                    if section_doc_id not in doc_role_name_to_id:
                        doc_role_name_to_id[section_doc_id] = {}
                    doc_role_name_to_id[section_doc_id][role.name.lower()] = role.role_id
                for task in entities["tasks"]:
                    self.task_name_to_id[task.name.lower()] = task.task_id
                    
            except Exception as e:
                print(f"   ⚠️ 청크 {i+1} 처리 중 오류: {e}")
                continue

        # SOP segmentation 으로 section 단위 호출이 끝난 후 글로벌 task.order 재할당
        self._reassign_global_task_order(all_tasks)

        return {
            "processes": all_processes,
            "tasks": all_tasks,
            "roles": all_roles,
            "gateways": all_gateways,
            "events": all_events,
            "skills": all_skills,
            "dmn_decisions": all_decisions,
            "dmn_rules": all_rules,
            "current_step": "normalize_entities"
        }
    
    def normalize_entities(self, state: GraphState) -> GraphState:
        """Node: Normalize and deduplicate entities using vector search."""
        print("🔄 Normalizing and deduplicating entities...")

        # 정규화 시작마다 instruction-similarity 캐시 초기화.
        # (재호출/재실행 시 stale 비교가 누적되지 않도록)
        self._instr_sim_cache = {}

        processes = state.get("processes", [])
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        gateways = state.get("gateways", [])
        events = state.get("events", [])
        decisions = state.get("dmn_decisions", [])
        skills = state.get("skills", [])
        
        # 1. 먼저 프로세스 병합 (이름 + 내용 유사도 기반)
        unique_processes, process_id_mapping = self._merge_duplicate_processes(
            processes,
            tasks,
            roles,
            state.get("reference_chunks", []),
        )
        
        # 2. 태스크의 process_id를 병합된 프로세스로 업데이트
        tasks = self._update_task_process_ids(tasks, process_id_mapping)
        
        # 3. 게이트웨이, 이벤트의 process_id도 업데이트
        gateways = self._update_entity_process_ids(gateways, process_id_mapping, "gateway_id")
        events = self._update_entity_process_ids(events, process_id_mapping, "event_id")

        # 3.5 미연결 게이트웨이 제거 (gateway 정규화).
        #   normalize 는 tasks/roles/decisions/skills 만 중복 제거하고 gateway 는
        #   다루지 않았다. 추출/섹션 병합 후에도 어떤 sequence_flow 에도 연결되지 않은
        #   고립 게이트웨이(같은 의사결정의 빈 사본 등)가 남으면 BPMN 흐름을 어지럽히고,
        #   process-definition 생성의 fallback 판정도 교란한다. 여기서 정리한다.
        if gateways:
            wired_gw_ids: set = set()
            for sf in (self.sequence_flows or []):
                if not isinstance(sf, dict):
                    continue
                for k in ("from_id", "to_id", "from_task_id", "to_task_id"):
                    v = str(sf.get(k) or "").strip()
                    if v:
                        wired_gw_ids.add(v)
            gw_before = len(gateways)
            gateways = [
                g for g in gateways
                if str(getattr(g, "gateway_id", "") or "") in wired_gw_ids
            ]
            if len(gateways) != gw_before:
                print(f"   🧹 Gateways: {gw_before} → {len(gateways)} (미연결 게이트웨이 제거)")

        # 4. task_process_map도 업데이트
        self._update_task_process_map(process_id_mapping)

        # 5. 유사한 태스크 먼저 병합 (의미적 중복 제거)
        # - 동사 동의어 클래스 + 핵심 명사 overlap 기반 의미 비교
        # - role 경계 안/밖 모두에서 정규화
        tasks, task_id_mapping = self._merge_similar_tasks(tasks)
        print(f"   Tasks after merge: {len(tasks)}")

        # 5.5 ROLE 정규화 (직급은 보존, 이름 + 책임 task signature 기반)
        # - 직급 접미사를 인위로 제거하지 않음 ("기획처장" / "기획본부장" 은 분리 유지)
        # - 정규화된 task signature 가 거의 동일한 role 들만 같은 역할로 흡수
        # - task_role_map / role_decision_map / role_skill_map 도 동시에 갱신
        roles_before = len(roles)
        roles, role_id_remap = self._normalize_roles_by_responsibility(roles)
        if role_id_remap:
            print(
                f"   🔀 Role normalize (by responsibility): {roles_before} → {len(roles)} "
                f"(remapped {len(role_id_remap)} ids)"
            )

        # Deduplicate tasks
        unique_tasks = self._deduplicate_entities(tasks, "Task")

        # Deduplicate roles (이름 정확 일치 - 이미 위에서 의미 클러스터링 완료)
        unique_roles = self._deduplicate_entities(roles, "Role")
        
        # Deduplicate decisions
        unique_decisions = self._deduplicate_entities(decisions, "Decision")
        unique_skills = self._deduplicate_entities(skills, "Skill")
        
        print(f"   Processes: {len(processes)} → {len(unique_processes)} (merged {len(processes) - len(unique_processes)})")
        print(f"   Tasks: {len(tasks)} → {len(unique_tasks)}")
        print(f"   Roles: {len(roles)} → {len(unique_roles)}")
        print(f"   Decisions: {len(decisions)} → {len(unique_decisions)}")
        print(f"   Skills: {len(skills)} → {len(unique_skills)}")
        print(f"   Gateways: {len(gateways)} (미연결 제거 후)")

        # task 0 role 통계 출력 (위원 후보 풀 / 자문 풀 같은 비활성 role 식별).
        # task 가 1개도 할당되지 않은 role 은 BPMN 의 lane 으로 사용해도 빈 swimlane 만
        # 만들기 때문에, 시각화/BPMN 생성 단계에서 별도 처리 후보가 된다.
        role_task_counts: dict[str, int] = {}
        for rid in self.task_role_map.values():
            if rid:
                role_task_counts[rid] = role_task_counts.get(rid, 0) + 1
        idle_roles = [
            r for r in unique_roles
            if role_task_counts.get(getattr(r, "role_id", ""), 0) == 0
        ]
        if idle_roles:
            idle_names = [getattr(r, "name", "") for r in idle_roles]
            print(
                f"   ⚠️ Task 0 role {len(idle_roles)}/{len(unique_roles)}개 (위원 후보/자문 풀 가능성): "
                f"{idle_names}"
            )
        
        # Store in Neo4j
        for proc in unique_processes:
            self.neo4j.create_process(proc)
        
        for task in unique_tasks:
            self.neo4j.create_task(task)
        
        for role in unique_roles:
            self.neo4j.create_role(role)
        
        for gateway in gateways:
            self.neo4j.create_gateway(gateway)
        
        for event in events:
            self.neo4j.create_event(event)
        
        for decision in unique_decisions:
            self.neo4j.create_decision(decision)

        for skill in unique_skills:
            self.neo4j.create_skill(skill)
        
        for rule in state.get("dmn_rules", []):
            self.neo4j.create_rule(rule)
        
        # Create relationships in batch
        print("🔗 Creating entity relationships...")
        # Only create evidence links if not disabled
        evidence_map = {} if Config.EVIDENCE_MODE == "off" else self.entity_chunk_map
        self.neo4j.create_all_relationships(
            task_role_map=self.task_role_map,
            task_process_map=self.task_process_map,
            role_decision_map=self.role_decision_map,
            entity_chunk_map=evidence_map,
            role_skill_map=self.role_skill_map,
        )
        
        # Store gateways for sequence flow creation
        self.all_gateways = gateways
        
        # Create sequence flows (NEXT relationships between tasks/gateways)
        print("➡️ Creating sequence flows...")
        self._create_sequence_flows(unique_tasks, unique_processes)
        
        # Infer missing Task-Role relationships based on name matching
        self._infer_task_role_relationships(unique_tasks, unique_roles)
        
        # Infer missing Task-Process relationships 
        self._infer_task_process_relationships(unique_tasks, unique_processes)
        
        return {
            "processes": unique_processes,
            "tasks": unique_tasks,
            "roles": unique_roles,
            "skills": unique_skills,
            "dmn_decisions": unique_decisions,
            "current_step": "generate_skills"
        }
    
    def _create_sequence_flows(self, tasks: list, processes: list):
        """Create NEXT relationships between tasks/gateways based on extracted and inferred sequence flows."""
        created_flows = set()
        
        # Build ID sets for validation
        task_ids = {t.task_id for t in tasks}
        gateway_ids = {g.gateway_id for g in self.all_gateways} if hasattr(self, 'all_gateways') else set()
        
        # First, create explicit sequence flows from extraction
        for flow in self.sequence_flows:
            # Support both old format (from_task_id) and new format (from_id)
            from_id = flow.get("from_id") or flow.get("from_task_id")
            to_id = flow.get("to_id") or flow.get("to_task_id")
            from_type = flow.get("from_type", "task")
            to_type = flow.get("to_type", "task")
            condition = flow.get("condition", "") or ""
            
            if from_id and to_id and (from_id, to_id) not in created_flows:
                # Create the appropriate relationship based on types
                if from_type == "gateway" and to_type == "task":
                    self.neo4j.link_gateway_to_task(from_id, to_id, condition)
                elif from_type == "task" and to_type == "gateway":
                    self.neo4j.link_task_to_gateway(from_id, to_id)
                else:
                    # Task to Task
                    self.neo4j.link_task_sequence(from_id, to_id, condition)
                
                created_flows.add((from_id, to_id))
                
                if condition:
                    print(f"   ✓ Flow with condition: {from_type}:{from_id[:8]} → {to_type}:{to_id[:8]} [{condition}]")
        
        # Group tasks by process
        tasks_by_process = {}
        for task in tasks:
            proc_id = task.process_id or "default"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        # Create sequence flows for each process based on task order.
        # ----------------------------------------------------------------
        # CRITICAL (regression guard):
        # 추출 LLM 이 task 에 부여한 `order` 가 잘못되면 (예: 종결 task 인 "최종 결과 통보"가
        # order=1 로 부여) task_order 순 자동 chain 이 거꾸로 된 NEXT 관계를 Neo4j 에
        # 영구 저장하여, 이후 generator/validator 가 그 가짜 관계를 sequence 로 사용 →
        # BPMN 의 시작 task 가 뒤바뀌는 회귀가 발생한다.
        # 이를 막기 위해 LLM 이 명시적으로 추출한 sequence_flows 가 task 수 절반 이상이면
        # task_order 기반 자동 chain 생성 자체를 건너뛴다. (LLM 이 이미 흐름을 충분히
        # 명시했다고 신뢰)
        # ----------------------------------------------------------------
        total_tasks = sum(len(v) for v in tasks_by_process.values())
        explicit_count = len(created_flows)
        auto_chain_threshold = max(1, total_tasks // 2)
        skip_auto_chain = explicit_count >= auto_chain_threshold

        if skip_auto_chain:
            print(
                f"   ⏭️ Skip task_order auto-chain: explicit_flows={explicit_count}, "
                f"tasks={total_tasks} (threshold={auto_chain_threshold}). "
                "LLM 이 흐름을 명시했으므로 task_order 기반 가짜 NEXT 를 만들지 않음."
            )
        else:
            for proc_id, proc_tasks in tasks_by_process.items():
                sorted_tasks = sorted(proc_tasks, key=lambda t: t.order)
                for i in range(len(sorted_tasks) - 1):
                    from_task = sorted_tasks[i]
                    to_task = sorted_tasks[i + 1]
                    if (from_task.task_id, to_task.task_id) not in created_flows:
                        self.neo4j.link_task_sequence(from_task.task_id, to_task.task_id)
                        created_flows.add((from_task.task_id, to_task.task_id))

        # Also use Neo4j to create sequences for each process — same defensive gate.
        # create_task_sequence_for_process 도 task_order 순으로 NEXT 를 일괄 생성하므로
        # 동일하게 가짜 chain 의 원인이 될 수 있다.
        if not skip_auto_chain:
            for proc in processes:
                self.neo4j.create_task_sequence_for_process(proc.proc_id)

        print(f"   Created {len(created_flows)} sequence flows (NEXT relationships)")
    
    def _infer_task_role_relationships(self, tasks: list, roles: list):
        """Infer Task-Role relationships based on task descriptions."""
        role_keywords = {}
        for role in roles:
            keywords = [role.name.lower()]
            if role.org_unit:
                keywords.append(role.org_unit.lower())
            role_keywords[role.role_id] = keywords
        
        for task in tasks:
            if task.task_id in self.task_role_map:
                continue  # Already has a role
            
            task_text = (task.name + " " + task.description + " " + (getattr(task, "instruction", "") or "")).lower()
            
            for role_id, keywords in role_keywords.items():
                for keyword in keywords:
                    if keyword in task_text and len(keyword) > 2:
                        self.neo4j.link_task_to_role(task.task_id, role_id)
                        self.task_role_map[task.task_id] = role_id
                        break
                if task.task_id in self.task_role_map:
                    break
    
    def _infer_task_process_relationships(self, tasks: list, processes: list):
        """Ensure all tasks are linked to a process."""
        if not processes:
            return
        
        default_process_id = processes[0].proc_id
        
        for task in tasks:
            if not task.process_id:
                task.process_id = default_process_id
                self.neo4j.link_task_to_process(task.task_id, default_process_id)
    
    def _normalize_process_text(self, value: str) -> str:
        """Normalize text for process similarity scoring."""
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    def _tokenize_process_text(self, value: str) -> set[str]:
        """Tokenize process text to simple normalized terms."""
        norm = self._normalize_process_text(value)
        if not norm:
            return set()
        parts = re.split(r"[^0-9a-zA-Z가-힣]+", norm)
        tokens = {p for p in parts if len(p) >= 2}
        # Add compact form to reduce whitespace variation impact.
        compact = re.sub(r"\s+", "", norm)
        if len(compact) >= 3:
            tokens.add(compact)
        return tokens

    def _char_ngrams(self, value: str, n: int = 3) -> set[str]:
        """Build character n-gram set for fuzzy narrative similarity."""
        norm = re.sub(r"\s+", "", self._normalize_process_text(value))
        if len(norm) < n:
            return {norm} if norm else set()
        return {norm[i:i + n] for i in range(len(norm) - n + 1)}

    def _extract_business_keys(self, value: str) -> set[str]:
        """
        Extract business keys from text (e.g., RQ_BP_XXXX, AA_BB_1234).
        This uses only content text, not file names.
        """
        text = (value or "").upper()
        keys = set()
        patterns = [
            r"\b[A-Z]{1,8}_[A-Z]{1,8}_[A-Z0-9]{2,}\b",
            r"\b[A-Z]{1,8}_BP_[A-Z0-9]{2,}\b",
            r"\bBP_[A-Z0-9]{2,}\b",
        ]
        for pattern in patterns:
            for m in re.findall(pattern, text):
                keys.add(m.strip())
        return keys

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        """Compute Jaccard similarity for two sets."""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        return inter / union if union else 0.0

    def _build_process_signature(
        self,
        proc,
        tasks: list,
        role_id_to_name: dict[str, str],
        source_text: str = "",
    ) -> dict:
        """Build content signature for process similarity."""
        proc_tasks = [t for t in tasks if getattr(t, "process_id", "") == proc.proc_id]
        task_name_tokens = set()
        role_name_tokens = set()
        desc_tokens = self._tokenize_process_text(f"{proc.name} {proc.description} {proc.purpose}")
        narrative_source = [f"{proc.name} {proc.description} {proc.purpose}"]
        if source_text:
            narrative_source.append(source_text[:4000])
        bp_keys = self._extract_business_keys(" ".join(narrative_source))

        for task in proc_tasks:
            task_name = getattr(task, "name", "") or ""
            task_desc = getattr(task, "description", "") or ""
            task_inst = getattr(task, "instruction", "") or ""
            task_name_tokens.update(self._tokenize_process_text(task_name))
            task_name_tokens.update(self._tokenize_process_text(task_desc))
            narrative_source.append(f"{task_name} {task_desc} {task_inst}")
            bp_keys.update(self._extract_business_keys(f"{task_name} {task_desc} {task_inst}"))

            role_id = self.task_role_map.get(getattr(task, "task_id", ""))
            role_name = role_id_to_name.get(role_id, "")
            if role_name:
                role_name_tokens.update(self._tokenize_process_text(role_name))
                narrative_source.append(role_name)
                bp_keys.update(self._extract_business_keys(role_name))

        narrative_text = " ".join(narrative_source)
        # Sequence-flow conditions also carry strong process identity clues.
        process_task_ids = {getattr(t, "task_id", "") for t in proc_tasks}
        flow_texts = []
        for flow in self.sequence_flows:
            from_id = flow.get("from_id") or flow.get("from_task_id")
            to_id = flow.get("to_id") or flow.get("to_task_id")
            if from_id in process_task_ids or to_id in process_task_ids:
                cond = (flow.get("condition") or "").strip()
                if cond:
                    flow_texts.append(cond)
        if flow_texts:
            flow_blob = " ".join(flow_texts[:30])
            desc_tokens.update(self._tokenize_process_text(flow_blob))
            narrative_text = f"{narrative_text} {flow_blob}"
            bp_keys.update(self._extract_business_keys(flow_blob))

        return {
            "task_tokens": task_name_tokens,
            "role_tokens": role_name_tokens,
            "desc_tokens": desc_tokens,
            "narrative_ngrams": self._char_ngrams(narrative_text, n=3),
            "bp_keys": bp_keys,
            "task_count": len(proc_tasks),
            "name_norm": self._normalize_process_text(getattr(proc, "name", "")),
        }

    def _process_similarity(self, sig_a: dict, sig_b: dict) -> dict:
        """
        Content-based process similarity score.
        Name similarity is intentionally a weak signal and never the only criterion.
        """
        task_sim = self._jaccard_similarity(sig_a["task_tokens"], sig_b["task_tokens"])
        role_sim = self._jaccard_similarity(sig_a["role_tokens"], sig_b["role_tokens"])
        desc_sim = self._jaccard_similarity(sig_a["desc_tokens"], sig_b["desc_tokens"])
        narrative_sim = self._jaccard_similarity(sig_a["narrative_ngrams"], sig_b["narrative_ngrams"])
        name_sim = 1.0 if sig_a["name_norm"] and sig_a["name_norm"] == sig_b["name_norm"] else 0.0
        bp_overlap = bool(sig_a["bp_keys"] & sig_b["bp_keys"])
        bp_disjoint = bool(sig_a["bp_keys"]) and bool(sig_b["bp_keys"]) and not bp_overlap

        # 이름보다 실제 내용(태스크/역할/서술/식별키) 중심 가중치
        weighted = (
            (task_sim * 0.42)
            + (role_sim * 0.16)
            + (desc_sim * 0.12)
            + (narrative_sim * 0.24)
            + (name_sim * 0.03)
        )

        # Strong content evidence: task overlap is high
        if task_sim >= 0.62 and (sig_a["task_count"] > 0 and sig_b["task_count"] > 0):
            weighted = max(weighted, 0.9)

        # Shared business key inside content strongly indicates same business process.
        if bp_overlap:
            weighted = max(weighted, 0.94)

        # If explicit business keys conflict, lower score to avoid false merge.
        if bp_disjoint:
            weighted *= 0.75

        # Sparse task overlap but similar narrative + role can still be same process (e.g., part1/part2 split docs).
        if task_sim < 0.2 and narrative_sim >= 0.45 and role_sim >= 0.2:
            weighted = max(weighted, 0.72)

        return {
            "score": weighted,
            "task_sim": task_sim,
            "role_sim": role_sim,
            "desc_sim": desc_sim,
            "narrative_sim": narrative_sim,
            "name_sim": name_sim,
            "bp_overlap": bp_overlap,
            "bp_disjoint": bp_disjoint,
        }

    def _decide_process_merge(self, sig_a: dict, sig_b: dict, sim_detail: dict) -> tuple[bool, str, float]:
        """
        Decide whether two processes should be merged using explicit rules.
        Goal: stable and explainable merge behavior, not just one weighted score.
        """
        score = float(sim_detail.get("score", 0.0))
        task_sim = float(sim_detail.get("task_sim", 0.0))
        role_sim = float(sim_detail.get("role_sim", 0.0))
        narrative_sim = float(sim_detail.get("narrative_sim", 0.0))
        desc_sim = float(sim_detail.get("desc_sim", 0.0))
        bp_overlap = bool(sim_detail.get("bp_overlap"))
        bp_disjoint = bool(sim_detail.get("bp_disjoint"))
        name_same = bool(sim_detail.get("name_sim"))
        task_count_a = int(sig_a.get("task_count", 0) or 0)
        task_count_b = int(sig_b.get("task_count", 0) or 0)

        # Hard no-merge rules
        if bp_disjoint and task_sim < 0.45:
            return False, "business_keys_conflict", 0.99
        if (
            task_count_a > 0
            and task_count_b > 0
            and task_sim < 0.12
            and role_sim < 0.12
            and narrative_sim < 0.30
        ):
            return False, "too_little_shared_structure", 0.99

        # Strong merge rules
        if bp_overlap:
            return True, "shared_business_key", 0.55
        if task_count_a > 0 and task_count_b > 0 and task_sim >= 0.62:
            return True, "high_task_overlap", 0.68
        if name_same and (task_sim >= 0.35 or role_sim >= 0.35 or narrative_sim >= 0.50):
            return True, "same_name_with_content_support", 0.64

        # Guardrail: same name alone is never enough
        if name_same and task_sim < 0.15 and role_sim < 0.15 and narrative_sim < 0.30:
            return False, "same_name_but_not_same_content", 0.64

        # Score-based fallback with explicit minimum content support
        threshold = 0.68
        if task_sim < 0.2 and narrative_sim >= 0.45:
            threshold = 0.64
        if (
            score >= threshold
            and (
                task_sim >= 0.25
                or role_sim >= 0.25
                or narrative_sim >= 0.45
                or desc_sim >= 0.45
            )
        ):
            return True, "weighted_score_with_content_support", threshold

        return False, "below_explicit_merge_rules", threshold

    def _merge_duplicate_processes(
        self,
        processes: list,
        tasks: list,
        roles: list,
        chunks: list = None,
    ) -> tuple[list, dict]:
        """
        내용 기반(태스크/역할/설명) 유사도로 중복 프로세스를 병합하고,
        병합된 프로세스 ID 매핑을 반환.

        Returns:
            tuple: (병합된 프로세스 목록, {기존 process_id -> 병합된 process_id} 매핑)
        """
        if not processes:
            return [], {}

        # 역할 ID -> 이름 매핑
        role_id_to_name = {}
        for role in roles or []:
            rid = getattr(role, "role_id", "")
            if rid:
                role_id_to_name[rid] = getattr(role, "name", "") or ""

        # 청크 인덱스: process/task 엔티티가 어떤 원문 청크에서 나왔는지 연결
        chunk_text_by_id = {}
        chunk_doc_by_id = {}
        doc_text_fragments = {}
        for ch in chunks or []:
            cid = getattr(ch, "chunk_id", "")
            if cid:
                chunk_text_by_id[cid] = getattr(ch, "text", "") or ""
                chunk_doc_by_id[cid] = getattr(ch, "doc_id", "") or ""
                did = getattr(ch, "doc_id", "") or ""
                if did:
                    doc_text_fragments.setdefault(did, [])
                    txt = (getattr(ch, "text", "") or "").strip()
                    if txt:
                        # Keep bounded memory; enough for process key/context capture.
                        doc_text_fragments[did].append(txt[:1200])

        def _collect_process_evidence_text(proc_id: str) -> str:
            """
            Gather request-wide evidence text for a process from:
            - direct process chunk
            - task chunks belonging to that process
            - related document-level snippets
            """
            evidence_chunks = []
            related_doc_ids = set()

            proc_chunk_id = self.entity_chunk_map.get(proc_id, "")
            if proc_chunk_id:
                txt = chunk_text_by_id.get(proc_chunk_id, "")
                if txt:
                    evidence_chunks.append(txt[:2000])
                did = chunk_doc_by_id.get(proc_chunk_id, "")
                if did:
                    related_doc_ids.add(did)

            for t in tasks:
                if getattr(t, "process_id", "") != proc_id:
                    continue
                task_id = getattr(t, "task_id", "")
                if not task_id:
                    continue
                task_chunk_id = self.entity_chunk_map.get(task_id, "")
                if not task_chunk_id:
                    continue
                txt = chunk_text_by_id.get(task_chunk_id, "")
                if txt:
                    evidence_chunks.append(txt[:1800])
                did = chunk_doc_by_id.get(task_chunk_id, "")
                if did:
                    related_doc_ids.add(did)

            # Add doc-level context to recover business keys often only stated once in the file.
            doc_contexts = []
            for did in related_doc_ids:
                parts = doc_text_fragments.get(did, [])
                if not parts:
                    continue
                doc_contexts.append(" ".join(parts[:10])[:5000])

            merged = " ".join(evidence_chunks + doc_contexts).strip()
            return merged[:12000]

        # 프로세스별 시그니처 준비
        signatures = {
            proc.proc_id: self._build_process_signature(
                proc,
                tasks,
                role_id_to_name,
                source_text=_collect_process_evidence_text(proc.proc_id),
            )
            for proc in processes
        }

        # Union-Find for clustering
        parent = {proc.proc_id: proc.proc_id for proc in processes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Pairwise similarity-based merging
        for i in range(len(processes)):
            for j in range(i + 1, len(processes)):
                a = processes[i]
                b = processes[j]
                sig_a = signatures.get(a.proc_id, {})
                sig_b = signatures.get(b.proc_id, {})
                sim_detail = self._process_similarity(sig_a, sig_b)
                sim = float(sim_detail.get("score", 0.0))

                should_merge, merge_reason, threshold = self._decide_process_merge(sig_a, sig_b, sim_detail)

                if should_merge:
                    union(a.proc_id, b.proc_id)
                    print(
                        f"   🔗 내용 기반 프로세스 병합 후보 채택: "
                        f"'{a.name}' ↔ '{b.name}' "
                        f"(score={sim:.2f}, task={sim_detail.get('task_sim', 0.0):.2f}, "
                        f"role={sim_detail.get('role_sim', 0.0):.2f}, "
                        f"narr={sim_detail.get('narrative_sim', 0.0):.2f}, "
                        f"bp_overlap={sim_detail.get('bp_overlap')}, "
                        f"reason={merge_reason})"
                    )
                else:
                    # 디버깅: 왜 병합이 안 되었는지 점수 근거를 항상 남긴다.
                    print(
                        f"   🧪 프로세스 유사도 평가: "
                        f"'{a.name}' ↔ '{b.name}' "
                        f"(score={sim:.2f}, threshold={threshold:.2f}, "
                        f"task={sim_detail.get('task_sim', 0.0):.2f}, "
                        f"role={sim_detail.get('role_sim', 0.0):.2f}, "
                        f"desc={sim_detail.get('desc_sim', 0.0):.2f}, "
                        f"narr={sim_detail.get('narrative_sim', 0.0):.2f}, "
                        f"bp_overlap={sim_detail.get('bp_overlap')}, "
                        f"bp_disjoint={sim_detail.get('bp_disjoint')}, "
                        f"reason={merge_reason})"
                    )

        # Build clusters
        clusters = {}
        for proc in processes:
            root = find(proc.proc_id)
            clusters.setdefault(root, []).append(proc)

        unique_processes = []
        process_id_mapping = {}

        for _, proc_group in clusters.items():
            # 대표 프로세스 선택: 태스크 수 많은 순 > 설명 긴 순
            def _rank(p):
                sig = signatures.get(p.proc_id, {})
                desc_len = len((getattr(p, "description", "") or "") + (getattr(p, "purpose", "") or ""))
                return (sig.get("task_count", 0), desc_len)

            primary = max(proc_group, key=_rank)
            unique_processes.append(primary)

            for other in proc_group:
                process_id_mapping[other.proc_id] = primary.proc_id
                if other.proc_id == primary.proc_id:
                    continue
                # 텍스트 정보는 비어있는 필드만 보강
                if getattr(other, "description", "") and not getattr(primary, "description", ""):
                    primary.description = other.description
                if getattr(other, "purpose", "") and not getattr(primary, "purpose", ""):
                    primary.purpose = other.purpose
                print(f"   🔗 프로세스 병합: '{other.name}' ({other.proc_id[:8]}...) → ({primary.proc_id[:8]}...)")

        return unique_processes, process_id_mapping
    
    def _update_task_process_ids(self, tasks: list, process_id_mapping: dict) -> list:
        """태스크의 process_id를 병합된 프로세스 ID로 업데이트."""
        for task in tasks:
            if task.process_id and task.process_id in process_id_mapping:
                old_id = task.process_id
                new_id = process_id_mapping[old_id]
                if old_id != new_id:
                    task.process_id = new_id
        return tasks
    
    def _update_entity_process_ids(self, entities: list, process_id_mapping: dict, id_field: str) -> list:
        """게이트웨이, 이벤트 등의 process_id를 병합된 프로세스 ID로 업데이트."""
        for entity in entities:
            if hasattr(entity, 'process_id') and entity.process_id:
                if entity.process_id in process_id_mapping:
                    old_id = entity.process_id
                    new_id = process_id_mapping[old_id]
                    if old_id != new_id:
                        entity.process_id = new_id
        return entities
    
    def _update_task_process_map(self, process_id_mapping: dict):
        """task_process_map의 process_id를 병합된 ID로 업데이트."""
        for task_id, proc_id in list(self.task_process_map.items()):
            if proc_id in process_id_mapping:
                new_proc_id = process_id_mapping[proc_id]
                if proc_id != new_proc_id:
                    self.task_process_map[task_id] = new_proc_id
    
    def _merge_similar_tasks(self, tasks: list) -> tuple[list, dict]:
        """유사한 태스크를 병합합니다.
        
        병합 기준:
        1. 같은 프로세스 내에서
        2. 같은 역할이 수행하며
        3. 이름이 유사하거나 하나가 다른 하나를 포함하는 경우
        
        Returns:
            tuple: (병합된 태스크 리스트, {old_task_id: new_task_id} 매핑)
        """
        if not tasks:
            return tasks, {}
        
        task_id_mapping = {}  # old_id -> new_id
        
        # 프로세스별로 그룹화
        tasks_by_process = {}
        for task in tasks:
            proc_id = task.process_id or "no_process"
            if proc_id not in tasks_by_process:
                tasks_by_process[proc_id] = []
            tasks_by_process[proc_id].append(task)
        
        merged_tasks = []
        
        for proc_id, proc_tasks in tasks_by_process.items():
            # 1차: 같은 role 안에서 인접 task 병합 (보수적, 기존 로직 유지)
            tasks_by_role = {}
            for task in proc_tasks:
                role_id = self.task_role_map.get(task.task_id, "no_role")
                if role_id not in tasks_by_role:
                    tasks_by_role[role_id] = []
                tasks_by_role[role_id].append(task)

            intermediate: list = []
            for role_id, role_tasks in tasks_by_role.items():
                merged_role_tasks = self._merge_tasks_by_similarity(role_tasks, task_id_mapping)
                intermediate.extend(merged_role_tasks)

            # 2차: process 단위 cross-role 의미 병합 (verb class + 핵심 명사 일치)
            #  - "투심위 참석 (추가설명 및 의견진술)" / "투심위 참석 및 사업설명 및 의견진술"
            #    같이 role 이 약간 다르거나 표현만 다른 동일 활동을 흡수.
            after_cross = self._merge_tasks_cross_role(intermediate, task_id_mapping)

            # 3차: 임베딩 cosine + 휴리스틱 교차검증 의미 병합
            #  - 휴리스틱(verb 사전)에 없는 도메인 동사로 표현된 동일 활동을 잡는다.
            #  - cosine ≥ 임계 AND verb_class 교집합(둘 다 명확할 때)을 모두 충족할 때만 merge.
            #  - cross-process 비교는 하지 않음 (false positive 방지).
            if Config.ENABLE_SEMANTIC_DEDUP and len(after_cross) >= 2:
                after_cross = self._merge_tasks_by_embedding(after_cross, task_id_mapping)

            merged_tasks.extend(after_cross)

        # task_role_map 업데이트
        for old_id, new_id in task_id_mapping.items():
            if old_id in self.task_role_map and old_id != new_id:
                self.task_role_map[new_id] = self.task_role_map[old_id]

        # NEW: merge 로 사라진 task id 를 가리키는 sequence_flows / task_process_map 도 함께 remap.
        # - 이 단계가 빠지면 self.sequence_flows 가 stale id 를 가진 채 _create_sequence_flows 로 흘러가
        #   silent 하게 flow 가 drop 되고, 결과적으로 LLM 에 전달되는 extracted.sequence_flows 가
        #   부실해진다. (LLM 이 task 순서를 추론하지 못해 임의 정렬하는 회귀의 직접 원인.)
        self._apply_task_id_remap(task_id_mapping)

        return merged_tasks, task_id_mapping

    def _apply_task_id_remap(self, task_id_mapping: dict) -> None:
        """task merge 결과(old_id → new_id)를 sequence_flows / task_process_map 에 일괄 반영.

        - self.sequence_flows 의 from_id/to_id/from_task_id/to_task_id 를 모두 remap
        - self-loop (양 끝이 같아진 flow) 제거
        - (from, to) 중복 제거
        - self.task_process_map 의 키도 새 id 로 이동
        """
        if not task_id_mapping:
            return

        try:
            remapped_count = 0
            seen_pairs: set[tuple[str, str]] = set()
            deduped_flows: list = []
            for flow in self.sequence_flows or []:
                if not isinstance(flow, dict):
                    continue
                new_flow = dict(flow)
                for key in ("from_id", "from_task_id", "to_id", "to_task_id"):
                    v = new_flow.get(key)
                    if isinstance(v, str) and v in task_id_mapping:
                        new_flow[key] = task_id_mapping[v]
                        remapped_count += 1
                fid = new_flow.get("from_id") or new_flow.get("from_task_id") or ""
                tid = new_flow.get("to_id") or new_flow.get("to_task_id") or ""
                if fid and tid and fid == tid:
                    continue
                pair = (str(fid), str(tid))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                deduped_flows.append(new_flow)

            before = len(self.sequence_flows or [])
            after = len(deduped_flows)
            if remapped_count or before != after:
                print(
                    f"   🔧 sequence_flows id remap: {before} → {after} "
                    f"(remapped fields={remapped_count})"
                )
            self.sequence_flows = deduped_flows
        except Exception as exc:
            print(f"   ⚠️ sequence_flows remap 실패 (무시하고 진행): {exc}")

        try:
            for old_id, new_id in task_id_mapping.items():
                if old_id == new_id:
                    continue
                if old_id in self.task_process_map:
                    if new_id not in self.task_process_map:
                        self.task_process_map[new_id] = self.task_process_map[old_id]
                    self.task_process_map.pop(old_id, None)
        except Exception as exc:
            print(f"   ⚠️ task_process_map remap 실패 (무시하고 진행): {exc}")

    def _merge_tasks_by_similarity(self, tasks: list, task_id_mapping: dict) -> list:
        """이름 유사도를 기반으로 태스크를 병합합니다."""
        if len(tasks) <= 1:
            return tasks

        def _should_merge_task_pair(left, right) -> tuple[bool, str]:
            left_name = str(left.name or "").strip().lower()
            right_name = str(right.name or "").strip().lower()
            order_gap = abs(int(getattr(left, "order", 0) or 0) - int(getattr(right, "order", 0) or 0))

            # 동사 동의어 클래스 비교 (예: 제공/제출 → "share")
            left_classes = self._task_verb_class(left_name)
            right_classes = self._task_verb_class(right_name)
            shared_classes = left_classes & right_classes

            # 둘 다 명확한 동사 클래스가 있는데 겹치지 않으면 다른 활동
            if left_classes and right_classes and not shared_classes:
                return False, "different_verb_class"

            # 이름이 정확히 같다고 해서 자동 merge 하지 않는다.
            # ("승인" / "승인" 같이 도메인상 반복되는 단계가 단순 동명으로 합쳐지는 사고 방지)
            # → instruction/description 임베딩 cosine 으로 의미 검증.
            if left_name == right_name:
                ok, reason, _sim = self._decide_same_name_merge(left, right)
                return ok, reason

            # substring 포함도 동일 정책. 짧은 이름이 다른 단계의 이름을 우연히
            # 포함하는 경우 (예: "승인" ⊂ "최종 승인") 가 false positive 의 주범이므로
            # instruction 신호로 한 번 더 검증한다.
            if left_name in right_name or right_name in left_name:
                ok, reason, _sim = self._decide_same_name_merge(left, right)
                if ok:
                    return True, f"name_contains_other_with_instr_sim({reason})"
                return False, f"name_contains_other_but_instr_diff({reason})"

            if self._have_same_core_words(left_name, right_name) and order_gap <= 3:
                return True, "same_core_words_nearby"

            similarity = self._calc_name_similarity(left_name, right_name)
            if order_gap <= 2 and similarity > 0.65:
                return True, "high_name_similarity_adjacent"

            # verb class 가 겹치고 핵심 명사도 충분히 겹치면 거리 무관 병합
            # (한쪽 명사가 1개뿐이면 단어 하나로 묶이는 사고 방지 → 거부)
            if shared_classes:
                left_nouns = self._task_core_nouns(left_name)
                right_nouns = self._task_core_nouns(right_name)
                if left_nouns and right_nouns:
                    overlap = left_nouns & right_nouns
                    min_size = min(len(left_nouns), len(right_nouns))
                    if min_size >= 2 and overlap and len(overlap) >= max(2, min_size // 2):
                        return True, "shared_verb_class_and_core_nouns"

            return False, "below_task_merge_rules"
        
        # order로 정렬
        sorted_tasks = sorted(tasks, key=lambda t: t.order)
        merged = []
        skip_indices = set()
        
        for i, task in enumerate(sorted_tasks):
            if i in skip_indices:
                continue
            
            task_name = task.name.lower().strip()
            merged_with = []
            
            # 다른 태스크와 비교
            for j, other_task in enumerate(sorted_tasks):
                if i == j or j in skip_indices:
                    continue
                
                other_name = other_task.name.lower().strip()
                
                should_merge, merge_reason = _should_merge_task_pair(task, other_task)
                if should_merge:
                    merged_with.append((j, other_task))
                    skip_indices.add(j)
            
            # 병합 수행
            if merged_with:
                # 가장 긴 이름을 가진 태스크를 대표로 선택
                all_related = [task] + [t for _, t in merged_with]
                representative = max(all_related, key=lambda t: len(t.name))

                # 대표 task 의 order 는 클러스터 내 최소 order 로 갱신.
                # (대표가 "가장 긴 이름" 기준이라 원본 순서가 뒤로 밀린 task 가 선택되면
                #  merged_tasks 가 정렬됐을 때 task 가 본래 위치보다 훨씬 뒤로 이동해
                #  LLM/extract 단계에서 순서가 망가진다.)
                try:
                    min_order = min(int(getattr(x, "order", 0) or 0) for x in all_related)
                    if int(getattr(representative, "order", 0) or 0) > min_order:
                        representative.order = min_order
                except Exception:
                    pass

                # 설명 통합
                descriptions = [t.description for t in all_related if t.description]
                if descriptions:
                    representative.description = " | ".join(set(descriptions))

                # 지침(instruction) 통합: 더 구체적인 지침 우선, 필요 시 줄 단위로 병합
                instructions = [
                    getattr(t, "instruction", "") for t in all_related if getattr(t, "instruction", "")
                ]
                if instructions:
                    # 원문 보존 우선:
                    # - 라인 수 제한/요약 없이 수집
                    # - 중복 라인만 제거하여 정보 손실을 최소화
                    merged_lines: list[str] = []
                    seen = set()
                    for src in instructions:
                        for ln in (src or "").splitlines():
                            s = ln.strip()
                            if not s:
                                continue
                            if s in seen:
                                continue
                            seen.add(s)
                            merged_lines.append(s)
                    representative.instruction = "\n".join(merged_lines).strip()
                
                # ID 매핑 기록
                for t in all_related:
                    if t.task_id != representative.task_id:
                        task_id_mapping[t.task_id] = representative.task_id
                
                merged.append(representative)
                print(f"   🔀 병합: {[t.name for t in all_related]} → {representative.name}")
            else:
                merged.append(task)
        
        return merged
    
    def _task_verb_class(self, name: str) -> set[str]:
        """Task 이름에 나타나는 동사를 동의어 클래스 집합으로 매핑."""
        if not name:
            return set()
        s = str(name)
        classes: set[str] = set()
        for cls, verbs in _TASK_VERB_SYNONYMS.items():
            for v in verbs:
                if v in s:
                    classes.add(cls)
                    break
        return classes

    def _task_core_nouns(self, name: str) -> set[str]:
        """Task 이름에서 동사/조사/일반 stopword 를 제거한 핵심 명사 토큰."""
        if not name:
            return set()
        cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", str(name))
        # 괄호 안 부연(예: "(추가설명 및 의견진술)") 도 제거
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        tokens = {tok for tok in cleaned.split() if len(tok) >= 2}
        # 동사 토큰 제거
        tokens = {tok for tok in tokens if tok not in _TASK_ALL_VERB_TOKENS}
        # 일반 stopword 제거
        tokens = {tok for tok in tokens if tok not in _TASK_NOUN_STOP_TOKENS}
        return tokens

    def _are_same_task_semantic(self, a, b) -> bool:
        """role 경계 너머에서도 의미상 동일한 task 인지 보수적으로 판정.

        chain 폭주(예: '자료' 한 단어로 'investo심위 ...' task 들이 한 그룹으로 묶이는 사고) 방지 위해:
          - verb class 는 양쪽 다 존재하고 교집합이 있어야 함 (한쪽만 있을 때는 매칭 안 함)
          - 핵심 명사 overlap 임계: min_size 별로 단계적 강화
              · min_size == 1: 거부 (단어 하나만으로 묶이지 않음)
              · min_size == 2: overlap == 2 (양쪽 명사 셋이 완전 일치해야 함)
              · min_size 3~4: overlap >= 2
              · min_size >= 5: overlap >= 3
          - substring 매칭은 6글자 이상에서만 허용 (짧은 공통 단어로 묶이는 것 방지)
        """
        an = str(getattr(a, "name", "") or "")
        bn = str(getattr(b, "name", "") or "")
        al, bl = an.lower().strip(), bn.lower().strip()
        if not al or not bl:
            return False
        # 이름이 같거나 한쪽이 다른쪽을 충분히 포함하면, instruction 신호로 한 번 더 검증.
        # 결재 체인 ("팀장 승인" / "본부장 승인") 처럼 role 만 다르고 이름은 같은
        # 별개 단계가 자동으로 묶이는 사고를 방지한다.
        if al == bl or (len(al) >= 6 and al in bl) or (len(bl) >= 6 and bl in al):
            ok, _reason, _sim = self._decide_same_name_merge(a, b)
            return ok

        a_verbs = self._task_verb_class(an)
        b_verbs = self._task_verb_class(bn)

        # verb class 가 한쪽만 있으면 의미 비교가 어려우므로 매칭하지 않음
        # (한쪽이 verb 없는 짧은 phrase 인 경우 anchor 로 잘못 묶이는 사고 방지)
        if not (a_verbs and b_verbs):
            return False
        # verb class 교집합 필수
        if not (a_verbs & b_verbs):
            return False

        a_nouns = self._task_core_nouns(an)
        b_nouns = self._task_core_nouns(bn)
        if not a_nouns or not b_nouns:
            return False
        overlap = a_nouns & b_nouns
        if not overlap:
            return False
        min_size = min(len(a_nouns), len(b_nouns))

        # 단계적 임계
        if min_size <= 1:
            # 한쪽 명사가 1개뿐이면 anchor 가 너무 약함 → 거부
            return False
        if min_size == 2:
            # 양쪽 다 작은 명사 셋 → 완전 일치 요구
            return len(overlap) == 2
        if min_size <= 4:
            return len(overlap) >= 2
        # min_size >= 5
        return len(overlap) >= 3

    def _merge_tasks_cross_role(self, tasks: list, task_id_mapping: dict) -> list:
        """role 경계 너머에서 의미상 동일한 task 를 한 번 더 묶는다."""
        if len(tasks) <= 1:
            return tasks

        sorted_tasks = sorted(tasks, key=lambda t: getattr(t, "order", 0) or 0)
        skip: set[int] = set()
        result: list = []

        for i, t in enumerate(sorted_tasks):
            if i in skip:
                continue
            cluster = [t]
            for j in range(i + 1, len(sorted_tasks)):
                if j in skip:
                    continue
                other = sorted_tasks[j]
                if self._are_same_task_semantic(t, other):
                    cluster.append(other)
                    skip.add(j)

            if len(cluster) > 1:
                # 대표: 가장 긴 이름 (정보량 많음). 동률이면 가장 처음.
                rep = max(cluster, key=lambda x: (len(getattr(x, "name", "") or ""),))
                # 대표 task 의 order 는 클러스터 내 최소 order 로 보정 (원본 순서 보존)
                try:
                    min_order = min(int(getattr(x, "order", 0) or 0) for x in cluster)
                    if int(getattr(rep, "order", 0) or 0) > min_order:
                        rep.order = min_order
                except Exception:
                    pass
                # description 통합
                descs = [getattr(x, "description", "") for x in cluster if getattr(x, "description", "")]
                if descs:
                    rep.description = " | ".join(dict.fromkeys(descs))
                # instruction 통합 (라인 단위 dedup)
                instrs = [getattr(x, "instruction", "") for x in cluster if getattr(x, "instruction", "")]
                if instrs:
                    seen_lines: set[str] = set()
                    merged_lines: list[str] = []
                    for src in instrs:
                        for ln in (src or "").splitlines():
                            s = ln.strip()
                            if s and s not in seen_lines:
                                seen_lines.add(s)
                                merged_lines.append(s)
                    rep.instruction = "\n".join(merged_lines).strip()
                # ID 매핑 (기존 매핑도 transitively 갱신)
                rep_id = rep.task_id
                for x in cluster:
                    if x.task_id != rep_id:
                        task_id_mapping[x.task_id] = rep_id
                # 기존 매핑 중 새로 흡수된 ID 를 가리키던 것도 rep 으로 재가리킴
                for old_id, new_id in list(task_id_mapping.items()):
                    if new_id in {x.task_id for x in cluster if x.task_id != rep_id}:
                        task_id_mapping[old_id] = rep_id
                print(
                    f"   🔀 cross-role 병합: "
                    f"{[getattr(x, 'name', '') for x in cluster]} → {getattr(rep, 'name', '')}"
                )
                result.append(rep)
            else:
                result.append(t)

        return result

    # ------------------------------------------------------------------
    # 임베딩 기반 의미 병합 (휴리스틱 한계 보완)
    # ------------------------------------------------------------------
    def _build_task_embed_text(self, task) -> str:
        """task 임베딩 입력 텍스트. name + instruction 의 첫 라인 일부를 결합해
        의미 신호를 강화한다. (단순 name 만으로는 짧은 한국어 task 의 cosine 가
        도메인 어휘에 의해 흔들릴 수 있음)
        """
        name = (getattr(task, "name", "") or "").strip()
        instr = (getattr(task, "instruction", "") or "").strip()
        # instruction 첫 1~2 라인만 (라벨 노이즈 줄이기 위해 간단 절단)
        snippet = ""
        if instr:
            for ln in instr.splitlines():
                s = ln.strip()
                if not s:
                    continue
                snippet = (snippet + " " + s).strip() if snippet else s
                if len(snippet) >= 200:
                    break
        if snippet:
            return f"{name}. {snippet[:240]}"
        return name

    def _merge_tasks_by_embedding(self, tasks: list, task_id_mapping: dict) -> list:
        """임베딩 cosine + 휴리스틱 교차검증으로 의미적 동일 task 병합.

        한국어 도메인 어휘 (예: "실무위원회 X" / "실무위원회 Y") 끼리는 cosine 만으로
        false positive 가 쉽게 나기 때문에, 두 단계 임계 정책을 사용한다:

          A) 강한 휴리스틱 confirm + 약한 cosine 임계 (TASK_SEMANTIC_COSINE_MIN, 기본 0.85)
             강한 휴리스틱 = (한쪽이 다른쪽 substring 6글자+ 포함)
                          OR (verb class 양쪽 명확 + 교집합 + noun Jaccard >= TASK_NOUN_JACCARD_MIN)
          B) 강한 cosine 단독 신호 (TASK_SEMANTIC_HIGH_COSINE, 기본 0.92)
             휴리스틱이 약해도 cosine 이 매우 높으면 허용 (도메인 동사 사전에 없는 표현 흡수)

        그 외에는 모두 reject. process 단위로만 비교 (cross-process 비교 X).
        임베딩 호출 실패 시 graceful degrade.
        """
        if len(tasks) <= 1:
            return tasks

        embed_texts = [self._build_task_embed_text(t) for t in tasks]
        try:
            embeddings = self.vector_search.embed_texts(embed_texts)
        except Exception as exc:
            print(f"   ⚠️ Task 임베딩 호출 실패 → 임베딩 병합 스킵: {exc}")
            return tasks

        if not embeddings or len(embeddings) != len(tasks):
            return tasks

        low_thr = float(self._cfg("TASK_SEMANTIC_COSINE_MIN"))
        high_thr = float(self._cfg("TASK_SEMANTIC_HIGH_COSINE"))
        jaccard_min = float(self._cfg("TASK_NOUN_JACCARD_MIN"))

        sorted_idx = sorted(range(len(tasks)), key=lambda k: int(getattr(tasks[k], "order", 0) or 0))
        skip: set[int] = set()
        result: list = []
        merged_total = 0

        for pos_i, i in enumerate(sorted_idx):
            if i in skip:
                continue
            cluster: list = [tasks[i]]
            for pos_j in range(pos_i + 1, len(sorted_idx)):
                j = sorted_idx[pos_j]
                if j in skip:
                    continue

                ti, tj = tasks[i], tasks[j]
                ni = (getattr(ti, "name", "") or "").strip().lower()
                nj = (getattr(tj, "name", "") or "").strip().lower()
                if not ni or not nj:
                    continue
                if ni == nj:
                    # 이름이 같다고 해서 임베딩 비교를 건너뛰지 않는다.
                    # instruction/description 신호로 의미 동일성을 한 번 더 검증.
                    ok, reason, sim = self._decide_same_name_merge(ti, tj)
                    if ok:
                        if sim is not None:
                            print(
                                f"   🧠 same-name merge ({reason}): "
                                f"'{getattr(ti, 'name', '')}' ↔ '{getattr(tj, 'name', '')}'"
                            )
                        cluster.append(tj)
                        skip.add(j)
                    else:
                        print(
                            f"   🚫 same-name 분리 유지 ({reason}): "
                            f"'{getattr(ti, 'name', '')}' ↔ '{getattr(tj, 'name', '')}'"
                        )
                    continue

                try:
                    cos = self.vector_search.cosine_similarity(embeddings[i], embeddings[j])
                except Exception:
                    continue

                # 임계 미달이면 즉시 reject
                if cos < low_thr:
                    continue

                # 강한 휴리스틱 confirm 여부 판정
                is_substring = (len(ni) >= 6 and ni in nj) or (len(nj) >= 6 and nj in ni)
                vi = self._task_verb_class(getattr(ti, "name", ""))
                vj = self._task_verb_class(getattr(tj, "name", ""))
                ai = self._task_core_nouns(getattr(ti, "name", ""))
                aj = self._task_core_nouns(getattr(tj, "name", ""))

                # verb class 가 양쪽 다 명확한데 교집합이 없으면 무조건 reject
                # ("선정" vs "검토" 같은 명백히 다른 동사)
                if vi and vj and not (vi & vj):
                    continue

                strong_heuristic = False
                if is_substring:
                    strong_heuristic = True
                elif vi and vj and (vi & vj) and ai and aj:
                    overlap = ai & aj
                    union = ai | aj
                    jaccard = (len(overlap) / len(union)) if union else 0.0
                    if jaccard >= jaccard_min:
                        strong_heuristic = True

                if strong_heuristic:
                    # 약한 임계로 통과
                    pass
                elif cos >= high_thr:
                    # 강한 cosine 단독 신호로 통과
                    pass
                else:
                    # 어느 정책도 충족 못함
                    continue

                cluster.append(tj)
                skip.add(j)

            if len(cluster) > 1:
                # 대표: 가장 긴 이름 (정보량 많음)
                rep = max(cluster, key=lambda x: len(getattr(x, "name", "") or ""))
                # 대표 task 의 order 는 클러스터 내 최소 order 로 보정 (원본 순서 보존)
                try:
                    min_order = min(int(getattr(x, "order", 0) or 0) for x in cluster)
                    if int(getattr(rep, "order", 0) or 0) > min_order:
                        rep.order = min_order
                except Exception:
                    pass
                self._merge_task_metadata(cluster, rep)
                rep_id = rep.task_id
                cluster_other_ids = {x.task_id for x in cluster if x.task_id != rep_id}
                for x in cluster:
                    if x.task_id != rep_id:
                        task_id_mapping[x.task_id] = rep_id
                # transitively 갱신: 기존 매핑 중 흡수된 ID 를 가리키던 것도 rep 으로
                for old_id, new_id in list(task_id_mapping.items()):
                    if new_id in cluster_other_ids:
                        task_id_mapping[old_id] = rep_id
                merged_total += len(cluster) - 1
                names_dbg = [getattr(x, "name", "") for x in cluster]
                # 임계 정보도 함께 로깅 (디버깅 편의)
                print(
                    f"   🧠 임베딩 병합 (cos_low={low_thr:.2f}/high={high_thr:.2f}): "
                    f"{names_dbg} → {rep.name}"
                )
                result.append(rep)
            else:
                result.append(tasks[i])

        if merged_total:
            print(f"   🧠 임베딩 병합 합계: {merged_total} 개 task 흡수")

        return result

    def _merge_task_metadata(self, cluster: list, rep) -> None:
        """task description / instruction 통합 (라인 단위 dedup)."""
        descs = [getattr(x, "description", "") for x in cluster if getattr(x, "description", "")]
        if descs:
            rep.description = " | ".join(dict.fromkeys(descs))
        instrs = [getattr(x, "instruction", "") for x in cluster if getattr(x, "instruction", "")]
        if instrs:
            seen_lines: set[str] = set()
            merged_lines: list[str] = []
            for src in instrs:
                for ln in (src or "").splitlines():
                    s = ln.strip()
                    if s and s not in seen_lines:
                        seen_lines.add(s)
                        merged_lines.append(s)
            rep.instruction = "\n".join(merged_lines).strip()

    # ------------------------------------------------------------------
    # Dedup 강도 (사용자 [도구 설정] 다이얼로그 값)
    # ------------------------------------------------------------------
    def set_dedup_level(self, level: str | None) -> None:
        """사용자가 선택한 dedup 강도를 적용한다.

        Args:
            level: "concise" | "standard" | "detailed". 그 외/None 은 standard 로 폴백.
        """
        normalized = (level or "standard").strip().lower()
        if normalized not in _DEDUP_LEVEL_PROFILES:
            print(f"   ⚠️ 알 수 없는 dedup level '{level}' → 'standard' 로 폴백")
            normalized = "standard"
        self._dedup_level = normalized
        self._dedup_overrides = dict(_DEDUP_LEVEL_PROFILES[normalized])
        if self._dedup_overrides:
            print(
                f"   ⚙️ dedup level='{normalized}' 적용 — 임계값 오버라이드 "
                f"{self._dedup_overrides}"
            )
        else:
            print(f"   ⚙️ dedup level='{normalized}' 적용 — Config 기본값 사용")

        # SOP 분할 단계 (PDFExtractor) 에도 같은 level 을 동기 적용한다.
        # (executor 의 _build_state_from_memento_chunks 경로 외에 graph.py 의
        #  self.pdf_extractor 를 직접 사용하는 경로에서도 일관된 동작 보장.)
        try:
            if hasattr(self, "pdf_extractor") and hasattr(self.pdf_extractor, "set_segmentation_level"):
                self.pdf_extractor.set_segmentation_level(normalized)
        except Exception:
            pass

    def _cfg(self, name: str):
        """임계값 조회 헬퍼.

        - 사용자 dedup level 의 오버라이드가 있으면 그 값을 우선 사용
        - 없으면 Config 의 기본 attribute 를 그대로 반환
        """
        if name in self._dedup_overrides:
            return self._dedup_overrides[name]
        return getattr(Config, name)

    # ------------------------------------------------------------------
    # 이름이 같은 task 페어의 instruction/description 의미 유사도 검증
    # (1차/2차/3차 dedup 단계에서 공통으로 사용 — 자동 통과 방지용)
    # ------------------------------------------------------------------
    def _task_extra_text(self, task) -> str:
        """task 의 의미 비교용 보조 텍스트.

        instruction 우선, 없으면 description 으로 폴백. 너무 긴 경우 임베딩 입력
        토큰을 아끼기 위해 절단한다 (도메인 task 지침은 대개 600자 안에 핵심이 있음).
        """
        txt = (getattr(task, "instruction", "") or "").strip()
        if not txt:
            txt = (getattr(task, "description", "") or "").strip()
        return txt[:600]

    def _instructions_similarity(self, a, b) -> float | None:
        """두 task 의 instruction/description 임베딩 cosine 유사도.

        Returns:
            float (0.0 ~ 1.0) — 비교 가능한 경우
            None — 한쪽이라도 보조 텍스트가 비었거나 임베딩 호출이 실패한 경우

        같은 페어가 여러 단계에서 평가될 수 있으므로 self._instr_sim_cache 로
        결과를 재사용한다.
        """
        aid = getattr(a, "task_id", "") or ""
        bid = getattr(b, "task_id", "") or ""
        cache_key: tuple[str, str] | None = None
        if aid and bid:
            cache_key = (aid, bid) if aid <= bid else (bid, aid)
            if cache_key in self._instr_sim_cache:
                return self._instr_sim_cache[cache_key]

        txt_a = self._task_extra_text(a)
        txt_b = self._task_extra_text(b)
        if not txt_a or not txt_b:
            if cache_key is not None:
                self._instr_sim_cache[cache_key] = None
            return None

        try:
            embs = self.vector_search.embed_texts([txt_a, txt_b])
        except Exception as exc:
            print(f"   ⚠️ instruction 임베딩 호출 실패 → 비교 스킵: {exc}")
            if cache_key is not None:
                self._instr_sim_cache[cache_key] = None
            return None

        if not embs or len(embs) != 2:
            if cache_key is not None:
                self._instr_sim_cache[cache_key] = None
            return None

        try:
            sim = float(self.vector_search.cosine_similarity(embs[0], embs[1]))
        except Exception:
            if cache_key is not None:
                self._instr_sim_cache[cache_key] = None
            return None

        if cache_key is not None:
            self._instr_sim_cache[cache_key] = sim
        return sim

    def _decide_same_name_merge(self, a, b) -> tuple[bool, str, float | None]:
        """이름이 정확히 같은 task 페어를 합칠지 의미 신호로 판정.

        정책:
          - instruction/description 비교 가능: cosine 임계 충족 시에만 merge.
            · 같은 role 끼리: TASK_SAME_NAME_INSTR_COSINE_SAME_ROLE (기본 0.78)
            · 다른 role 끼리: TASK_SAME_NAME_INSTR_COSINE_DIFF_ROLE (기본 0.86)
          - 비교 불가 (한쪽이라도 텍스트 부재 / 임베딩 실패):
            안전한 기본값으로 merge 허용 (기존 동작 유지, BPMN 손실 방지).

        Returns:
            (merge 여부, 사유, cosine 값 또는 None)
        """
        sim = self._instructions_similarity(a, b)
        if sim is None:
            return True, "same_name_no_instruction_signal", None

        role_a = self.task_role_map.get(getattr(a, "task_id", ""))
        role_b = self.task_role_map.get(getattr(b, "task_id", ""))
        same_role = bool(role_a) and bool(role_b) and role_a == role_b
        threshold = (
            float(self._cfg("TASK_SAME_NAME_INSTR_COSINE_SAME_ROLE"))
            if same_role
            else float(self._cfg("TASK_SAME_NAME_INSTR_COSINE_DIFF_ROLE"))
        )

        if sim >= threshold:
            return True, f"same_name_instr_sim_pass({sim:.2f}>={threshold:.2f})", sim
        return False, f"same_name_instr_sim_fail({sim:.2f}<{threshold:.2f})", sim

    # ------------------------------------------------------------------
    # Role 정규화 (직급 보존 + 책임 task signature 기반 클러스터링)
    # ------------------------------------------------------------------
    def _role_display_key(self, name: str) -> str:
        """Role 이름의 보수적 정규화 키 (직급은 보존).

        괄호/구분기호/공백만 제거한 lowercase 형태. 직급 접미사는
        의도적으로 보존하여 "기획처장" / "기획본부장" 같이 다른 직급은
        분리된 채로 둔다. ("기획처장" / "기획 처장" / "기획처장(주관)" 정도만
        같은 키로 흡수)
        """
        if not name:
            return ""
        s = str(name).strip()
        for p in _ROLE_NOISE_PATTERNS:
            s = re.sub(p, " ", s)
        # "또는/및/and/or" 로 연결된 다중 역할은 첫 부분만 키로 사용
        parts = re.split(r"\s+(?:또는|혹은|및|or|and)\s+", s, flags=re.IGNORECASE)
        primary = (parts[0] if parts else s).strip()
        # 마지막 조사 "의" 만 제거
        primary = re.sub(r"의$", "", primary).strip()
        # 공백 제거하여 표기 흔들림 흡수 ("기획처장" == "기획 처장")
        primary = re.sub(r"\s+", "", primary)
        return primary.lower()

    def _normalize_roles_by_responsibility(
        self, roles: list
    ) -> tuple[list, dict]:
        """직급은 보존하면서 사실상 동일한 책임을 수행하는 Role 들을 흡수.

        병합 규칙 (Union-Find):
          1. 보수적 이름 정규화 (괄호/구분기호/공백만 제거, 직급 유지) 후 정확 일치
          2. 정규화된 task signature (assigned task_id set) Jaccard >= TASK_SIG_JACCARD_MIN
             AND 양쪽 task 수 >= TASK_SIG_MIN_COUNT 인 경우 흡수
             - 이름이 달라도 실질 책임 task 가 거의 같으면 같은 역할로 본다는 가정
             - 이 단계는 task merge 가 끝난 뒤 호출되어야 의미 있음

        직급 접미사 ("처장"/"본부장" 등) 는 인위로 제거하지 않으므로 task 책임이
        다른 직급들은 자연히 분리 유지된다.

        흡수된 role_id 는 self.task_role_map / self.role_decision_map /
        self.role_skill_map 에서도 대표 ID 로 재매핑된다.

        Returns:
            (canonical_roles, {old_role_id: representative_role_id})
        """
        if not roles:
            return roles, {}

        # ----- task signature 빌드 -----
        # task_role_map: task_id -> role_id (이미 task merge 후 정규화된 task_id)
        role_to_taskids: dict[str, set[str]] = {}
        for tid, rid in self.task_role_map.items():
            if not rid:
                continue
            role_to_taskids.setdefault(rid, set()).add(tid)

        # role 별 process scope 결정 (process 격리 정책).
        # - role 의 task 들이 모두 단일 process 에 속하면 그 proc_id 가 scope.
        # - 여러 process 에 걸쳐 있으면 "_multi" (보수적으로 같은 _multi 끼리만 비교).
        # - task 가 하나도 없으면 "_orphan" (다른 그룹으로 절대 흡수되지 않음).
        # task_process_map 은 task merge 단계에서 갱신되어 있으며,
        # task_id_mapping 으로 흡수된 ID 도 표준 ID 로 일관 조회 가능해야 한다.
        def _role_proc_scope(task_ids: set[str]) -> str:
            if not task_ids:
                return "_orphan"
            procs = set()
            for tid in task_ids:
                pid = self.task_process_map.get(tid)
                if pid:
                    procs.add(pid)
            if not procs:
                return "_orphan"
            if len(procs) == 1:
                return next(iter(procs))
            return "_multi"

        items: list[dict] = []
        for i, r in enumerate(roles):
            nm = (getattr(r, "name", "") or "").strip()
            rid = getattr(r, "role_id", None)
            tasks = role_to_taskids.get(rid, set()) if rid else set()
            items.append({
                "idx": i,
                "role": r,
                "name": nm,
                "nkey": self._role_display_key(nm),
                "rid": rid,
                "tasks": tasks,
                "proc_scope": _role_proc_scope(tasks),
            })

        n = len(items)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        def _same_scope(i: int, j: int) -> bool:
            """두 role 이 같은 process scope 일 때만 병합 허용.

            process 격리 정책:
              - 같은 proc_id        → 비교 가능
              - 둘 다 "_multi"      → 비교 가능 (이미 cross-process role)
              - 둘 다 "_orphan"     → 비교 가능 (둘 다 task 없음, 이름만으로)
              - 그 외 (proc_id ↔ _multi / _orphan, 다른 proc_id 끼리) → 차단
            """
            return items[i]["proc_scope"] == items[j]["proc_scope"]

        # ----- 1단계: 이름 정규화 키 정확 일치 (process scope 별 그룹) -----
        # 같은 직책 이름이 여러 process 에 등장해도 process 가 다르면 합치지 않음.
        by_key_per_scope: dict[tuple[str, str], int] = {}
        for i, it in enumerate(items):
            k = it["nkey"]
            if not k:
                continue
            scope = it["proc_scope"]
            bucket_key = (scope, k)
            if bucket_key in by_key_per_scope:
                _union(by_key_per_scope[bucket_key], i)
            else:
                by_key_per_scope[bucket_key] = i

        # ----- 2단계: task signature Jaccard 기반 (process scope 별) -----
        # 양쪽 다 충분한 task (>=2) 를 가진 role 끼리만 비교 → 노이즈 방지.
        # task_id 가 process 별로 분리되어 있으므로 cross-process 합병은 자연 차단되지만,
        # 정책 일관성을 위해 명시적으로 same_scope 검사도 둔다.
        TASK_SIG_MIN_COUNT = 2
        TASK_SIG_JACCARD_MIN = 0.7

        candidates = [
            i for i, it in enumerate(items)
            if len(it["tasks"]) >= TASK_SIG_MIN_COUNT
        ]
        for i_idx in range(len(candidates)):
            for j_idx in range(i_idx + 1, len(candidates)):
                i, j = candidates[i_idx], candidates[j_idx]
                if _find(i) == _find(j):
                    continue
                if not _same_scope(i, j):
                    continue
                ta, tb = items[i]["tasks"], items[j]["tasks"]
                inter = ta & tb
                if not inter:
                    continue
                union = ta | tb
                jacc = len(inter) / len(union)
                if jacc >= TASK_SIG_JACCARD_MIN:
                    _union(i, j)

        # ----- 3단계: 임베딩 cosine + task signature 교차검증 (process scope 별) -----
        # 직급 보존 정책상 cosine 단독으로는 절대 합치지 않는다.
        # ("기획본부장" / "기획처장" 의 cosine 은 보통 0.95+ 로 매우 높지만 직급/책임이 다름)
        # 두 신호가 동시에 강할 때만 흡수:
        #   · 같은 process scope (cross-process 차단)
        #   · cosine ≥ ROLE_SEMANTIC_COSINE_MIN (이름 의미 매우 유사)
        #   · AND 양쪽 다 task ≥ ROLE_EMBED_MIN_TASKS
        #   · AND task signature Jaccard ≥ ROLE_EMBED_JACCARD_MIN (책임도 거의 같음)
        if Config.ENABLE_SEMANTIC_DEDUP and n >= 2:
            try:
                role_texts = [it["name"] for it in items]
                role_embeddings = self.vector_search.embed_texts(role_texts)
            except Exception as exc:
                print(f"   ⚠️ Role 임베딩 호출 실패 → 임베딩 단계 스킵: {exc}")
                role_embeddings = None

            if role_embeddings and len(role_embeddings) == n:
                cos_min = float(self._cfg("ROLE_SEMANTIC_COSINE_MIN"))
                ROLE_EMBED_MIN_TASKS = 2
                ROLE_EMBED_JACCARD_MIN = 0.5
                for i in range(n):
                    for j in range(i + 1, n):
                        if _find(i) == _find(j):
                            continue
                        if not _same_scope(i, j):
                            continue
                        ta, tb = items[i]["tasks"], items[j]["tasks"]
                        # 한쪽이라도 task 부족하면 임베딩 단독 합병 금지 (직급 다른 후보 흡수 방지)
                        if len(ta) < ROLE_EMBED_MIN_TASKS or len(tb) < ROLE_EMBED_MIN_TASKS:
                            continue
                        try:
                            cos = self.vector_search.cosine_similarity(
                                role_embeddings[i], role_embeddings[j]
                            )
                        except Exception:
                            continue
                        if cos < cos_min:
                            continue
                        union = ta | tb
                        if not union:
                            continue
                        jacc = len(ta & tb) / len(union)
                        if jacc < ROLE_EMBED_JACCARD_MIN:
                            continue
                        print(
                            f"   🧠 Role 임베딩 병합: '{items[i]['name']}' ↔ '{items[j]['name']}' "
                            f"(cos={cos:.3f}, task_jaccard={jacc:.2f})"
                        )
                        _union(i, j)

        # ----- 그룹화 → 대표 선정 -----
        groups: dict[int, list[int]] = {}
        for i in range(n):
            r = _find(i)
            groups.setdefault(r, []).append(i)

        canonical_roles: list = []
        role_id_remap: dict[str, str] = {}

        for _, member_indices in groups.items():
            members = [items[k] for k in member_indices]
            # 대표 선정 우선순위:
            #   1) 책임 task 수가 가장 많은 role (책임이 명확한 표현)
            #   2) 이름이 가장 짧은 표현 (일반화된 표기)
            #   3) 등장 순
            members_sorted = sorted(
                members,
                key=lambda m: (-len(m["tasks"]), len(m["name"]), m["idx"]),
            )
            rep = members_sorted[0]["role"]
            rep_id = getattr(rep, "role_id", None)
            canonical_roles.append(rep)
            if not rep_id:
                continue
            for m in members:
                mid = m["rid"]
                if mid and mid != rep_id:
                    role_id_remap[mid] = rep_id

            if len(members) > 1:
                names = [m["name"] for m in members]
                print(
                    f"   🔀 Role 클러스터: {names} → {getattr(rep, 'name', '')}"
                )

        if not role_id_remap:
            return canonical_roles, role_id_remap

        # task_role_map 갱신 (transitively: 흡수된 ID 를 가리키던 항목들 모두 대표로)
        new_task_role_map: dict[str, str] = {}
        for tid, rid in self.task_role_map.items():
            new_task_role_map[tid] = role_id_remap.get(rid, rid)
        self.task_role_map = new_task_role_map

        # role_decision_map 갱신 (병합 시 중복 제거)
        new_rdmap: dict[str, list] = {}
        for rid, dlist in self.role_decision_map.items():
            new_rid = role_id_remap.get(rid, rid)
            new_rdmap.setdefault(new_rid, []).extend(dlist or [])
        self.role_decision_map = {k: list(dict.fromkeys(v)) for k, v in new_rdmap.items()}

        # role_skill_map 갱신
        new_rsmap: dict[str, list] = {}
        for rid, slist in self.role_skill_map.items():
            new_rid = role_id_remap.get(rid, rid)
            new_rsmap.setdefault(new_rid, []).extend(slist or [])
        self.role_skill_map = {k: list(dict.fromkeys(v)) for k, v in new_rsmap.items()}

        return canonical_roles, role_id_remap

    def _have_same_core_words(self, name1: str, name2: str) -> bool:
        """두 이름이 핵심 단어를 공유하는지 확인합니다."""
        # 한국어 조사/어미 제거를 위한 간단한 처리
        stop_words = {'및', '의', '을', '를', '이', '가', '에', '로', '으로', '와', '과', '에서', '부터', '까지'}

        normalized1 = name1.replace('및', ' ')
        normalized2 = name2.replace('및', ' ')
        words1 = {w.strip() for w in normalized1.split() if w.strip()} - stop_words
        words2 = {w.strip() for w in normalized2.split() if w.strip()} - stop_words
        if not words1 or not words2:
            compact1 = re.split(r"[^0-9a-zA-Z가-힣]+", name1)
            compact2 = re.split(r"[^0-9a-zA-Z가-힣]+", name2)
            words1 = {w for w in compact1 if len(w) >= 2} - stop_words
            words2 = {w for w in compact2 if len(w) >= 2} - stop_words
        
        # 공통 단어가 2개 이상이면 유사
        common = words1 & words2
        return len(common) >= 1 and len(common) >= min(len(words1), len(words2)) * 0.5
    
    def _calc_name_similarity(self, name1: str, name2: str) -> float:
        """두 이름의 유사도를 계산합니다 (0~1)."""
        # 간단한 Jaccard similarity
        set1 = set(name1)
        set2 = set(name2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_entities(self, entities: list, entity_type: str) -> list:
        """Deduplicate entities based on name similarity."""
        seen_names = {}
        unique = []
        
        for entity in entities:
            name = entity.name.lower().strip()
            
            if name in seen_names:
                continue
            
            # IMPORTANT:
            # - 이 단계는 "이번 실행에서 추출된 엔티티들" 내부의 중복 제거가 1차 목적입니다.
            # - 기존 DB(Neo4j)에 이미 유사 엔티티가 있다고 해서 여기서 스킵해버리면,
            #   해당 실행에서 Task/Role이 0개가 되어 BPMN이 비어버리는 문제가 발생할 수 있습니다.
            # - 따라서 Task/Role은 벡터서치 기반 'merge => skip'를 적용하지 않습니다.
            if entity_type not in ("Task", "Role"):
                # Check for similar existing entities
                try:
                    match, score, action = self.vector_search.find_similar_entity(
                        entity_type, entity.name, getattr(entity, 'description', '')
                    )
                    
                    if action == "merge" and match:
                        continue
                except Exception:
                    pass
            
            seen_names[name] = entity
            unique.append(entity)
        
        return unique

    def generate_skills(self, state: GraphState) -> GraphState:
        """Node: Generate skill documents from extracted ontology skills."""
        print("📝 Generating skill documents...")

        extracted_skills = state.get("skills", [])
        skills = []
        skill_docs = {}

        for skill in extracted_skills:
            markdown = self.skill_generator.generate(skill)

            safe_name = "".join(
                c if c.isalnum() or c in "._-" else "_"
                for c in (skill.name or "skill")
            )
            filename = f"{safe_name}.skill.md"
            filepath = Config.OUTPUT_DIR / filename

            self.skill_generator.save(markdown, str(filepath))
            skill.md_path = str(filepath)
            skill_docs[skill.skill_id] = markdown
            skills.append(skill)
        
        print(f"   Generated {len(skills)} skill documents")
        
        return {
            "skills": skills,
            "skill_docs": skill_docs,
            "current_step": "generate_dmn"
        }
    
    def generate_dmn(self, state: GraphState) -> GraphState:
        """Node: Generate DMN decision tables."""
        print("📊 Generating DMN decision tables...")
        
        decisions = state.get("dmn_decisions", [])
        rules = state.get("dmn_rules", [])
        
        if decisions:
            dmn_xml = self.dmn_generator.generate(decisions, rules)
            
            dmn_path = Config.OUTPUT_DIR / "decisions.dmn"
            self.dmn_generator.save(dmn_xml, str(dmn_path))
            
            dmn_json = self.dmn_generator.generate_json(decisions, rules)
            json_path = Config.OUTPUT_DIR / "decisions.json"
            self.dmn_generator.save(dmn_json, str(json_path))
            
            # Link decisions to roles that make them
            for role_id, decision_ids in self.role_decision_map.items():
                for decision_id in decision_ids:
                    self.neo4j.link_role_to_decision(role_id, decision_id)
            
            print(f"   Generated {len(decisions)} decision tables")
            
            return {
                "dmn_xml": dmn_xml,
                "current_step": "assemble_bpmn"
            }
        
        return {
            "dmn_xml": None,
            "current_step": "assemble_bpmn"
        }
    
    def assemble_bpmn(self, state: GraphState) -> GraphState:
        """Node: Assemble final BPMN XML - one per process."""
        print("🔧 Assembling BPMN XML...")
        
        processes = state.get("processes", [])
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        gateways = state.get("gateways", [])
        events = state.get("events", [])
        
        if not processes:
            # Create a default process if none exists
            default_process = Process(
                name="Extracted Process",
                purpose="Automatically extracted from document",
                description="Process extracted from PDF document"
            )
            processes = [default_process]
            # Assign all unassigned tasks to the default process
            for task in tasks:
                if not task.process_id:
                    task.process_id = default_process.proc_id
        
        # Generate BPMN for each process
        bpmn_xmls = {}  # process_id -> bpmn_xml
        bpmn_files = {}  # process_id -> file_path
        
        print(f"   Processing {len(processes)} process(es)...")
        
        for process in processes:
            print(f"\n   📋 Processing: {process.name} (ID: {process.proc_id})")
            
            # Filter entities by process_id
            process_tasks = [t for t in tasks if t.process_id == process.proc_id or (not t.process_id and process == processes[0])]
            process_gateways = [g for g in gateways if g.process_id == process.proc_id or (not g.process_id and process == processes[0])]
            process_events = [e for e in events if e.process_id == process.proc_id or (not e.process_id and process == processes[0])]
            
            # Assign unassigned entities to this process (only for first process)
            if process == processes[0]:
                for task in process_tasks:
                    if not task.process_id:
                        task.process_id = process.proc_id
                for gateway in process_gateways:
                    if not gateway.process_id:
                        gateway.process_id = process.proc_id
                for event in process_events:
                    if not event.process_id:
                        event.process_id = process.proc_id
            
            print(f"      Tasks: {len(process_tasks)}, Gateways: {len(process_gateways)}, Events: {len(process_events)}")
            
            # Get roles used by tasks in this process
            process_task_ids = {t.task_id for t in process_tasks}
            process_role_ids = set()
            for task_id, role_id in self.task_role_map.items():
                if task_id in process_task_ids:
                    process_role_ids.add(role_id)
            
            process_roles = [r for r in roles if r.role_id in process_role_ids]
            print(f"      Roles: {len(process_roles)}")
            
            # Get sequence flows from Neo4j for this process
            neo4j_sequence_flows = self.neo4j.get_sequence_flows(process.proc_id)
            print(f"      Sequence flows: {len(neo4j_sequence_flows)}")
            
            # Log flows with conditions
            flows_with_conditions = [f for f in neo4j_sequence_flows if f.get("condition")]
            if flows_with_conditions:
                print(f"      Conditional flows: {len(flows_with_conditions)}")
                for flow in flows_with_conditions[:3]:  # Show first 3
                    print(f"         {flow.get('from_name')} → {flow.get('to_name')}: {flow.get('condition')}")
            
            # Generate BPMN XML for this process
            bpmn_xml = self.bpmn_generator.generate(
                process=process,
                tasks=process_tasks,
                roles=process_roles,
                gateways=process_gateways,
                events=process_events,
                task_role_map=self.task_role_map,
                neo4j_sequence_flows=neo4j_sequence_flows
            )
            
            # Save to file with process-specific name
            # Use sanitized process name for filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in process.name)
            safe_name = safe_name.replace(' ', '_').replace('-', '_')[:40]  # Limit length
            # Use full proc_id (already UUID) for uniqueness
            bpmn_filename = f"process_{safe_name}_{process.proc_id}.bpmn"
            bpmn_path = Config.OUTPUT_DIR / bpmn_filename
            
            self.bpmn_generator.save(bpmn_xml, str(bpmn_path))
            print(f"      ✅ Saved: {bpmn_filename}")
            
            bpmn_xmls[process.proc_id] = bpmn_xml
            bpmn_files[process.proc_id] = str(bpmn_path)
        
        # For backward compatibility, keep the first BPMN as the main one
        main_bpmn_xml = bpmn_xmls.get(processes[0].proc_id) if processes else None
        
        # Also save the first one as process.bpmn for backward compatibility
        if main_bpmn_xml:
            default_bpmn_path = Config.OUTPUT_DIR / "process.bpmn"
            self.bpmn_generator.save(main_bpmn_xml, str(default_bpmn_path))
        
        return {
            "bpmn_xml": main_bpmn_xml,  # Backward compatibility
            "bpmn_xmls": bpmn_xmls,  # All BPMN XMLs keyed by process_id
            "bpmn_files": bpmn_files,  # File paths keyed by process_id
            "current_step": "validate_consistency"
        }
    
    def validate_consistency(self, state: GraphState) -> GraphState:
        """Node: Validate consistency of generated artifacts."""
        print("✔️ Validating consistency...")
        
        errors = []
        tasks = state.get("tasks", [])
        roles = state.get("roles", [])
        
        task_role_coverage = len(self.task_role_map)
        if tasks and task_role_coverage < len(tasks) * 0.3:
            errors.append(f"경고: {len(tasks) - task_role_coverage}개 태스크에 역할이 지정되지 않았습니다.")
        
        if len(roles) == 0 and len(tasks) > 0:
            errors.append("역할(Role)이 추출되지 않았습니다.")
        
        if errors:
            for err in errors:
                print(f"   ⚠️ {err}")
            return {
                "error": "; ".join(errors),
                "current_step": "export_artifacts"
            }
        
        print("   ✅ All validations passed")
        return {
            "error": None,
            "current_step": "export_artifacts"
        }
    
    def export_artifacts(self, state: GraphState) -> GraphState:
        """Node: Export final artifacts."""
        print("📦 Exporting artifacts...")
        
        # Print relationship statistics
        print(f"\n📊 Relationship Statistics:")
        print(f"   - Task → Task (NEXT/Sequence): {len(self.sequence_flows)}")
        print(f"   - Task → Role (PERFORMED_BY): {len(self.task_role_map)}")
        print(f"   - Task → Process (HAS_TASK): {len(self.task_process_map)}")
        print(f"   - Role → Decision (MAKES_DECISION): {sum(len(v) for v in self.role_decision_map.values())}")
        print(f"   - Entity → Document (SUPPORTED_BY): {len(self.entity_chunk_map)}")
        print(f"   - Role → Skill (HAS_SKILL): {sum(len(v) for v in self.role_skill_map.values())}")
        
        output_summary = {
            "bpmn_path": str(Config.OUTPUT_DIR / "process.bpmn"),
            "dmn_path": str(Config.OUTPUT_DIR / "decisions.dmn") if state.get("dmn_xml") else None,
            "skill_count": len(state.get("skills", [])),
            "process_count": len(state.get("processes", [])),
            "task_count": len(state.get("tasks", [])),
            "role_count": len(state.get("roles", []))
        }
        
        print(f"\n✅ Export complete!")
        print(f"   - BPMN: {output_summary['bpmn_path']}")
        if output_summary['dmn_path']:
            print(f"   - DMN: {output_summary['dmn_path']}")
        print(f"   - Skills: {output_summary['skill_count']} documents")
        print(f"   - Processes: {output_summary['process_count']}")
        print(f"   - Tasks: {output_summary['task_count']}")
        print(f"   - Roles: {output_summary['role_count']}")
        
        return {
            "current_step": "completed"
        }


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    
    workflow_handler = PDF2BPMNWorkflow()
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("ingest_pdf", workflow_handler.ingest_pdf)
    workflow.add_node("segment_sections", workflow_handler.segment_sections)
    workflow.add_node("extract_candidates", workflow_handler.extract_candidates)
    workflow.add_node("normalize_entities", workflow_handler.normalize_entities)
    workflow.add_node("generate_skills", workflow_handler.generate_skills)
    workflow.add_node("generate_dmn", workflow_handler.generate_dmn)
    workflow.add_node("assemble_bpmn", workflow_handler.assemble_bpmn)
    workflow.add_node("validate_consistency", workflow_handler.validate_consistency)
    workflow.add_node("export_artifacts", workflow_handler.export_artifacts)
    
    workflow.set_entry_point("ingest_pdf")
    
    workflow.add_edge("ingest_pdf", "segment_sections")
    workflow.add_edge("segment_sections", "extract_candidates")
    workflow.add_edge("extract_candidates", "normalize_entities")
    workflow.add_edge("normalize_entities", "generate_skills")
    workflow.add_edge("generate_skills", "generate_dmn")
    workflow.add_edge("generate_dmn", "assemble_bpmn")
    workflow.add_edge("assemble_bpmn", "validate_consistency")
    workflow.add_edge("validate_consistency", "export_artifacts")
    workflow.add_edge("export_artifacts", END)
    
    return workflow


def compile_workflow_with_checkpointer():
    """Compile workflow."""
    workflow = create_workflow()
    return workflow.compile()
