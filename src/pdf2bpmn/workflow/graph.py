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


class PDF2BPMNWorkflow:
    """Orchestrates the PDF to BPMN conversion workflow."""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.entity_extractor = EntityExtractor()
        self.neo4j = Neo4jClient()
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
        
        for section in sections:
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
            all_tasks.extend(entities["tasks"])
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
        
        # Filter valid sections
        valid_sections = [s for s in sections if s.content and len(s.content.strip()) >= 50]
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
                all_tasks.extend(entities["tasks"])
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
        
        # 4. task_process_map도 업데이트
        self._update_task_process_map(process_id_mapping)
        
        # 5. 유사한 태스크 병합 (같은 역할의 연속 업무)
        tasks, task_id_mapping = self._merge_similar_tasks(tasks)
        print(f"   Tasks after merge: {len(tasks)}")
        
        # Deduplicate tasks
        unique_tasks = self._deduplicate_entities(tasks, "Task")
        
        # Deduplicate roles
        unique_roles = self._deduplicate_entities(roles, "Role")
        
        # Deduplicate decisions
        unique_decisions = self._deduplicate_entities(decisions, "Decision")
        unique_skills = self._deduplicate_entities(skills, "Skill")
        
        print(f"   Processes: {len(processes)} → {len(unique_processes)} (merged {len(processes) - len(unique_processes)})")
        print(f"   Tasks: {len(tasks)} → {len(unique_tasks)}")
        print(f"   Roles: {len(roles)} → {len(unique_roles)}")
        print(f"   Decisions: {len(decisions)} → {len(unique_decisions)}")
        print(f"   Skills: {len(skills)} → {len(unique_skills)}")
        
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
        
        # Create sequence flows for each process based on task order
        for proc_id, proc_tasks in tasks_by_process.items():
            sorted_tasks = sorted(proc_tasks, key=lambda t: t.order)
            
            for i in range(len(sorted_tasks) - 1):
                from_task = sorted_tasks[i]
                to_task = sorted_tasks[i + 1]
                
                if (from_task.task_id, to_task.task_id) not in created_flows:
                    self.neo4j.link_task_sequence(from_task.task_id, to_task.task_id)
                    created_flows.add((from_task.task_id, to_task.task_id))
        
        # Also use Neo4j to create sequences for each process
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
                self.neo4j.link_task_to_role  # Link via HAS_TASK
                with self.neo4j.session() as session:
                    session.run("""
                        MATCH (p:Process {proc_id: $proc_id})
                        MATCH (t:Task {task_id: $task_id})
                        MERGE (p)-[:HAS_TASK]->(t)
                    """, {"proc_id": default_process_id, "task_id": task.task_id})
    
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
            # 역할별로 그룹화
            tasks_by_role = {}
            for task in proc_tasks:
                role_id = self.task_role_map.get(task.task_id, "no_role")
                if role_id not in tasks_by_role:
                    tasks_by_role[role_id] = []
                tasks_by_role[role_id].append(task)
            
            for role_id, role_tasks in tasks_by_role.items():
                # 같은 역할의 태스크들 중 유사한 것들 병합
                merged_role_tasks = self._merge_tasks_by_similarity(role_tasks, task_id_mapping)
                merged_tasks.extend(merged_role_tasks)
        
        # task_role_map 업데이트
        for old_id, new_id in task_id_mapping.items():
            if old_id in self.task_role_map and old_id != new_id:
                self.task_role_map[new_id] = self.task_role_map[old_id]
        
        return merged_tasks, task_id_mapping
    
    def _merge_tasks_by_similarity(self, tasks: list, task_id_mapping: dict) -> list:
        """이름 유사도를 기반으로 태스크를 병합합니다."""
        if len(tasks) <= 1:
            return tasks

        strong_verbs = [
            "접수", "검토", "승인", "반려", "작성", "제출", "통보", "발송", "지급", "정산",
            "심사", "개최", "참석", "배부", "확정", "확인", "등록", "처리",
        ]

        def _action_markers(name: str) -> set[str]:
            s = str(name or "").strip()
            return {v for v in strong_verbs if v in s}

        def _should_merge_task_pair(left, right) -> tuple[bool, str]:
            left_name = str(left.name or "").strip().lower()
            right_name = str(right.name or "").strip().lower()
            order_gap = abs(int(getattr(left, "order", 0) or 0) - int(getattr(right, "order", 0) or 0))
            left_actions = _action_markers(left_name)
            right_actions = _action_markers(right_name)
            shared_actions = left_actions & right_actions

            # Different core action words usually mean different tasks.
            if left_actions and right_actions and not shared_actions:
                return False, "different_core_action"

            if left_name == right_name:
                return True, "same_name"

            if left_name in right_name or right_name in left_name:
                return True, "name_contains_other"

            if self._have_same_core_words(left_name, right_name) and order_gap <= 2:
                return True, "same_core_words_nearby"

            similarity = self._calc_name_similarity(left_name, right_name)
            if order_gap <= 1 and similarity > 0.72:
                return True, "high_name_similarity_adjacent"

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
