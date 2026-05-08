"""Graph database client and schema management (Apache AGE backend)."""
import hashlib
import json
import re
from typing import Any, Optional
from contextlib import contextmanager
from uuid import uuid4
from datetime import datetime

import psycopg

from ..config import Config
from ..models.entities import (
    Document, Section, Process, Task, Role, Gateway, Event,
    Skill, DMNDecision, DMNRule, Evidence, Ambiguity,
    ReferenceChunk, ProcessDefFragment, Alias, Conflict
)


class _AgeRecord:
    """Neo4j-like record wrapper for AGE result rows."""

    def __init__(self, columns: list[str], values: list[Any]):
        self._columns = columns
        self._values = values
        self._by_name = dict(zip(columns, values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._by_name[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._by_name.get(key, default)


class _AgeResult:
    """Neo4j-like result wrapper with iterator/single()."""

    def __init__(self, records: list[_AgeRecord]):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _AgeSession:
    """Session adapter that exposes run() like neo4j.Session."""

    def __init__(self, client: "Neo4jClient"):
        self._client = client

    def run(self, query: str, params: Optional[dict] = None):
        return self._client._run_cypher(query, params or {})

    def close(self):
        return None


class Neo4jClient:
    """Compatibility client (keeps name) backed by Apache AGE."""
    PROCESS_CORE_LABELS = [
        "Process",
        "Task",
        "Role",
        "Agent",
        "Gateway",
        "Event",
        "Skill",
        "DMNDecision",
        "DMNRule",
    ]
    _GRAPH_MAX_LEN = 63
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        dsn: str = None,
        graph_name: str = None,
    ):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.age_dsn = dsn or Config.AGE_DSN
        self.graph_name = re.sub(
            r"[^0-9A-Za-z_]",
            "_",
            graph_name or Config.AGE_GRAPH_NAME,
        )
        self._conn: Optional[psycopg.Connection] = None

    @classmethod
    def build_graph_name(
        cls,
        tenant_id: str,
        todo_id: str,
        prefix: str = "g",
    ) -> str:
        """
        Build deterministic AGE graph name from tenant + todo scope.

        PostgreSQL schema identifiers are practically limited to 63 chars,
        so we sanitize and shorten with a stable hash suffix when needed.
        """
        def _safe_part(value: str, fallback: str) -> str:
            text = re.sub(r"[^0-9A-Za-z_]", "_", str(value or "")).strip("_")
            return text or fallback

        t = _safe_part(tenant_id, "tenant")
        w = _safe_part(todo_id, "todo")
        p = _safe_part(prefix, "g")
        base = f"{p}_{t}_{w}"
        if len(base) <= cls._GRAPH_MAX_LEN:
            return base

        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        keep = max(8, cls._GRAPH_MAX_LEN - (len(p) + 1 + len(digest) + 1))
        trimmed = base[:keep].rstrip("_")
        return f"{p}_{trimmed}_{digest}"

    def _connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.age_dsn, autocommit=True)
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS age")
                cur.execute('LOAD \'age\'')
                cur.execute('SET search_path = ag_catalog, "$user", public')
        return self._conn
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None
    
    @contextmanager
    def session(self):
        session = _AgeSession(self)
        try:
            yield session
        finally:
            session.close()

    def _to_cypher_literal(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return "[" + ", ".join(self._to_cypher_literal(v) for v in value) + "]"
        if isinstance(value, dict):
            pairs = []
            for k, v in value.items():
                key = re.sub(r"[^0-9A-Za-z_]", "_", str(k))
                pairs.append(f"{key}: {self._to_cypher_literal(v)}")
            return "{" + ", ".join(pairs) + "}"
        return json.dumps(str(value), ensure_ascii=False)

    def _apply_params(self, query: str, params: dict[str, Any]) -> str:
        rendered = query
        for key, value in (params or {}).items():
            rendered = re.sub(
                rf"\${re.escape(key)}(?![A-Za-z0-9_])",
                self._to_cypher_literal(value),
                rendered,
            )
        return rendered

    def _split_return_items(self, return_clause: str) -> list[str]:
        items = []
        buf = []
        depth = 0
        in_s = False
        in_d = False
        i = 0
        while i < len(return_clause):
            ch = return_clause[i]
            if ch == "'" and not in_d:
                in_s = not in_s
                buf.append(ch)
                i += 1
                continue
            if ch == '"' and not in_s:
                in_d = not in_d
                buf.append(ch)
                i += 1
                continue
            if in_s or in_d:
                buf.append(ch)
                i += 1
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                item = "".join(buf).strip()
                if item:
                    items.append(item)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            items.append(tail)
        return items

    def _extract_return_aliases(self, query: str) -> list[str]:
        q = query
        idx = q.upper().rfind("RETURN")
        if idx < 0:
            return []
        clause = q[idx + len("RETURN"):].strip()
        upper_clause = clause.upper()
        for stopper in [" ORDER BY ", " LIMIT ", " SKIP ", " WITH "]:
            sidx = upper_clause.find(stopper)
            if sidx >= 0:
                clause = clause[:sidx].strip()
                upper_clause = clause.upper()
        raw_items = self._split_return_items(clause)
        aliases: list[str] = []
        for i, item in enumerate(raw_items):
            m = re.search(r"\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", item, flags=re.I)
            if m:
                aliases.append(m.group(1))
            else:
                aliases.append(f"col_{i+1}")
        seen = set()
        unique = []
        for i, a in enumerate(aliases):
            candidate = a
            n = 2
            while candidate in seen:
                candidate = f"{a}_{n}"
                n += 1
            seen.add(candidate)
            unique.append(candidate)
        return unique

    def _parse_agtype_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float, bool, list, dict)):
            return value
        s = str(value).strip()
        if not s:
            return s
        for suffix in ("::vertex", "::edge", "::path", "::agtype"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                break
        try:
            return json.loads(s)
        except Exception:
            pass
        if s.startswith('"') and s.endswith('"'):
            try:
                return json.loads(s)
            except Exception:
                return s[1:-1]
        return s

    def _run_cypher(self, query: str, params: dict[str, Any]) -> _AgeResult:
        rendered = self._apply_params(query, params or {})
        aliases = self._extract_return_aliases(rendered)
        wrapped_query = rendered
        if not aliases:
            wrapped_query = f"{rendered}\nRETURN 1 AS _ok"
            aliases = ["_ok"]
        cols = ", ".join(f"{a} agtype" for a in aliases)
        safe_graph_name = self.graph_name.replace("'", "''")
        sql = f"SELECT * FROM cypher('{safe_graph_name}', $$ {wrapped_query} $$) AS ({cols})"

        conn = self._connect()
        records: list[_AgeRecord] = []
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            for row in rows:
                parsed = [self._parse_agtype_value(v) for v in row]
                records.append(_AgeRecord(aliases, parsed))
        return _AgeResult(records)

    def _label_predicate(self, var_name: str, labels: list[str]) -> str:
        safe_labels = [re.sub(r"[^0-9A-Za-z_]", "", str(l)) for l in (labels or []) if str(l).strip()]
        safe_labels = [l for l in safe_labels if l]
        if not safe_labels:
            return "false"
        return " OR ".join(f"'{lbl}' IN labels({var_name})" for lbl in safe_labels)

    def verify_connection(self) -> bool:
        """Verify AGE/PostgreSQL connection is working."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def init_schema(self):
        """
        Initialize AGE graph.
        Neo4j constraints/fulltext/vector indexes are replaced by application-level logic.
        """
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                (self.graph_name,),
            )
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"SELECT ag_catalog.create_graph('{self.graph_name}')")
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)."""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def clear_process_core_labels(self, labels: Optional[list[str]] = None) -> dict[str, Any]:
        """Delete only process-core labels and keep unrelated graph data."""
        target_labels = labels or self.PROCESS_CORE_LABELS
        if not target_labels:
            return {"labels": [], "deleted_nodes": 0}

        where_clause = self._label_predicate("n", target_labels)

        try:
            with self.session() as session:
                before_query = f"""
                MATCH (n)
                WHERE {where_clause}
                RETURN count(n)
                """
                before_record = session.run(before_query).single()
                before_count = int(before_record[0]) if before_record else 0

                delete_query = f"""
                MATCH (n)
                WHERE {where_clause}
                DETACH DELETE n
                """
                session.run(delete_query)
        except psycopg.errors.InvalidSchemaName:
            # B안(tenant+todo graph)에서 첫 실행 시 그래프가 아직 없을 수 있다.
            # 이 경우 cleanup은 no-op으로 처리한다.
            return {"labels": target_labels, "deleted_nodes": 0}

        return {
            "labels": target_labels,
            "deleted_nodes": before_count,
        }
    
    # ==================== Create Operations ====================
    
    def create_document(self, doc: Document) -> str:
        """Create a Document node."""
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.title = $title,
            d.source = $source,
            d.page_count = $page_count,
            d.uploaded_at = $uploaded_at,
            d.version = $version,
            d.created_by = $created_by
        RETURN d.doc_id
        """
        with self.session() as session:
            result = session.run(query, {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "page_count": doc.page_count,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "version": doc.version,
                "created_by": doc.created_by
            })
            return result.single()[0]
    
    def create_section(self, section: Section) -> str:
        """Create a Section node and link to Document."""
        create_query = """
        MERGE (s:Section {section_id: $section_id})
        SET s.heading = $heading,
            s.level = $level,
            s.page_from = $page_from,
            s.page_to = $page_to,
            s.content = $content
        RETURN s.section_id
        """
        link_query = """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (s:Section {section_id: $section_id})
        MERGE (d)-[:HAS_SECTION]->(s)
        RETURN s.section_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "section_id": section.section_id,
                "heading": section.heading,
                "level": section.level,
                "page_from": section.page_from,
                "page_to": section.page_to,
                "content": section.content[:5000] if section.content else ""
            })
            row = result.single()
            section_id = (row[0] if row else section.section_id)
            if section.doc_id:
                try:
                    session.run(
                        link_query,
                        {"doc_id": section.doc_id, "section_id": section_id},
                    )
                except Exception:
                    pass
            return section_id
    
    def create_chunk(self, chunk: ReferenceChunk) -> str:
        """Create a ReferenceChunk node."""
        create_query = """
        MERGE (c:ReferenceChunk {chunk_id: $chunk_id})
        SET c.page = $page,
            c.span = $span,
            c.text = $text,
            c.hash = $hash,
            c.embedding = $embedding
        RETURN c.chunk_id
        """
        link_query = """
        MATCH (c:ReferenceChunk {chunk_id: $chunk_id})
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c)-[:FROM_DOCUMENT]->(d)
        RETURN c.chunk_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "span": chunk.span,
                "text": chunk.text,
                "hash": chunk.hash,
                "embedding": chunk.embedding
            })
            row = result.single()
            chunk_id = (row[0] if row else chunk.chunk_id)
            if chunk.doc_id:
                try:
                    session.run(
                        link_query,
                        {"chunk_id": chunk_id, "doc_id": chunk.doc_id},
                    )
                except Exception:
                    pass
            return chunk_id
    
    def create_process(self, process: Process) -> str:
        """Create a Process node."""
        query = """
        MERGE (p:Process {proc_id: $proc_id})
        SET p.name = $name,
            p.purpose = $purpose,
            p.description = $description,
            p.triggers = $triggers,
            p.outcomes = $outcomes,
            p.version = $version,
            p.created_by = $created_by
        RETURN p.proc_id
        """
        with self.session() as session:
            result = session.run(query, {
                "proc_id": process.proc_id,
                "name": process.name,
                "purpose": process.purpose,
                "description": process.description,
                "triggers": process.triggers,
                "outcomes": process.outcomes,
                "version": process.version,
                "created_by": process.created_by
            })
            return result.single()[0]
    
    def create_task(self, task: Task) -> str:
        """Create a Task node and link to Process."""
        create_query = """
        MERGE (t:Task {task_id: $task_id})
        SET t.name = $name,
            t.task_type = $task_type,
            t.description = $description,
            t.instruction = $instruction,
            t.task_order = $order,
            t.version = $version,
            t.created_by = $created_by
        RETURN t.task_id
        """
        link_query = """
        MATCH (p:Process {proc_id: $process_id})
        MATCH (t:Task {task_id: $task_id})
        MERGE (p)-[:HAS_TASK]->(t)
        RETURN t.task_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "task_id": task.task_id,
                "process_id": task.process_id,
                "name": task.name,
                "task_type": task.task_type.value,
                "description": task.description,
                "instruction": getattr(task, "instruction", "") or "",
                "order": task.order,
                "version": task.version,
                "created_by": task.created_by
            })
            row = result.single()
            task_id = row[0] if row else task.task_id
            if task.process_id:
                try:
                    session.run(
                        link_query,
                        {"process_id": task.process_id, "task_id": task_id},
                    )
                except Exception:
                    pass
            return task_id
    
    def create_role(self, role: Role) -> str:
        """Create a Role node."""
        query = """
        MERGE (r:Role {role_id: $role_id})
        SET r.name = $name,
            r.org_unit = $org_unit,
            r.persona_hint = $persona_hint,
            r.version = $version
        RETURN r.role_id
        """
        with self.session() as session:
            result = session.run(query, {
                "role_id": role.role_id,
                "name": role.name,
                "org_unit": role.org_unit,
                "persona_hint": role.persona_hint,
                "version": role.version
            })
            return result.single()[0]
    
    def create_gateway(self, gateway: Gateway) -> str:
        """Create a Gateway node."""
        create_query = """
        MERGE (g:Gateway {gateway_id: $gateway_id})
        SET g.gateway_type = $gateway_type,
            g.condition = $condition,
            g.description = $description
        RETURN g.gateway_id
        """
        link_query = """
        MATCH (p:Process {proc_id: $process_id})
        MATCH (g:Gateway {gateway_id: $gateway_id})
        MERGE (p)-[:HAS_GATEWAY]->(g)
        RETURN g.gateway_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "gateway_id": gateway.gateway_id,
                "process_id": gateway.process_id,
                "gateway_type": gateway.gateway_type.value,
                "condition": gateway.condition,
                "description": gateway.description
            })
            row = result.single()
            gateway_id = row[0] if row else gateway.gateway_id
            if gateway.process_id:
                try:
                    session.run(
                        link_query,
                        {"process_id": gateway.process_id, "gateway_id": gateway_id},
                    )
                except Exception:
                    pass
            return gateway_id
    
    def create_event(self, event: Event) -> str:
        """Create an Event node."""
        create_query = """
        MERGE (e:Event {event_id: $event_id})
        SET e.event_type = $event_type,
            e.name = $name,
            e.trigger = $trigger
        RETURN e.event_id
        """
        link_query = """
        MATCH (p:Process {proc_id: $process_id})
        MATCH (e:Event {event_id: $event_id})
        MERGE (p)-[:HAS_EVENT]->(e)
        RETURN e.event_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "event_id": event.event_id,
                "process_id": event.process_id,
                "event_type": event.event_type.value,
                "name": event.name,
                "trigger": event.trigger
            })
            row = result.single()
            event_id = row[0] if row else event.event_id
            if event.process_id:
                try:
                    session.run(
                        link_query,
                        {"process_id": event.process_id, "event_id": event_id},
                    )
                except Exception:
                    pass
            return event_id
    
    def create_skill(self, skill: Skill) -> str:
        """Create a Skill node."""
        query = """
        MERGE (s:Skill {skill_id: $skill_id})
        SET s.name = $name,
            s.summary = $summary,
            s.purpose = $purpose,
            s.inputs = $inputs,
            s.outputs = $outputs,
            s.preconditions = $preconditions,
            s.procedure = $procedure,
            s.exceptions = $exceptions,
            s.tools = $tools,
            s.md_path = $md_path,
            s.version = $version
        RETURN s.skill_id
        """
        with self.session() as session:
            result = session.run(query, {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "summary": skill.summary,
                "purpose": skill.purpose,
                "inputs": str(skill.inputs),
                "outputs": str(skill.outputs),
                "preconditions": skill.preconditions,
                "procedure": skill.procedure,
                "exceptions": skill.exceptions,
                "tools": skill.tools,
                "md_path": skill.md_path,
                "version": skill.version
            })
            return result.single()[0]
    
    def create_decision(self, decision: DMNDecision) -> str:
        """Create a DMNDecision node."""
        query = """
        MERGE (d:DMNDecision {decision_id: $decision_id})
        SET d.name = $name,
            d.description = $description,
            d.input_data = $input_data,
            d.output_data = $output_data
        RETURN d.decision_id
        """
        with self.session() as session:
            result = session.run(query, {
                "decision_id": decision.decision_id,
                "name": decision.name,
                "description": decision.description,
                "input_data": decision.input_data,
                "output_data": decision.output_data
            })
            return result.single()[0]
    
    def create_rule(self, rule: DMNRule) -> str:
        """Create a DMNRule node and link to Decision."""
        create_query = """
        MERGE (r:DMNRule {rule_id: $rule_id})
        SET r.when_condition = $when_condition,
            r.then_result = $then_result,
            r.confidence = $confidence
        RETURN r.rule_id
        """
        link_query = """
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MATCH (r:DMNRule {rule_id: $rule_id})
        MERGE (d)-[:HAS_RULE]->(r)
        RETURN r.rule_id
        """
        with self.session() as session:
            result = session.run(create_query, {
                "rule_id": rule.rule_id,
                "decision_id": rule.decision_id,
                "when_condition": rule.when,
                "then_result": rule.then,
                "confidence": rule.confidence
            })
            row = result.single()
            rule_id = row[0] if row else rule.rule_id
            if rule.decision_id:
                try:
                    session.run(
                        link_query,
                        {"decision_id": rule.decision_id, "rule_id": rule_id},
                    )
                except Exception:
                    pass
            return rule_id
    
    def create_ambiguity(self, ambiguity: Ambiguity) -> str:
        """Create an Ambiguity node for HITL questions."""
        query = """
        MERGE (a:Ambiguity {amb_id: $amb_id})
        SET a.entity_type = $entity_type,
            a.entity_id = $entity_id,
            a.question = $question,
            a.options = $options,
            a.status = $status,
            a.answer = $answer
        RETURN a.amb_id
        """
        with self.session() as session:
            result = session.run(query, {
                "amb_id": ambiguity.amb_id,
                "entity_type": ambiguity.entity_type,
                "entity_id": ambiguity.entity_id,
                "question": ambiguity.question,
                "options": ambiguity.options,
                "status": ambiguity.status.value,
                "answer": ambiguity.answer
            })
            return result.single()[0]
    
    def create_evidence_link(
        self, 
        entity_type: str, 
        entity_id: str, 
        chunk_id: str
    ):
        """Create SUPPORTED_BY relationship between entity and chunk."""
        query = f"""
        MATCH (e:{entity_type} {{{entity_type.lower()}_id: $entity_id}})
        MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
        MERGE (e)-[:SUPPORTED_BY]->(c)
        """
        # Handle different ID field names
        id_field_map = {
            "Process": "proc_id",
            "Task": "task_id",
            "Role": "role_id",
            "Gateway": "gateway_id",
            "Event": "event_id",
            "Skill": "skill_id",
            "DMNDecision": "decision_id",
            "DMNRule": "rule_id"
        }
        id_field = id_field_map.get(entity_type, f"{entity_type.lower()}_id")
        
        query = f"""
        MATCH (e:{entity_type} {{{id_field}: $entity_id}})
        MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
        MERGE (e)-[:SUPPORTED_BY]->(c)
        """
        with self.session() as session:
            session.run(query, {
                "entity_id": entity_id,
                "chunk_id": chunk_id
            })
    
    def link_task_to_role(self, task_id: str, role_id: str):
        """Create PERFORMED_BY relationship between Task and Role."""
        query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (r:Role {role_id: $role_id})
        MERGE (t)-[:PERFORMED_BY]->(r)
        """
        with self.session() as session:
            session.run(query, {"task_id": task_id, "role_id": role_id})
    
    def link_task_to_skill(self, task_id: str, skill_id: str):
        """Create USES_SKILL relationship between Task and Skill."""
        query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (t)-[:USES_SKILL]->(s)
        """
        with self.session() as session:
            session.run(query, {"task_id": task_id, "skill_id": skill_id})
    
    def link_process_to_decision(self, proc_id: str, decision_id: str):
        """Create USES_DECISION relationship."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (p)-[:USES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"proc_id": proc_id, "decision_id": decision_id})
    
    def link_role_to_skill(self, role_id: str, skill_id: str):
        """Create HAS_SKILL relationship between Role and Skill."""
        query = """
        MATCH (r:Role {role_id: $role_id})
        MATCH (s:Skill {skill_id: $skill_id})
        MERGE (r)-[:HAS_SKILL]->(s)
        """
        with self.session() as session:
            session.run(query, {"role_id": role_id, "skill_id": skill_id})

    def create_agent(self, agent_id: str, name: str = "", role: str = "", tenant_id: str = ""):
        """Create or update an Agent node."""
        query = """
        MERGE (a:Agent {agent_id: $agent_id})
        SET a.name = $name,
            a.role = $role,
            a.tenant_id = $tenant_id
        RETURN a.agent_id
        """
        with self.session() as session:
            session.run(
                query,
                {
                    "agent_id": agent_id,
                    "name": name or "",
                    "role": role or "",
                    "tenant_id": tenant_id or "",
                },
            )

    def link_role_to_agent_in_process(self, proc_id: str, role_name: str, agent_id: str):
        """Link process role to assigned agent by role name."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(:Task)-[:PERFORMED_BY]->(r:Role)
        WHERE toLower(trim(r.name)) = toLower(trim($role_name))
        MATCH (a:Agent {agent_id: $agent_id})
        MERGE (r)-[:ASSIGNED_AGENT]->(a)
        """
        with self.session() as session:
            session.run(
                query,
                {"proc_id": proc_id, "role_name": role_name or "", "agent_id": agent_id},
            )

    def link_agent_to_skill_by_name(self, agent_id: str, skill_name: str):
        """Link agent to skill by exact skill name."""
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        MATCH (s:Skill)
        WHERE toLower(trim(s.name)) = toLower(trim($skill_name))
        MERGE (a)-[:USES_SKILL]->(s)
        """
        with self.session() as session:
            session.run(query, {"agent_id": agent_id, "skill_name": skill_name or ""})
    
    def link_role_to_decision(self, role_id: str, decision_id: str):
        """Create MAKES_DECISION relationship between Role and DMNDecision."""
        query = """
        MATCH (r:Role {role_id: $role_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (r)-[:MAKES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"role_id": role_id, "decision_id": decision_id})
    
    def link_skill_to_decision(self, skill_id: str, decision_id: str):
        """Create USES_DECISION relationship between Skill and DMNDecision."""
        query = """
        MATCH (s:Skill {skill_id: $skill_id})
        MATCH (d:DMNDecision {decision_id: $decision_id})
        MERGE (s)-[:USES_DECISION]->(d)
        """
        with self.session() as session:
            session.run(query, {"skill_id": skill_id, "decision_id": decision_id})
    
    def link_chunk_to_document(self, chunk_id: str, doc_id: str):
        """Create FROM_DOCUMENT relationship between ReferenceChunk and Document."""
        query = """
        MATCH (c:ReferenceChunk {chunk_id: $chunk_id})
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c)-[:FROM_DOCUMENT]->(d)
        """
        with self.session() as session:
            session.run(query, {"chunk_id": chunk_id, "doc_id": doc_id})

    def link_task_to_process(self, task_id: str, proc_id: str):
        """Create HAS_TASK relationship between Process and Task."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        MATCH (t:Task {task_id: $task_id})
        MERGE (p)-[:HAS_TASK]->(t)
        """
        with self.session() as session:
            session.run(query, {"task_id": task_id, "proc_id": proc_id})
    
    def link_task_sequence(self, from_task_id: str, to_task_id: str, condition: str = None):
        """Create NEXT (sequence flow) relationship between Tasks."""
        query = """
        MATCH (t1:Task {task_id: $from_task_id})
        MATCH (t2:Task {task_id: $to_task_id})
        MERGE (t1)-[r:NEXT]->(t2)
        SET r.condition = $condition
        """
        with self.session() as session:
            session.run(query, {
                "from_task_id": from_task_id,
                "to_task_id": to_task_id,
                "condition": condition
            })
    
    def link_gateway_to_task(self, gateway_id: str, task_id: str, condition: str = None, is_incoming: bool = False):
        """Create flow relationship from Gateway to Task (outgoing from gateway)."""
        if is_incoming:
            # Legacy support - use link_task_to_gateway instead
            self.link_task_to_gateway(task_id, gateway_id)
            return
        
        query = """
        MATCH (g:Gateway {gateway_id: $gateway_id})
        MATCH (t:Task {task_id: $task_id})
        MERGE (g)-[r:NEXT]->(t)
        SET r.condition = $condition
        """
        with self.session() as session:
            session.run(query, {
                "gateway_id": gateway_id,
                "task_id": task_id,
                "condition": condition or ""
            })
    
    def link_task_to_gateway(self, task_id: str, gateway_id: str, condition: str = None):
        """Create flow relationship from Task to Gateway (incoming to gateway)."""
        query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (g:Gateway {gateway_id: $gateway_id})
        MERGE (t)-[r:NEXT]->(g)
        SET r.condition = $condition
        """
        with self.session() as session:
            session.run(query, {
                "task_id": task_id,
                "gateway_id": gateway_id,
                "condition": condition or ""
            })
    
    def link_event_to_task(self, event_id: str, task_id: str, is_start: bool = True):
        """Create flow relationship between Event and Task."""
        if is_start:
            query = """
            MATCH (e:Event {event_id: $event_id})
            MATCH (t:Task {task_id: $task_id})
            MERGE (e)-[:NEXT]->(t)
            """
        else:
            query = """
            MATCH (t:Task {task_id: $task_id})
            MATCH (e:Event {event_id: $event_id})
            MERGE (t)-[:NEXT]->(e)
            """
        with self.session() as session:
            session.run(query, {"event_id": event_id, "task_id": task_id})
    
    def create_task_sequence_for_process(self, proc_id: str):
        """Create NEXT relationships between tasks in a process based on order."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(t:Task)
        WITH t ORDER BY coalesce(t.task_order, 0)
        WITH collect(t) as tasks
        UNWIND range(0, size(tasks)-2) as i
        WITH tasks[i] as t1, tasks[i+1] as t2
        MERGE (t1)-[:NEXT]->(t2)
        """
        with self.session() as session:
            session.run(query, {"proc_id": proc_id})
    
    def create_all_relationships(
        self,
        task_role_map: dict,
        task_process_map: dict,
        role_decision_map: dict,
        entity_chunk_map: dict,
        role_skill_map: dict = None
    ):
        """Create all relationships in batch."""
        with self.session() as session:
            # Task -> Role (PERFORMED_BY)
            for task_id, role_id in task_role_map.items():
                session.run("""
                    MATCH (t:Task {task_id: $task_id})
                    MATCH (r:Role {role_id: $role_id})
                    MERGE (t)-[:PERFORMED_BY]->(r)
                """, {"task_id": task_id, "role_id": role_id})
            
            # Task -> Process (belongs to, via HAS_TASK from Process)
            for task_id, proc_id in task_process_map.items():
                session.run("""
                    MATCH (p:Process {proc_id: $proc_id})
                    MATCH (t:Task {task_id: $task_id})
                    MERGE (p)-[:HAS_TASK]->(t)
                """, {"task_id": task_id, "proc_id": proc_id})
            
            # Role -> DMNDecision (MAKES_DECISION)
            for role_id, decision_ids in role_decision_map.items():
                for decision_id in decision_ids:
                    session.run("""
                        MATCH (r:Role {role_id: $role_id})
                        MATCH (d:DMNDecision {decision_id: $decision_id})
                        MERGE (r)-[:MAKES_DECISION]->(d)
                    """, {"role_id": role_id, "decision_id": decision_id})
            
            # Entity -> ReferenceChunk (SUPPORTED_BY) for evidence
            id_field_map = {
                "Process": "proc_id",
                "Task": "task_id",
                "Role": "role_id",
                "Gateway": "gateway_id",
                "Event": "event_id",
                "Skill": "skill_id",
                "DMNDecision": "decision_id",
                "DMNRule": "rule_id"
            }
            
            for entity_id, chunk_id in entity_chunk_map.items():
                # Try to match with each entity type
                for entity_type, id_field in id_field_map.items():
                    try:
                        result = session.run(f"""
                            MATCH (e:{entity_type} {{{id_field}: $entity_id}})
                            MATCH (c:ReferenceChunk {{chunk_id: $chunk_id}})
                            MERGE (e)-[:SUPPORTED_BY]->(c)
                            RETURN e
                        """, {"entity_id": entity_id, "chunk_id": chunk_id})
                        if result.single():
                            break
                    except:
                        continue
            
            # Process -> Skill (HAS_SKILL): derived from Task->Role and Task->Process mappings
            if role_skill_map and task_role_map and task_process_map:
                created = set()
                for task_id, role_id in task_role_map.items():
                    proc_id = task_process_map.get(task_id)
                    if not proc_id:
                        continue
                    for skill_id in role_skill_map.get(role_id, []):
                        key = (proc_id, skill_id)
                        if key in created:
                            continue
                        created.add(key)
                        session.run(
                            """
                            MATCH (p:Process {proc_id: $proc_id})
                            MATCH (s:Skill {skill_id: $skill_id})
                            MERGE (p)-[:HAS_SKILL]->(s)
                            """,
                            {"proc_id": proc_id, "skill_id": skill_id},
                        )
    
    # ==================== Query Operations ====================
    
    def get_all_processes(self) -> list[dict]:
        """Get all processes."""
        query = """
        MATCH (p:Process)
        RETURN p {.*} as process
        ORDER BY p.name
        """
        with self.session() as session:
            result = session.run(query)
            return [record["process"] for record in result]
    
    def get_process_with_details(self, proc_id: str) -> dict:
        """Get process with all related entities."""
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        OPTIONAL MATCH (p)-[:HAS_TASK]->(t:Task)
        OPTIONAL MATCH (p)-[:HAS_GATEWAY]->(g:Gateway)
        OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
        OPTIONAL MATCH (r)-[:ASSIGNED_AGENT]->(a:Agent)
        OPTIONAL MATCH (a)-[:USES_SKILL]->(agent_skill:Skill)
        OPTIONAL MATCH (p)-[:HAS_SKILL]->(ps:Skill)
        RETURN p {.*} as process,
               collect(DISTINCT t {.*}) as tasks,
               collect(DISTINCT g {.*}) as gateways,
               collect(DISTINCT e {.*}) as events,
               collect(DISTINCT r {.*}) as roles,
               collect(DISTINCT a {.*}) as agents,
               collect(DISTINCT agent_skill {.*}) as agent_skills,
               collect(DISTINCT ps {.*}) as process_skills
        """
        with self.session() as session:
            result = session.run(query, {"proc_id": proc_id})
            record = result.single()
            if record:
                raw_skills = []
                raw_skills.extend(record["agent_skills"] or [])
                raw_skills.extend(record["process_skills"] or [])
                dedup_skills = []
                seen_skill_ids = set()
                for s in raw_skills:
                    if not isinstance(s, dict):
                        continue
                    sid = str(s.get("skill_id") or "").strip()
                    if not sid or sid in seen_skill_ids:
                        continue
                    seen_skill_ids.add(sid)
                    dedup_skills.append(s)
                return {
                    "process": record["process"],
                    "tasks": record["tasks"],
                    "gateways": record["gateways"],
                    "events": record["events"],
                    "roles": record["roles"],
                    "agents": record["agents"],
                    "skills": dedup_skills,
                }
            return None
    
    def get_process_entities_for_bpmn(self, proc_id: str) -> dict:
        """Get all entities for a process to generate BPMN.
        
        Returns:
            dict with Process, Task, Gateway, Event, Role objects and task_role_map
        """
        from ..models.entities import (
            Process, Task, Role, Gateway, Event,
            TaskType, GatewayType, EventType
        )
        
        query = """
        MATCH (p:Process {proc_id: $proc_id})
        OPTIONAL MATCH (p)-[:HAS_TASK]->(t:Task)
        OPTIONAL MATCH (p)-[:HAS_GATEWAY]->(g:Gateway)
        OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
        RETURN p,
               collect(DISTINCT t) as tasks,
               collect(DISTINCT g) as gateways,
               collect(DISTINCT e) as events,
               collect(DISTINCT r) as roles
        """
        
        with self.session() as session:
            result = session.run(query, {"proc_id": proc_id})
            record = result.single()
            
            if not record or not record["p"]:
                return None
            
            # Convert Process
            proc_data = dict(record["p"])
            process = Process(
                proc_id=proc_data["proc_id"],
                name=proc_data.get("name", ""),
                purpose=proc_data.get("purpose", ""),
                description=proc_data.get("description", ""),
                triggers=proc_data.get("triggers", []),
                outcomes=proc_data.get("outcomes", [])
            )
            
            # Convert Tasks and build task_role_map in one query
            tasks = []
            task_role_map = {}
            
            # Get all task-role relationships for this process in one query
            task_role_query = """
            MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(t:Task)-[:PERFORMED_BY]->(r:Role)
            RETURN t.task_id as task_id, r.role_id as role_id
            """
            task_role_result = session.run(task_role_query, {"proc_id": proc_id})
            for tr_record in task_role_result:
                task_role_map[tr_record["task_id"]] = tr_record["role_id"]
            
            # Convert tasks
            for task_data in record["tasks"]:
                if not task_data:
                    continue
                task_dict = dict(task_data)
                task = Task(
                    task_id=task_dict["task_id"],
                    process_id=proc_id,
                    name=task_dict.get("name", ""),
                    task_type=TaskType(task_dict.get("task_type", "human")),
                    description=task_dict.get("description", ""),
                    instruction=task_dict.get("instruction", ""),
                    order=task_dict.get("task_order", task_dict.get("order", 0))
                )
                tasks.append(task)
            
            # Convert Gateways
            gateways = []
            for gateway_data in record["gateways"]:
                if not gateway_data:
                    continue
                gateway_dict = dict(gateway_data)
                gateway = Gateway(
                    gateway_id=gateway_dict["gateway_id"],
                    process_id=proc_id,
                    name=gateway_dict.get("name", ""),
                    gateway_type=GatewayType(gateway_dict.get("gateway_type", "exclusive")),
                    condition=gateway_dict.get("condition", ""),
                    description=gateway_dict.get("description", "")
                )
                gateways.append(gateway)
            
            # Convert Events
            events = []
            for event_data in record["events"]:
                if not event_data:
                    continue
                event_dict = dict(event_data)
                event = Event(
                    event_id=event_dict["event_id"],
                    process_id=proc_id,
                    event_type=EventType(event_dict.get("event_type", "start")),
                    name=event_dict.get("name", ""),
                    trigger=event_dict.get("trigger", "")
                )
                events.append(event)
            
            # Convert Roles (distinct)
            roles = []
            seen_role_ids = set()
            for role_data in record["roles"]:
                if not role_data:
                    continue
                role_dict = dict(role_data)
                role_id = role_dict.get("role_id")
                if role_id and role_id not in seen_role_ids:
                    role = Role(
                        role_id=role_id,
                        name=role_dict.get("name", ""),
                        org_unit=role_dict.get("org_unit", ""),
                        persona_hint=role_dict.get("persona_hint", "")
                    )
                    roles.append(role)
                    seen_role_ids.add(role_id)
            
            return {
                "process": process,
                "tasks": tasks,
                "gateways": gateways,
                "events": events,
                "roles": roles,
                "task_role_map": task_role_map
            }
    
    def get_open_ambiguities(self) -> list[dict]:
        """Get all open ambiguity questions."""
        query = """
        MATCH (a:Ambiguity {status: 'open'})
        RETURN a {.*} as ambiguity
        ORDER BY a.created_at
        """
        with self.session() as session:
            result = session.run(query)
            return [record["ambiguity"] for record in result]
    
    def resolve_ambiguity(self, amb_id: str, answer: str):
        """Resolve an ambiguity with user's answer."""
        query = """
        MATCH (a:Ambiguity {amb_id: $amb_id})
        SET a.status = 'resolved',
            a.answer = $answer,
            a.resolved_at = $resolved_at
        RETURN a.amb_id
        """
        with self.session() as session:
            session.run(
                query,
                {"amb_id": amb_id, "answer": answer, "resolved_at": datetime.utcnow().isoformat()},
            )
    
    def get_sequence_flows(self, process_id: str = None) -> list[dict]:
        """Get all NEXT relationships with their conditions.
        
        Args:
            process_id: Optional process ID to filter flows within a specific process.
                       If None, returns all flows.
        
        Returns:
            list of {from_id, from_type, from_name, to_id, to_type, to_name, condition}
        """
        if process_id:
            # AGE compatibility: avoid Neo4j-only label predicate syntax (from:Task, CASE WHEN from:Task ...)
            query = """
            MATCH (p:Process {proc_id: $process_id})
            MATCH (p)-[rf]->(from)
            WHERE type(rf) IN ['HAS_TASK', 'HAS_GATEWAY', 'HAS_EVENT']
            MATCH (from)-[r:NEXT]->(to)
            MATCH (p)-[rt]->(to)
            WHERE type(rt) IN ['HAS_TASK', 'HAS_GATEWAY', 'HAS_EVENT']
              AND (
                ('Task' IN labels(from))
                OR ('Gateway' IN labels(from))
                OR ('Event' IN labels(from))
              )
              AND (
                ('Task' IN labels(to))
                OR ('Gateway' IN labels(to))
                OR ('Event' IN labels(to))
              )
            RETURN labels(from) as from_labels,
                   properties(from) as from_props,
                   labels(to) as to_labels,
                   properties(to) as to_props,
                   properties(r) as rel_props
            """
            params = {"process_id": process_id}
        else:
            query = """
            MATCH (from)-[r:NEXT]->(to)
            WHERE (
                ('Task' IN labels(from))
                OR ('Gateway' IN labels(from))
                OR ('Event' IN labels(from))
            )
            AND (
                ('Task' IN labels(to))
                OR ('Gateway' IN labels(to))
                OR ('Event' IN labels(to))
            )
            RETURN labels(from) as from_labels,
                   properties(from) as from_props,
                   labels(to) as to_labels,
                   properties(to) as to_props,
                   properties(r) as rel_props
            """
            params = {}
        
        with self.session() as session:
            result = session.run(query, params)
            flows = []

            def _entity_info(labels: list[str], props: dict) -> tuple[str, str, str]:
                labels_set = set(labels or [])
                p = props or {}
                if "Task" in labels_set:
                    return str(p.get("task_id") or ""), "Task", str(p.get("name") or "")
                if "Gateway" in labels_set:
                    return str(p.get("gateway_id") or ""), "Gateway", str(p.get("name") or "")
                if "Event" in labels_set:
                    return str(p.get("event_id") or ""), "Event", str(p.get("name") or "")
                return "", "", str(p.get("name") or "")

            for record in result:
                from_id, from_type, from_name = _entity_info(
                    record.get("from_labels") or [],
                    record.get("from_props") or {},
                )
                to_id, to_type, to_name = _entity_info(
                    record.get("to_labels") or [],
                    record.get("to_props") or {},
                )
                rel_props = record.get("rel_props") or {}
                if not from_id or not to_id:
                    continue
                flows.append({
                    "from_id": from_id,
                    "from_type": from_type,
                    "from_name": from_name,
                    "to_id": to_id,
                    "to_type": to_type,
                    "to_name": to_name,
                    "condition": rel_props.get("condition"),
                })
            return flows

    def get_process_graph_elements(self, proc_id: str) -> dict:
        """
        Return a process subgraph as Cytoscape-compatible elements.
        This is intended for UI visualization of the *actual extracted Neo4j graph*.

        Output:
          {
            "process_id": "<proc_id>",
            "elements": [ {data:{id,...}}, {data:{id,source,target,...}}, ... ],
            "counts": {"nodes": N, "edges": M}
          }
        """
        detail = self.get_process_with_details(proc_id)
        if not detail:
            return None

        proc = detail.get("process") or {}
        tasks = detail.get("tasks") or []
        gateways = detail.get("gateways") or []
        events = detail.get("events") or []
        roles = detail.get("roles") or []
        agents = detail.get("agents") or []
        skills = detail.get("skills") or []

        def _node_id(kind: str, raw_id: str) -> str:
            return f"{kind}:{raw_id}"

        elements: list[dict] = []
        node_ids: set[str] = set()

        # Nodes
        p_id = str(proc.get("proc_id") or proc_id)
        p_node = _node_id("Process", p_id)
        node_ids.add(p_node)
        elements.append({"data": {"id": p_node, "type": "Process", "label": proc.get("name") or "Process", **proc}})

        def _add_nodes(kind: str, items: list, id_key: str, label_key: str = "name"):
            for it in items:
                if not isinstance(it, dict):
                    continue
                rid = str(it.get(id_key) or "").strip()
                if not rid:
                    continue
                nid = _node_id(kind, rid)
                if nid in node_ids:
                    continue
                node_ids.add(nid)
                label = str(it.get(label_key) or rid)
                elements.append({"data": {"id": nid, "type": kind, "label": label, **it}})

        _add_nodes("Task", tasks, "task_id")
        _add_nodes("Gateway", gateways, "gateway_id")
        _add_nodes("Event", events, "event_id")
        _add_nodes("Role", roles, "role_id")
        _add_nodes("Agent", agents, "agent_id")
        _add_nodes("Skill", skills, "skill_id")

        # Instruction nodes (Task와 분리된 별도 노드)
        task_instruction_map: dict[str, str] = {}
        for t in tasks:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("task_id") or "").strip()
            inst = str(t.get("instruction") or "").strip()
            if not tid or not inst:
                continue
            task_instruction_map[tid] = inst
            inst_id = f"{tid}:instruction"
            nid = _node_id("Instruction", inst_id)
            if nid in node_ids:
                continue
            node_ids.add(nid)
            first_line = inst.splitlines()[0] if inst.splitlines() else inst
            label = first_line[:120] + ("..." if len(first_line) > 120 else "")
            elements.append(
                {
                    "data": {
                        "id": nid,
                        "type": "Instruction",
                        "label": label or "Instruction",
                        "instruction_id": inst_id,
                        "task_id": tid,
                        "instruction": inst,
                    }
                }
            )

        # Edges: process containment
        def _edge(eid: str, source: str, target: str, rel_type: str, extra: dict = None):
            d = {"id": eid, "source": source, "target": target, "type": rel_type, "label": rel_type}
            if extra:
                d.update(extra)
            elements.append({"data": d})

        for t in tasks:
            tid = str((t or {}).get("task_id") or "").strip()
            if tid:
                _edge(f"HAS_TASK:{p_id}->{tid}", p_node, _node_id("Task", tid), "HAS_TASK")
        for g in gateways:
            gid = str((g or {}).get("gateway_id") or "").strip()
            if gid:
                _edge(f"HAS_GATEWAY:{p_id}->{gid}", p_node, _node_id("Gateway", gid), "HAS_GATEWAY")
        for e in events:
            eid0 = str((e or {}).get("event_id") or "").strip()
            if eid0:
                _edge(f"HAS_EVENT:{p_id}->{eid0}", p_node, _node_id("Event", eid0), "HAS_EVENT")

        # Edges: Task -> Role (actual relations)
        with self.session() as session:
            rel_rows = session.run(
                """
                MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(t:Task)
                OPTIONAL MATCH (t)-[:PERFORMED_BY]->(r:Role)
                RETURN t.task_id as task_id,
                       collect(DISTINCT r.role_id) as role_ids
                """,
                {"proc_id": proc_id},
            )
            for r in rel_rows:
                task_id = str(r.get("task_id") or "").strip()
                if not task_id:
                    continue
                for role_id in (r.get("role_ids") or []):
                    rid = str(role_id or "").strip()
                    if rid:
                        _edge(
                            f"PERFORMED_BY:{task_id}->{rid}",
                            _node_id("Task", task_id),
                            _node_id("Role", rid),
                            "PERFORMED_BY",
                        )

            # Edges: Task -> Instruction
            for task_id, inst in task_instruction_map.items():
                if not inst:
                    continue
                _edge(
                    f"HAS_INSTRUCTION:Task:{task_id}",
                    _node_id("Task", task_id),
                    _node_id("Instruction", f"{task_id}:instruction"),
                    "HAS_INSTRUCTION",
                )

            role_agent_rows = session.run(
                """
                MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(:Task)-[:PERFORMED_BY]->(r:Role)
                OPTIONAL MATCH (r)-[:ASSIGNED_AGENT]->(a:Agent)
                RETURN r.role_id as role_id, collect(DISTINCT a.agent_id) as agent_ids
                """,
                {"proc_id": proc_id},
            )
            for row in role_agent_rows:
                rid = str(row.get("role_id") or "").strip()
                if not rid:
                    continue
                for agent_id in (row.get("agent_ids") or []):
                    aid = str(agent_id or "").strip()
                    if aid:
                        _edge(
                            f"ASSIGNED_AGENT:{rid}->{aid}",
                            _node_id("Role", rid),
                            _node_id("Agent", aid),
                            "ASSIGNED_AGENT",
                        )

            agent_skill_rows = session.run(
                """
                MATCH (p:Process {proc_id: $proc_id})-[:HAS_TASK]->(:Task)-[:PERFORMED_BY]->(:Role)-[:ASSIGNED_AGENT]->(a:Agent)
                OPTIONAL MATCH (a)-[:USES_SKILL]->(s:Skill)
                RETURN a.agent_id as agent_id, collect(DISTINCT s.skill_id) as skill_ids
                """,
                {"proc_id": proc_id},
            )
            for row in agent_skill_rows:
                aid = str(row.get("agent_id") or "").strip()
                if not aid:
                    continue
                for skill_id in (row.get("skill_ids") or []):
                    sid = str(skill_id or "").strip()
                    if sid:
                        _edge(
                            f"USES_SKILL:Agent:{aid}->Skill:{sid}",
                            _node_id("Agent", aid),
                            _node_id("Skill", sid),
                            "USES_SKILL",
                        )

            process_skill_rows = session.run(
                """
                MATCH (p:Process {proc_id: $proc_id})-[:HAS_SKILL]->(s:Skill)
                RETURN collect(DISTINCT s.skill_id) as skill_ids
                """,
                {"proc_id": proc_id},
            )
            ps_record = process_skill_rows.single() if process_skill_rows else None
            if ps_record:
                for skill_id in (ps_record.get("skill_ids") or []):
                    sid = str(skill_id or "").strip()
                    if sid:
                        _edge(
                            f"HAS_SKILL:Process:{p_id}->Skill:{sid}",
                            _node_id("Process", p_id),
                            _node_id("Skill", sid),
                            "HAS_SKILL",
                        )

        # Edges: NEXT (sequence) within this process
        flows = self.get_sequence_flows(proc_id)
        for f in flows or []:
            from_type = str(f.get("from_type") or "")
            to_type = str(f.get("to_type") or "")
            from_id = str(f.get("from_id") or "").strip()
            to_id = str(f.get("to_id") or "").strip()
            if not from_id or not to_id:
                continue
            condition = str(f.get("condition") or "").strip()
            _edge(
                f"NEXT:{from_type}:{from_id}->{to_type}:{to_id}",
                _node_id(from_type, from_id),
                _node_id(to_type, to_id),
                "NEXT",
                {"condition": condition, "label": condition or ""},
            )

        # Basic counts
        node_count = len([x for x in elements if isinstance(x, dict) and x.get("data", {}).get("source") is None])
        edge_count = len(elements) - node_count
        return {"process_id": proc_id, "elements": elements, "counts": {"nodes": node_count, "edges": edge_count}}

    def get_full_graph_elements(self, max_nodes: int = 3000) -> dict:
        """
        Return process-core whole graph as Cytoscape-compatible elements.
        Includes all nodes/edges among PROCESS_CORE_LABELS.
        """
        target_labels = self.PROCESS_CORE_LABELS
        max_nodes = max(100, min(int(max_nodes or 3000), 10000))

        with self.session() as session:
            node_where_clause = self._label_predicate("n", target_labels)
            node_rows = session.run(
                f"""
                MATCH (n)
                WHERE {node_where_clause}
                RETURN labels(n) as labels, properties(n) as props
                LIMIT $max_nodes
                """,
                {"max_nodes": max_nodes},
            )

            nodes = []
            for row in node_rows:
                labels = row.get("labels") or []
                props = row.get("props") or {}
                if not labels:
                    continue
                nodes.append({"labels": labels, "props": props})

            if not nodes:
                return {"elements": [], "counts": {"nodes": 0, "edges": 0}}

            # Build node id mapping
            id_field_map = {
                "Process": "proc_id",
                "Task": "task_id",
                "Role": "role_id",
                "Agent": "agent_id",
                "Gateway": "gateway_id",
                "Event": "event_id",
                "Skill": "skill_id",
                "DMNDecision": "decision_id",
                "DMNRule": "rule_id",
            }

            elements: list[dict] = []
            known_node_ids: set[str] = set()

            def _node_id_for(labels: list[str], props: dict) -> tuple[str, str]:
                primary = ""
                for cand in target_labels:
                    if cand in labels:
                        primary = cand
                        break
                if not primary:
                    primary = labels[0] if labels else "Unknown"
                id_field = id_field_map.get(primary, "")
                business_id = str(props.get(id_field) or "").strip() if id_field else ""
                node_id = f"{primary}:{business_id or uuid4().hex}"
                return primary, node_id

            for n in nodes:
                labels = n["labels"]
                props = n["props"]
                primary, node_id = _node_id_for(labels, props)
                if node_id in known_node_ids:
                    continue
                known_node_ids.add(node_id)

                label = (
                    str(props.get("name") or "")
                    or str(props.get(id_field_map.get(primary, "")) or "")
                    or primary
                )
                node_data = {"id": node_id, "type": primary, "label": label, **props}
                elements.append({"data": node_data})

            rel_where_a = self._label_predicate("a", target_labels)
            rel_where_b = self._label_predicate("b", target_labels)
            rel_rows = session.run(
                f"""
                MATCH (a)-[r]->(b)
                WHERE ({rel_where_a})
                  AND ({rel_where_b})
                RETURN labels(a) as a_labels, properties(a) as a_props,
                       labels(b) as b_labels, properties(b) as b_props,
                       type(r) as rel_type, properties(r) as rel_props
                LIMIT $max_edges
                """,
                {"max_edges": max_nodes * 8},
            )

            edge_count = 0
            for row in rel_rows:
                a_labels = row.get("a_labels") or []
                a_props = row.get("a_props") or {}
                b_labels = row.get("b_labels") or []
                b_props = row.get("b_props") or {}
                rel_type = str(row.get("rel_type") or "")
                rel_props = row.get("rel_props") or {}
                _, src = _node_id_for(a_labels, a_props)
                _, dst = _node_id_for(b_labels, b_props)
                if not src or not dst:
                    continue
                edge_id = f"{rel_type}:{src}->{dst}"
                edge_data = {"id": edge_id, "source": src, "target": dst, "type": rel_type, "label": rel_type, **rel_props}
                elements.append({"data": edge_data})
                edge_count += 1

        return {"elements": elements, "counts": {"nodes": len(nodes), "edges": edge_count}}

    # ==================== Request Graph Snapshot ====================

    def save_request_graph_snapshots(
        self,
        run_id: str,
        integrated_graph: dict,
        process_graphs: dict[str, dict],
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Persist request-level integrated graph + per-process graph snapshots.
        Snapshots are stored separately from process-core labels so they survive core graph cleanup.
        """
        if not run_id:
            return {"saved": False, "reason": "empty_run_id"}

        now = datetime.utcnow().isoformat()
        meta = metadata or {}
        task_id = str(meta.get("task_id") or "").strip()
        integrated_payload = json.dumps(integrated_graph or {}, ensure_ascii=False)

        with self.session() as session:
            # Upsert run envelope
            session.run(
                """
                MERGE (r:GraphRun {run_id: $run_id})
                SET r.updated_at = $now,
                    r.created_at = coalesce(r.created_at, $now),
                    r.task_id = $task_id,
                    r.metadata = $metadata
                """,
                {
                    "run_id": run_id,
                    "now": now,
                    "task_id": task_id,
                    "metadata": json.dumps(meta, ensure_ascii=False),
                },
            )

            # Replace integrated snapshot (latest-only per run)
            session.run(
                """
                MATCH (r:GraphRun {run_id: $run_id})
                OPTIONAL MATCH (r)-[:HAS_SNAPSHOT]->(old:GraphSnapshot {snapshot_type: 'integrated'})
                DETACH DELETE old
                CREATE (s:GraphSnapshot {
                    snapshot_id: $snapshot_id,
                    run_id: $run_id,
                    snapshot_type: 'integrated',
                    proc_id: '',
                    payload_json: $payload_json,
                    created_at: $now
                })
                MERGE (r)-[:HAS_SNAPSHOT]->(s)
                """,
                {
                    "snapshot_id": str(uuid4()),
                    "run_id": run_id,
                    "payload_json": integrated_payload,
                    "now": now,
                },
            )

            # Replace process snapshots for this run
            for proc_id, graph in (process_graphs or {}).items():
                payload = json.dumps(graph or {}, ensure_ascii=False)
                session.run(
                    """
                    MATCH (r:GraphRun {run_id: $run_id})
                    OPTIONAL MATCH (r)-[:HAS_SNAPSHOT]->(old:GraphSnapshot {snapshot_type: 'process', proc_id: $proc_id})
                    DETACH DELETE old
                    CREATE (s:GraphSnapshot {
                        snapshot_id: $snapshot_id,
                        run_id: $run_id,
                        snapshot_type: 'process',
                        proc_id: $proc_id,
                        payload_json: $payload_json,
                        created_at: $now
                    })
                    MERGE (r)-[:HAS_SNAPSHOT]->(s)
                    """,
                    {
                        "snapshot_id": str(uuid4()),
                        "run_id": run_id,
                        "proc_id": str(proc_id or ""),
                        "payload_json": payload,
                        "now": now,
                    },
                )

        return {
            "saved": True,
            "run_id": run_id,
            "process_snapshot_count": len(process_graphs or {}),
        }

    def get_request_integrated_graph(self, run_id: str) -> Optional[dict]:
        """Fetch integrated graph snapshot for a run."""
        with self.session() as session:
            rec = session.run(
                """
                MATCH (:GraphRun {run_id: $run_id})-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'integrated'})
                RETURN s.payload_json as payload_json
                ORDER BY s.created_at DESC
                LIMIT 1
                """,
                {"run_id": run_id},
            ).single()
            if not rec:
                return None
            try:
                return json.loads(rec["payload_json"] or "{}")
            except Exception:
                return None

    def get_latest_request_integrated_graph(self) -> Optional[dict]:
        """Fetch latest integrated graph snapshot across all runs."""
        with self.session() as session:
            rec = session.run(
                """
                MATCH (r:GraphRun)-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'integrated'})
                RETURN r.run_id as run_id, r.task_id as task_id, s.payload_json as payload_json, s.created_at as created_at
                ORDER BY s.created_at DESC
                LIMIT 1
                """
            ).single()
            if not rec:
                return None
            try:
                payload = json.loads(rec["payload_json"] or "{}")
                if isinstance(payload, dict):
                    payload.setdefault("run_id", rec.get("run_id"))
                    payload.setdefault("task_id", rec.get("task_id"))
                return payload
            except Exception:
                return None

    def get_request_process_graph(self, run_id: str, proc_id: str) -> Optional[dict]:
        """Fetch process graph snapshot for a run/proc pair."""
        with self.session() as session:
            rec = session.run(
                """
                MATCH (:GraphRun {run_id: $run_id})-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'process', proc_id: $proc_id})
                RETURN s.payload_json as payload_json
                ORDER BY s.created_at DESC
                LIMIT 1
                """,
                {"run_id": run_id, "proc_id": proc_id},
            ).single()
            if not rec:
                return None
            try:
                return json.loads(rec["payload_json"] or "{}")
            except Exception:
                return None

    def get_latest_request_process_graph_by_proc_id(self, proc_id: str) -> Optional[dict]:
        """Fetch latest process graph snapshot by proc_id across all runs."""
        if not proc_id:
            return None
        with self.session() as session:
            rec = session.run(
                """
                MATCH (r:GraphRun)-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'process', proc_id: $proc_id})
                RETURN r.run_id as run_id, r.task_id as task_id, s.payload_json as payload_json, s.created_at as created_at
                ORDER BY s.created_at DESC
                LIMIT 1
                """,
                {"proc_id": proc_id},
            ).single()
            if not rec:
                return None
            try:
                payload = json.loads(rec["payload_json"] or "{}")
                if isinstance(payload, dict):
                    payload.setdefault("run_id", rec.get("run_id"))
                    payload.setdefault("task_id", rec.get("task_id"))
                    payload.setdefault("proc_id", proc_id)
                return payload
            except Exception:
                return None

    def get_latest_integrated_graph_by_proc_id(self, proc_id: str) -> Optional[dict]:
        """
        Fetch latest integrated graph snapshot by proc_id.
        This follows:
          process snapshot(proc_id) -> GraphRun -> integrated snapshot
        """
        if not proc_id:
            return None
        with self.session() as session:
            rec = session.run(
                """
                MATCH (r:GraphRun)-[:HAS_SNAPSHOT]->(ps:GraphSnapshot {snapshot_type: 'process', proc_id: $proc_id})
                MATCH (r)-[:HAS_SNAPSHOT]->(isn:GraphSnapshot {snapshot_type: 'integrated'})
                RETURN r.run_id as run_id, isn.payload_json as payload_json, isn.created_at as created_at
                ORDER BY isn.created_at DESC
                LIMIT 1
                """,
                {"proc_id": proc_id},
            ).single()
            if not rec:
                return None
            try:
                payload = json.loads(rec["payload_json"] or "{}")
                if isinstance(payload, dict):
                    payload.setdefault("run_id", rec.get("run_id"))
                    payload.setdefault("source_proc_id", proc_id)
                return payload
            except Exception:
                return None

    def get_latest_request_integrated_graph_by_task(self, task_id: str) -> Optional[dict]:
        """Fetch latest integrated graph snapshot by request task_id."""
        if not task_id:
            return None
        with self.session() as session:
            # 1) exact task_id property match
            rec = session.run(
                """
                MATCH (r:GraphRun {task_id: $task_id})-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'integrated'})
                RETURN r.run_id as run_id, r.task_id as matched_task_id, s.payload_json as payload_json, s.created_at as created_at
                ORDER BY s.created_at DESC
                LIMIT 1
                """,
                {"task_id": task_id},
            ).single()

            # 2) legacy fallback: run_id prefix (run_id is usually "{task_id}-{suffix}")
            if not rec:
                rec = session.run(
                    """
                    MATCH (r:GraphRun)-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'integrated'})
                    WHERE r.run_id STARTS WITH ($task_id + '-')
                    RETURN r.run_id as run_id, r.task_id as matched_task_id, s.payload_json as payload_json, s.created_at as created_at
                    ORDER BY s.created_at DESC
                    LIMIT 1
                    """,
                    {"task_id": task_id},
                ).single()

            # 3) metadata fallback for old rows where task_id wasn't promoted as property
            if not rec:
                rec = session.run(
                    """
                    MATCH (r:GraphRun)-[:HAS_SNAPSHOT]->(s:GraphSnapshot {snapshot_type: 'integrated'})
                    WHERE coalesce(r.metadata, '') CONTAINS $task_id
                    RETURN r.run_id as run_id, r.task_id as matched_task_id, s.payload_json as payload_json, s.created_at as created_at
                    ORDER BY s.created_at DESC
                    LIMIT 1
                    """,
                    {"task_id": task_id},
                ).single()
            if not rec:
                return None
            try:
                payload = json.loads(rec["payload_json"] or "{}")
                if isinstance(payload, dict):
                    payload.setdefault("run_id", rec.get("run_id"))
                    payload.setdefault("task_id", rec.get("matched_task_id") or task_id)
                return payload
            except Exception:
                return None
    
    def search_similar_by_name(
        self, 
        entity_type: str, 
        name: str, 
        limit: int = 5
    ) -> list[dict]:
        """Best-effort text search for AGE (fulltext procedure replacement)."""
        safe_label = re.sub(r"[^0-9A-Za-z_]", "", entity_type) or "Process"
        query = f"""
        MATCH (n:{safe_label})
        WHERE toLower(coalesce(n.name, '')) CONTAINS toLower($search_term)
           OR toLower(coalesce(n.description, '')) CONTAINS toLower($search_term)
           OR toLower(coalesce(n.purpose, '')) CONTAINS toLower($search_term)
        RETURN n {{.*, score: 1.0}} as entity
        LIMIT $limit
        """
        with self.session() as session:
            try:
                result = session.run(query, {
                    "search_term": name,
                    "limit": limit
                })
                return [record["entity"] for record in result]
            except Exception:
                return []

