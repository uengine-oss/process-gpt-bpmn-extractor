"""LLM-based enrichment of clustered instructions into reusable skill cards.

`ProcessPostProcessor.build_skill_clusters` 가 만들어낸 "공통 지침 클러스터"를
받아서, 각 클러스터를 운영 가능한 SOP/스킬 카드 한 장으로 정제하기 위해 LLM 을
호출한다. 결과는 다음 키들을 가진 dict 형태이다:

    {
        "safe_name": "policy-criteria-check",   # 영문 kebab-case slug
        "name": "정책 기준 일치성 검증",            # 한국어 표시명
        "description": "...",                    # frontmatter 1~2 문장
        "summary": "...",                        # 3~5 문장 개요
        "when_to_use": [...],                    # 사용 시점/트리거
        "inputs": [...],                         # 사전 조건/입력
        "outputs": [...],                        # 산출물
        "procedure": [{"title", "detail"}, ...], # 단계별 절차
        "examples": [{"scenario","input","output"}, ...],
        "notes": [...],                          # 주의/제약
        "source_activity_ids": [...],            # 원본 activity id
        "coverage_count": 3,                     # 커버리지
        "canonical": "..."                       # 원본 캐노니컬 문장
    }

LLM 호출에 실패하거나 토글이 꺼진 경우, ProcessPostProcessor.fallback_skill_card
를 사용한 폴백 결과를 그대로 반환한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Config
from .process_post_processor import ProcessPostProcessor, SkillCluster


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema (LLM 출력 검증)
# ---------------------------------------------------------------------------


class SkillProcedureStep(BaseModel):
    title: str = Field(description="단계의 짧은 한국어 제목 (예: '정책 기준 정합성 확인')")
    detail: str = Field(description="이 단계에서 실제로 무엇을 어떻게 하는지 2~4문장 한국어 설명")


class SkillExample(BaseModel):
    scenario: str = Field(description="이 스킬이 사용되는 구체적 시나리오 한 줄 요약")
    input: str = Field(description="해당 시나리오에서의 입력/사전 조건 (한국어)")
    output: str = Field(description="기대되는 결과/산출물 (한국어)")


class EnrichedSkill(BaseModel):
    safe_name: str = Field(
        description=(
            "영문 kebab-case 식별자 (예: 'policy-criteria-check'). "
            "소문자 + 단어 구분은 '-'. 3~6 단어가 적절. 한글/공백/특수문자 금지."
        )
    )
    name: str = Field(
        description=(
            "사람이 읽는 한국어 스킬 이름 (예: '정책 기준 일치성 검증'). "
            "'공통지침', '기타', '스킬', '#N' 같은 일반/형식적 단어 금지."
        )
    )
    description: str = Field(
        description="frontmatter 용 1~2 문장 요약. 무엇을 하고 언제 쓰는지가 한 눈에 보이도록."
    )
    summary: str = Field(
        description="3~5 문장 개요. 이 스킬이 해결하는 문제와 산출물을 설명."
    )
    when_to_use: List[str] = Field(
        default_factory=list,
        description="이 스킬을 사용해야 하는 트리거/조건/질문 예시. 4~6개.",
    )
    inputs: List[str] = Field(
        default_factory=list,
        description="필요한 입력 데이터/사전 조건. 3~5개.",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="결과물/산출물. 2~4개.",
    )
    procedure: List[SkillProcedureStep] = Field(
        default_factory=list,
        description="단계별 절차. 4~7단계.",
    )
    examples: List[SkillExample] = Field(
        default_factory=list,
        description="실전 예시 1~2개.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="운영 시 주의/제약/정책. 3~5개.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SKILL_ENRICHMENT_PROMPT = """당신은 비즈니스 프로세스에서 반복되는 절차를 재사용 가능한 "스킬"로 정제하는 전문가입니다.
아래는 동일/유사한 의도를 가진 여러 활동(activity)에서 공통적으로 등장하는 지침 단위들입니다.
이 클러스터를 한 명의 운영자가 그대로 따라할 수 있는 SOP 형태의 스킬 카드 한 장으로 정제하세요.

[프로세스 이름]
{process_name}

[클러스터 캐노니컬 문장(원본 첫 문장)]
{canonical}

[클러스터에 속한 지침 단위들]
{units_block}

[클러스터에 속한 원본 활동들]
{activity_block}

요구사항:
- name 은 도메인 의미가 분명한 한국어 명사구. "공통지침", "기타", "스킬", "절차" 같이 일반·형식적 단어 사용 금지.
- safe_name 은 반드시 영문 소문자 + 하이픈(kebab-case). 3~6 단어. 한글·공백·특수문자 금지.
- description 은 frontmatter 용 1~2 문장 요약 (무엇/언제).
- summary 는 3~5 문장. 무엇을, 왜, 어떤 산출물로 만드는지.
- when_to_use 는 사용자 질문 형태나 트리거 조건 형태로 4~6개.
- inputs/outputs 는 데이터 단위(서류, 시스템 레코드, 결과 코드 등) 명사구로.
- procedure 는 4~7 단계. 각 단계는 (title, detail) 한국어. detail 은 2~4 문장 구체 설명.
- examples 는 1~2 개의 구체 시나리오. (scenario, input, output) 모두 한국어.
- notes 는 운영 시 주의/제약/정책 3~5개. 한국어.
- 출력은 반드시 JSON 한 객체로만 응답. 추가 텍스트, 마크다운 펜스(```), 설명 금지.

{format_instructions}
"""


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------


class SkillEnricher:
    """LLM 으로 클러스터 → 풍부한 스킬 카드를 생성한다."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self.model = (model or Config.SKILL_LLM_MODEL or Config.LLM_MODEL).strip()
        self.timeout = float(timeout if timeout is not None else Config.SKILL_LLM_TIMEOUT_SEC)
        self.max_concurrency = int(
            max_concurrency if max_concurrency is not None else Config.SKILL_LLM_MAX_CONCURRENCY
        )
        if self.max_concurrency < 1:
            self.max_concurrency = 1

        self._llm = ChatOpenAI(
            model=self.model,
            api_key=Config.OPENAI_API_KEY,
            base_url=(Config.LLM_BASE_URL or None),
            temperature=0,
            timeout=self.timeout,
            max_retries=1,
        )
        self._parser = JsonOutputParser(pydantic_object=EnrichedSkill)
        self._prompt = ChatPromptTemplate.from_template(
            SKILL_ENRICHMENT_PROMPT,
            partial_variables={"format_instructions": self._parser.get_format_instructions()},
        )
        self._chain = self._prompt | self._llm | self._parser

    @staticmethod
    def _format_units_block(cluster: SkillCluster) -> str:
        seen: set[str] = set()
        lines: List[str] = []
        for u in cluster.units or []:
            n = (u or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            lines.append(f"- {n}")
        return "\n".join(lines) if lines else "- (없음)"

    @staticmethod
    def _format_activity_block(
        cluster: SkillCluster, activity_by_id: Dict[str, Dict[str, Any]]
    ) -> str:
        lines: List[str] = []
        for aid in sorted(cluster.activity_ids):
            a = activity_by_id.get(aid) or {}
            name = str(a.get("name") or aid).strip()
            role = str(a.get("role") or "").strip()
            instr = " ".join(str(a.get("instruction") or "").split()).strip()
            desc = " ".join(str(a.get("description") or "").split()).strip()
            head = f"- [{aid}] {name}"
            if role:
                head += f" (역할: {role})"
            lines.append(head)
            if desc:
                lines.append(f"    설명: {desc}")
            if instr:
                lines.append(f"    지침: {instr}")
        return "\n".join(lines) if lines else "- (없음)"

    @staticmethod
    def _sanitize_safe_name(value: Any, fallback: str) -> str:
        s = str(value or "").strip().lower()
        s = re.sub(r"[^a-z0-9\-]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        if not s:
            s = (fallback or "skill").strip().lower()
            s = re.sub(r"[^a-z0-9\-]+", "-", s)
            s = re.sub(r"-{2,}", "-", s).strip("-")
        if not s:
            s = "skill"
        return s[:120]

    def _to_card(
        self,
        *,
        raw: Dict[str, Any],
        cluster: SkillCluster,
        idx: int,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM 응답 dict 를 표준 스킬 카드 dict 로 정규화."""
        if not isinstance(raw, dict):
            return fallback

        # safe_name 보정 (LLM 이 한글/공백을 흘리는 경우 대비)
        safe_name = self._sanitize_safe_name(
            raw.get("safe_name"),
            fallback=fallback.get("safe_name") or f"skill-{idx}",
        )

        name = str(raw.get("name") or "").strip() or fallback.get("name") or safe_name
        description = str(raw.get("description") or "").strip() or fallback.get("description") or ""
        summary = str(raw.get("summary") or "").strip() or fallback.get("summary") or ""

        def _str_list(v: Any) -> List[str]:
            if not isinstance(v, list):
                return []
            out: List[str] = []
            for x in v:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        out.append(s)
                elif isinstance(x, dict):
                    s = str(x.get("text") or x.get("value") or "").strip()
                    if s:
                        out.append(s)
            return out

        when_to_use = _str_list(raw.get("when_to_use"))
        inputs = _str_list(raw.get("inputs"))
        outputs = _str_list(raw.get("outputs"))
        notes = _str_list(raw.get("notes"))

        proc_raw = raw.get("procedure") or []
        procedure: List[Dict[str, str]] = []
        if isinstance(proc_raw, list):
            for i, step in enumerate(proc_raw, start=1):
                if isinstance(step, dict):
                    title = str(step.get("title") or f"단계 {i}").strip() or f"단계 {i}"
                    detail = str(step.get("detail") or "").strip()
                    if detail:
                        procedure.append({"title": title, "detail": detail})

        ex_raw = raw.get("examples") or []
        examples: List[Dict[str, str]] = []
        if isinstance(ex_raw, list):
            for ex in ex_raw:
                if isinstance(ex, dict):
                    scenario = str(ex.get("scenario") or "").strip()
                    if not scenario:
                        continue
                    examples.append(
                        {
                            "scenario": scenario,
                            "input": str(ex.get("input") or "").strip(),
                            "output": str(ex.get("output") or "").strip(),
                        }
                    )

        # procedure_text(호환): 단계 detail 들을 합쳐서 평문 절차서로 복원
        if procedure:
            procedure_text = "\n".join(
                f"{i + 1}. {p['title']} - {p['detail']}" for i, p in enumerate(procedure)
            )
        else:
            procedure_text = fallback.get("procedure_text") or summary or fallback.get("canonical") or ""

        return {
            "id": safe_name,
            "safe_name": safe_name,
            "name": name,
            "description": description,
            "summary": summary,
            "when_to_use": when_to_use,
            "inputs": inputs,
            "outputs": outputs,
            "procedure": procedure,
            "examples": examples,
            "notes": notes,
            "source_activity_ids": fallback.get("source_activity_ids") or sorted(cluster.activity_ids),
            "coverage_count": len(cluster.activity_ids),
            "canonical": fallback.get("canonical") or cluster.canonical,
            "procedure_text": procedure_text,
        }

    async def _enrich_one(
        self,
        *,
        cluster: SkillCluster,
        idx: int,
        process_name: str,
        activity_by_id: Dict[str, Dict[str, Any]],
        fallback: Dict[str, Any],
        sem: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        units_block = self._format_units_block(cluster)
        activity_block = self._format_activity_block(cluster, activity_by_id)
        canonical = (cluster.canonical or "").strip()

        async with sem:
            try:
                raw = await asyncio.wait_for(
                    self._chain.ainvoke(
                        {
                            "process_name": process_name or "(이름 없음)",
                            "canonical": canonical or "(없음)",
                            "units_block": units_block,
                            "activity_block": activity_block,
                        }
                    ),
                    timeout=self.timeout + 5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[SKILL][LLM] timeout (cluster #%d coverage=%d) → fallback",
                    idx, len(cluster.activity_ids),
                )
                return fallback
            except Exception as e:
                logger.warning(
                    "[SKILL][LLM] failed (cluster #%d coverage=%d): %s → fallback",
                    idx, len(cluster.activity_ids), e,
                )
                return fallback

        try:
            card = self._to_card(raw=raw, cluster=cluster, idx=idx, fallback=fallback)
            logger.info(
                "[SKILL][LLM] ✓ cluster #%d coverage=%d → name=%r safe_name=%r steps=%d examples=%d",
                idx, len(cluster.activity_ids),
                card.get("name"), card.get("safe_name"),
                len(card.get("procedure") or []),
                len(card.get("examples") or []),
            )
            return card
        except Exception as e:
            logger.warning(
                "[SKILL][LLM] normalize failed (cluster #%d): %s → fallback",
                idx, e,
            )
            return fallback

    async def enrich_clusters(
        self,
        *,
        clusters: List[SkillCluster],
        process_name: str,
        activity_by_id: Dict[str, Dict[str, Any]],
        post_processor: ProcessPostProcessor,
    ) -> List[Dict[str, Any]]:
        """
        주어진 reusable 클러스터 리스트를 LLM 으로 enrich 한다.
        실패하거나 LLM 토글이 꺼진 경우 fallback_skill_card 결과를 사용한다.
        반환 카드들의 safe_name 은 충돌 시 자동으로 -2, -3 ... 접미사를 붙인다.
        """
        if not clusters:
            return []

        # 폴백 카드를 먼저 만들어 둔다 (LLM 실패 시 그대로 사용)
        fallbacks: List[Dict[str, Any]] = [
            post_processor.fallback_skill_card(c, idx) for idx, c in enumerate(clusters, start=1)
        ]

        if not Config.SKILL_LLM_ENRICHMENT or not Config.OPENAI_API_KEY:
            logger.info(
                "[SKILL][LLM] disabled (toggle=%s, has_api_key=%s) → use fallback for %d clusters",
                Config.SKILL_LLM_ENRICHMENT, bool(Config.OPENAI_API_KEY), len(clusters),
            )
            cards = fallbacks
        else:
            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = [
                self._enrich_one(
                    cluster=c,
                    idx=idx,
                    process_name=process_name,
                    activity_by_id=activity_by_id,
                    fallback=fallbacks[idx - 1],
                    sem=sem,
                )
                for idx, c in enumerate(clusters, start=1)
            ]
            cards = await asyncio.gather(*tasks)

        # safe_name 충돌 보정
        used: Dict[str, int] = {}
        for c in cards:
            base = str(c.get("safe_name") or "skill").strip() or "skill"
            n = used.get(base, 0)
            if n == 0:
                used[base] = 1
                continue
            n += 1
            used[base] = n
            new_name = f"{base}-{n}"
            c["safe_name"] = new_name
            c["id"] = new_name

        return list(cards)


def build_activity_index(proc_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """proc_json["activities"] 를 id 기반 dict 로 인덱싱."""
    out: Dict[str, Dict[str, Any]] = {}
    for a in proc_json.get("activities") or []:
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id") or "").strip()
        if aid:
            out[aid] = a
    return out


def render_skill_markdown(card: Dict[str, Any]) -> str:
    """
    표준화된 스킬 카드 dict 를 SKILL.md 형식으로 직렬화.
    참조: c:/Users/user/Documents/.../global-investment-analyzer/SKILL.md
    """
    name = str(card.get("name") or "Untitled Skill").strip()
    description = str(card.get("description") or "").strip()
    summary = str(card.get("summary") or "").strip()
    when_to_use = card.get("when_to_use") or []
    inputs = card.get("inputs") or []
    outputs = card.get("outputs") or []
    procedure = card.get("procedure") or []
    examples = card.get("examples") or []
    notes = card.get("notes") or []
    source_ids = card.get("source_activity_ids") or []
    coverage = card.get("coverage_count")
    canonical = str(card.get("canonical") or "").strip()

    def _fence_yaml(v: str) -> str:
        # YAML 안전: 콜론/줄바꿈 등이 있으면 따옴표로 감싼다.
        v = v.replace("\n", " ").strip()
        if not v:
            return '""'
        if any(ch in v for ch in [':', '#', '"', "'"]):
            escaped = v.replace('"', '\\"')
            return f"\"{escaped}\""
        return v

    lines: List[str] = []
    lines.append("---")
    lines.append(f"name: {_fence_yaml(name)}")
    if description:
        lines.append(f"description: {_fence_yaml(description)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")

    if summary:
        lines.append("## 개요")
        lines.append("")
        lines.append(summary)
        lines.append("")

    if when_to_use:
        lines.append("## 사용 시점")
        lines.append("")
        for item in when_to_use:
            s = str(item).strip()
            if s:
                lines.append(f"- {s}")
        lines.append("")

    if inputs:
        lines.append("## 입력 / 사전 조건")
        lines.append("")
        for item in inputs:
            s = str(item).strip()
            if s:
                lines.append(f"- {s}")
        lines.append("")

    if outputs:
        lines.append("## 산출물")
        lines.append("")
        for item in outputs:
            s = str(item).strip()
            if s:
                lines.append(f"- {s}")
        lines.append("")

    if procedure:
        lines.append("## 절차")
        lines.append("")
        for i, step in enumerate(procedure, start=1):
            title = str(step.get("title") or f"단계 {i}").strip() or f"단계 {i}"
            detail = str(step.get("detail") or "").strip()
            lines.append(f"### {i}. {title}")
            lines.append("")
            if detail:
                lines.append(detail)
                lines.append("")

    if examples:
        lines.append("## 실전 예시")
        lines.append("")
        for i, ex in enumerate(examples, start=1):
            scenario = str(ex.get("scenario") or "").strip()
            inp = str(ex.get("input") or "").strip()
            out = str(ex.get("output") or "").strip()
            lines.append(f"### 예시 {i}: {scenario or '시나리오'}")
            lines.append("")
            if inp:
                lines.append(f"- 입력: {inp}")
            if out:
                lines.append(f"- 산출: {out}")
            lines.append("")

    if notes:
        lines.append("## 주의사항")
        lines.append("")
        for item in notes:
            s = str(item).strip()
            if s:
                lines.append(f"- {s}")
        lines.append("")

    # 추적용 메타 (운영자가 볼 수 있게 하단에 보존)
    if source_ids or coverage is not None or canonical:
        lines.append("## 출처 (Source Activities)")
        lines.append("")
        if coverage is not None:
            lines.append(f"- coverage: {coverage}")
        if source_ids:
            lines.append("- activities: " + ", ".join(str(x) for x in source_ids))
        if canonical:
            lines.append(f"- canonical: {canonical}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "EnrichedSkill",
    "SkillEnricher",
    "SkillProcedureStep",
    "SkillExample",
    "build_activity_index",
    "render_skill_markdown",
]
