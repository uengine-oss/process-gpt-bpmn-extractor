"""Post-processing helpers for task-level skills and lane agent decisions."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple


logger = logging.getLogger(__name__)


# 동일 클러스터로 묶을 지침 유사도 임계값.
# - 한국어 문장 변주(어미/조사/어순)에 강하도록 기본값을 보수적으로 낮게 둔다.
# - 환경변수 `SKILL_CLUSTER_SIMILARITY` 로 0.0~1.0 사이 값으로 조정 가능.
def _resolve_similarity_threshold() -> float:
    try:
        raw = os.getenv("SKILL_CLUSTER_SIMILARITY", "0.55")
        v = float(raw)
    except (TypeError, ValueError):
        return 0.55
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


SIMILARITY_THRESHOLD: float = _resolve_similarity_threshold()


@dataclass
class SkillCluster:
    canonical: str
    units: List[str]
    activity_ids: Set[str]


class ProcessPostProcessor:
    """Extract reusable skills from instructions and propose lane candidates."""

    HUMAN_REQUIRED_KWS = [
        "신청",
        "접수",
        "제출",
        "결재",
        "승인",
        "서명",
        "대면",
        "회의",
        "면담",
        "전화",
        "방문",
        "출석",
        "참석",
        "수령",
        "현장",
    ]
    AUTOMATION_HINT_KWS = [
        "자동",
        "에이전트",
        "봇",
        "생성",
        "요약",
        "정리",
        "분석",
        "검증",
        "추출",
        "조회",
        "검색",
        "분류",
        "추천",
        "집계",
    ]

    def __init__(
        self,
        *,
        min_ratio: float,
        min_count: int,
        lane_skill_min_tasks: int,
        require_automation: bool,
    ) -> None:
        self.min_ratio = max(0.0, float(min_ratio))
        self.min_count = max(1, int(min_count))
        self.lane_skill_min_tasks = max(1, int(lane_skill_min_tasks))
        self.require_automation = bool(require_automation)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[\"'`]+", " ", text)
        text = re.sub(r"[\(\)\[\]\{\}:,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {tok for tok in re.split(r"[^0-9a-zA-Z가-힣]+", text) if len(tok) >= 2}

    @staticmethod
    def _skill_key_from_text(text: str, idx: int) -> str:
        """Create stable-ish slug key for skill assignment."""
        tokens = [t for t in re.split(r"[^0-9a-zA-Z가-힣]+", text.lower()) if t]
        if tokens:
            key = "-".join(tokens[:6])
            key = re.sub(r"-{2,}", "-", key).strip("-")
        else:
            key = f"generated-skill-{idx}"
        if not key:
            key = f"generated-skill-{idx}"
        return key[:120]

    def _split_instruction_units(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"(?:\n+|[.;]|다\.)", text)
        out: List[str] = []
        seen: Set[str] = set()
        for p in parts:
            n = self._normalize_text(p)
            if len(n) < 8 or n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b or a in b or b in a:
            return 0.98
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        if inter == 0:
            return 0.0
        union = len(ta | tb)
        coverage = inter / max(1, min(len(ta), len(tb)))
        return max(inter / max(1, union), coverage)

    def build_skill_clusters(self, proc_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        지침을 단위로 분리한 뒤 토큰 자카드/coverage 기반으로 클러스터링하고
        threshold 를 통과한 "재사용 가능 클러스터"만 reusable 로 분리해서 반환한다.

        반환 구조:
            {
                "clusters": List[SkillCluster],   # 모든 클러스터 (디버그/로그용)
                "reusable": List[SkillCluster],   # threshold 통과 클러스터만
                "threshold": int,
                "stats": {...},
            }
        """
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list) or not activities:
            logger.info("[SKILL][CLUSTER] activities=0 → skip")
            return {"clusters": [], "reusable": [], "threshold": self.min_count, "stats": {}}

        threshold = max(self.min_count, int(math.ceil(len(activities) * self.min_ratio)))
        clusters: List[SkillCluster] = []

        empty_instruction_count = 0
        unit_total = 0
        for a in activities:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if not aid:
                continue
            source = str(a.get("instruction") or a.get("description") or "")
            units = self._split_instruction_units(source)
            if not units:
                empty_instruction_count += 1
                continue
            unit_total += len(units)
            for unit in units:
                matched = None
                for c in clusters:
                    if self._similarity(unit, c.canonical) >= SIMILARITY_THRESHOLD:
                        matched = c
                        break
                if matched is None:
                    matched = SkillCluster(canonical=unit, units=[unit], activity_ids={aid})
                    clusters.append(matched)
                else:
                    matched.units.append(unit)
                    matched.activity_ids.add(aid)

        reusable = [c for c in clusters if len(c.activity_ids) >= threshold]
        reusable.sort(key=lambda x: len(x.activity_ids), reverse=True)

        logger.info(
            "[SKILL][CLUSTER] activities=%d empty_instruction=%d units=%d clusters=%d "
            "threshold=%d sim=%.2f reusable=%d",
            len(activities), empty_instruction_count, unit_total,
            len(clusters), threshold, SIMILARITY_THRESHOLD, len(reusable),
        )
        for idx, c in enumerate(clusters, start=1):
            tag = "✓" if len(c.activity_ids) >= threshold else "·"
            logger.info(
                "[SKILL][CLUSTER]   %s #%d coverage=%d canonical=%r",
                tag, idx, len(c.activity_ids), c.canonical[:80],
            )

        return {
            "clusters": clusters,
            "reusable": reusable,
            "threshold": threshold,
            "stats": {
                "activities": len(activities),
                "empty_instruction": empty_instruction_count,
                "units": unit_total,
            },
        }

    def fallback_skill_card(self, cluster: "SkillCluster", idx: int) -> Dict[str, Any]:
        """
        LLM enrichment 실패/비활성 시 사용할 최소 형태의 스킬 카드.
        canonical 문장을 그대로 활용하지만 풍부한 형식 키들을 모두 채워 둔다.
        """
        canonical = (cluster.canonical or "").strip()
        title = canonical[:50] + ("..." if len(canonical) > 50 else "")
        slug = self._skill_key_from_text(title, idx)
        unique_units = []
        seen: Set[str] = set()
        for u in cluster.units:
            n = (u or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            unique_units.append(n)
        return {
            "id": slug,
            "safe_name": slug,
            "name": f"공통지침 {idx}: {title}",
            "description": canonical or f"공통지침 {idx}",
            "summary": canonical or "",
            "when_to_use": [],
            "inputs": [],
            "outputs": [],
            "procedure": [
                {"title": f"단계 {i + 1}", "detail": u}
                for i, u in enumerate(unique_units[:7])
            ],
            "examples": [],
            "notes": [],
            "source_activity_ids": sorted(cluster.activity_ids),
            "coverage_count": len(cluster.activity_ids),
            "canonical": canonical,
            # 호환용 (기존 소비자가 procedure_text 를 읽을 수 있어야 함)
            "procedure_text": canonical,
        }

    def apply_enriched_skills(
        self,
        proc_json: Dict[str, Any],
        enriched_skills: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        LLM 또는 폴백으로 만들어진 스킬 카드 리스트를 proc_json 에 반영한다.

        - proc_json["skills"] = enriched_skills (그대로 저장; safe_name 을 식별자로 사용)
        - 각 스킬의 source_activity_ids 를 보고 해당 activity.skills 에 safe_name 부착
        - 부착된 activity 는 agentMode/orchestration 을 deepagents 로 설정
        """
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list):
            activities = []

        by_activity: Dict[str, List[str]] = {}
        cleaned_skills: List[Dict[str, Any]] = []
        for s in enriched_skills:
            if not isinstance(s, dict):
                continue
            safe = str(s.get("safe_name") or s.get("id") or "").strip()
            if not safe:
                continue
            cleaned_skills.append(s)
            for aid in s.get("source_activity_ids") or []:
                aid_s = str(aid or "").strip()
                if not aid_s:
                    continue
                by_activity.setdefault(aid_s, []).append(safe)

        for a in activities:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if not aid:
                continue
            existing = a.get("skills")
            existing_list = [str(x).strip() for x in existing] if isinstance(existing, list) else []
            merged: List[str] = []
            seen: Set[str] = set()
            for sid in existing_list + by_activity.get(aid, []):
                if sid and sid not in seen:
                    seen.add(sid)
                    merged.append(sid)
            a["skills"] = merged
            if merged:
                a["agentMode"] = "complete"
                a["orchestration"] = "deepagents"

        proc_json["activities"] = activities
        proc_json["skills"] = cleaned_skills
        return {"skills": cleaned_skills}

    def enrich_with_task_skills(self, proc_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backward-compatible 진입점.
        LLM 호출 없이 클러스터링 → 폴백 카드 생성 → proc_json 부착까지 한 번에 수행한다.
        고품질 스킬 카드를 만들고 싶으면 build_skill_clusters() + (외부 LLM enrich) +
        apply_enriched_skills() 조합을 사용할 것.
        """
        result = self.build_skill_clusters(proc_json)
        reusable: List[SkillCluster] = result.get("reusable") or []
        threshold = result.get("threshold", self.min_count)
        skills: List[Dict[str, Any]] = [
            self.fallback_skill_card(c, idx) for idx, c in enumerate(reusable, start=1)
        ]
        self.apply_enriched_skills(proc_json, skills)
        return {"skills": skills, "threshold": threshold}

    def is_automation_eligible(self, activity: Dict[str, Any]) -> bool:
        text = self._normalize_text(
            f"{activity.get('name') or ''} {activity.get('instruction') or ''} {activity.get('description') or ''}"
        )
        if not text:
            return False
        for kw in self.AUTOMATION_HINT_KWS:
            if self._normalize_text(kw) in text:
                return True
        for kw in self.HUMAN_REQUIRED_KWS:
            if self._normalize_text(kw) in text:
                return False
        return True

    def collect_lane_skill_candidates(self, proc_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        activities = proc_json.get("activities") or []
        if not isinstance(activities, list):
            logger.info("[ASSIGN][LANE] activities=non-list → 0 candidates")
            return []

        lane_skill_to_activities: Dict[Tuple[str, str], List[str]] = {}
        activity_by_id: Dict[str, Dict[str, Any]] = {}
        missing_role_count = 0
        for a in activities:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            role = str(a.get("role") or "").strip()
            skill_ids = a.get("skills") or []
            if not aid or not isinstance(skill_ids, list):
                continue
            if not role:
                if skill_ids:
                    missing_role_count += 1
                continue
            activity_by_id[aid] = a
            for sid in [str(s).strip() for s in skill_ids if str(s).strip()]:
                lane_skill_to_activities.setdefault((role, sid), []).append(aid)

        out: List[Dict[str, Any]] = []
        rejected_below_min: List[Tuple[str, str, int]] = []
        rejected_automation: List[Tuple[str, str, int, int]] = []
        for (role, sid), aids in lane_skill_to_activities.items():
            unique_aids = sorted(set(aids))
            if len(unique_aids) < self.lane_skill_min_tasks:
                rejected_below_min.append((role, sid, len(unique_aids)))
                continue
            if self.require_automation:
                eligible = [
                    aid
                    for aid in unique_aids
                    if self.is_automation_eligible(activity_by_id.get(aid, {}))
                ]
                if len(eligible) < self.lane_skill_min_tasks:
                    rejected_automation.append((role, sid, len(unique_aids), len(eligible)))
                    continue
                unique_aids = eligible
            out.append({"role": role, "skill_id": sid, "activity_ids": unique_aids})

        out.sort(key=lambda x: (x["role"], -len(x["activity_ids"]), x["skill_id"]))

        logger.info(
            "[ASSIGN][LANE] pairs=%d candidates=%d rejected_below_min=%d "
            "rejected_automation=%d missing_role_with_skill=%d (min_tasks=%d, require_automation=%s)",
            len(lane_skill_to_activities), len(out), len(rejected_below_min),
            len(rejected_automation), missing_role_count,
            self.lane_skill_min_tasks, self.require_automation,
        )
        for role, sid, n in rejected_below_min:
            logger.info("[ASSIGN][LANE]   · skip(below_min) role=%r skill=%r count=%d", role, sid, n)
        for role, sid, n, eligible in rejected_automation:
            logger.info(
                "[ASSIGN][LANE]   · skip(automation) role=%r skill=%r count=%d eligible=%d",
                role, sid, n, eligible,
            )
        for c in out:
            logger.info(
                "[ASSIGN][LANE]   ✓ pick role=%r skill=%r tasks=%d",
                c["role"], c["skill_id"], len(c["activity_ids"]),
            )
        return out
