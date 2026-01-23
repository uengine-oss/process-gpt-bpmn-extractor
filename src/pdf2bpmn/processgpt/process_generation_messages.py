"""
Process generation message builders (backend-only).

This module intentionally contains only "pure" helpers (no network, no DB) so that
it can be moved into a standalone backend service later with minimal friction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def process_quality_system_instructions() -> str:
    """
    Extra constraints to improve BPMN quality beyond the baseline ProcessGPT prompt.
    Keep this as a separate system message so it can be adjusted independently.
    """
    return (
        "### (추가) BPMN 품질/정합성 강화 규칙\n"
        "- Gateway를 생성하는 경우, 반드시 **최소 2개 이상의 outgoing Sequence**가 있어야 합니다.\n"
        "- ExclusiveGateway: outgoing 각각에 condition(한글)을 반드시 지정하세요.\n"
        "- ParallelGateway: condition은 비워두고, 분기/병합 구조가 자연스럽게 되도록 설계하세요.\n"
        "- Gateway가 있는데 분기가 1개 뿐이라면 Gateway를 만들지 말고 직선 흐름으로 연결하세요.\n"
        "- 모든 Activity/Event/Gateway는 startEvent에서 시작해 endEvent로 끝나도록 **끊김 없이** 연결되어야 합니다.\n"
        "- Sequence는 항상 source/target을 가져야 하며, 고아 노드(연결되지 않은 노드)를 만들지 마세요.\n"
    )


def build_process_definition_messages(
    *,
    base_system_prompt: str,
    hints_simplified: Dict[str, Any],
    consulting_outline: Optional[str],
    extracted_summary: Dict[str, Any],
    user_request: str,
) -> List[Dict[str, str]]:
    """
    Build the messages array passed to the LLM for ProcessGPT process definition generation.
    """
    extra_system = (
        "### (추가) 역할/에이전트 매핑 힌트\n"
        "- 아래 JSON은 Neo4j 추출정보 + 조직도/유저목록을 보고 'role/activity'별로 최소 매핑을 만든 결과입니다.\n"
        "- 프로세스 생성 시 role 이름을 일관되게 사용하고, role에 매핑된 담당자/에이전트가 있으면 최대한 반영하세요.\n"
        "- agentMode는 항상 \"draft\", orchestration은 항상 \"crewai-action\"으로 고정됩니다(후처리에서 강제됨).\n"
        "- 출력 포맷은 기존 프롬프트의 JSON 형식을 그대로 따르되, role/활동 이름/ID의 일관성을 최우선으로 하세요.\n"
        "- 가능한 경우, 역할 endpoint는 실제 agent(user_id)를 가리키도록 하세요(에이전트가 필요한 경우 자동 생성됨).\n"
        f"\n[hints_simplified]\n{json.dumps(hints_simplified, ensure_ascii=False)}\n"
    )

    user_prompt = (
        ((f"컨설팅 기반 프로세스 초안(참고):\n{consulting_outline}\n\n") if consulting_outline else "")
        + f"사용자 생성/정의 요청:\n{user_request}\n\n"
        + "PDF/Neo4j에서 추출된 프로세스 정보:\n"
        + f"{json.dumps(extracted_summary, ensure_ascii=False)}\n"
    )

    return [
        {"role": "system", "content": base_system_prompt},
        {"role": "system", "content": extra_system},
        {"role": "system", "content": process_quality_system_instructions()},
        {"role": "user", "content": user_prompt},
    ]

