"""
Process generation message builders (backend-only).

This module intentionally contains only "pure" helpers (no network, no DB) so that
it can be moved into a standalone backend service later with minimal friction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def strict_json_only_system_instructions() -> str:
    """
    Hard guardrail to force *strict* JSON-only output.
    This is intentionally placed as the LAST system message to override earlier formatting hints.
    """
    return (
        "### (최우선) 출력 형식 강제\n"
        "- 당신의 출력은 **반드시 단 하나의 JSON 객체**여야 합니다.\n"
        "- 마크다운/설명 문장/코드블록(```)/백틱/머리말/꼬리말/주석을 **절대 출력하지 마세요**.\n"
        "- JSON 외의 어떤 문자도 출력하면 실패입니다.\n"
        "- JSON 내부에도 주석(//, /* */)은 금지입니다.\n"
        "- 문자열에 따옴표(\")/개행이 포함되면 반드시 JSON 규칙대로 이스케이프 하세요.\n"
        "- 만약 요구사항을 충족할 수 없거나 불확실하다면, 다음 중 하나로만 응답하세요:\n"
        '  - {"error":"insufficient_information"}\n'
        '  - {"error":"cannot_comply"}\n'
    )


def strict_json_only_no_error_system_instructions() -> str:
    """
    JSON-only 강제(생성 전용).
    IMPORTANT:
    - process definition 생성은 "무조건 생성"이므로 error 폴백을 허용하면 모델이 쉽게 {"error":"cannot_comply"}로 도망갑니다.
    - 따라서 이 경로에서는 error 폴백을 금지합니다.
    """
    return (
        "### (최우선) 출력 형식 강제 (생성 전용)\n"
        "- 당신의 출력은 **반드시 단 하나의 JSON 객체**여야 합니다.\n"
        "- 마크다운/설명 문장/코드블록(```)/백틱/머리말/꼬리말/주석을 **절대 출력하지 마세요**.\n"
        "- JSON 외의 어떤 문자도 출력하면 실패입니다.\n"
        "- JSON 내부에도 주석(//, /* */)은 금지입니다.\n"
        "- 절대로 {\"error\":...} 형태로 응답하지 마세요.\n"
        "- 입력이 일부 불완전하더라도, 제공된 extracted/컨설팅 초안을 바탕으로 **최선의 프로세스 정의 JSON을 생성**하세요.\n"
    )


def create_only_process_definition_system_instructions() -> str:
    """
    생성 전용(Process Definition Create-Only) 규칙.
    - 질의(askProcessDef), 수정(modifications) 등 다른 모드 규칙을 배제하여 혼선을 줄입니다.
    """
    return (
        "### (목표) 프로세스 정의 생성 전용\n"
        "- 당신의 작업은 오직 1가지: 제공된 extracted(추출 정보)와 컨설팅 초안에 기반해 **프로세스 정의 JSON**을 생성하는 것입니다.\n"
        "- 질의(askProcessDef) 응답이나 수정(modifications) 형식은 절대 사용하지 마세요.\n"
        "- 절대로 {\"processDefinition\":{...}} 처럼 중첩된 래퍼로 감싸지 마세요. 반드시 최상위에 processDefinitionId/processDefinitionName/elements가 있어야 합니다.\n"
        "\n"
        "### (필수) 구성 요소\n"
        "- StartEvent 1개, EndEvent 1개는 반드시 포함\n"
        "- Activity(UserActivity) 는 반드시 1개 이상 포함 (extracted.tasks 기반으로 구성)\n"
        "- elements에는 Event/Activity/Gateway/Sequence만 사용\n"
        "- 모든 non-sequence 요소는 sequence로 연결되어야 함(끊김 금지)\n"
        "- sequenceFlows 같은 별도 배열로 분리하지 말고, 흐름은 반드시 elements의 Sequence로 표현하세요.\n"
        "\n"
        "### (출력 스키마 예시 - 이 형태로 100% 출력)\n"
        "{\n"
        "  \"processDefinitionId\": \"<uuid>\",\n"
        "  \"processDefinitionName\": \"<한글 프로세스명>\",\n"
        "  \"description\": \"<한글 설명>\",\n"
        "  \"isHorizontal\": true,\n"
        "  \"data\": [],\n"
        "  \"roles\": [{\"name\":\"100센터\",\"endpoint\":\"role_100_center\",\"resolutionRule\":\"\",\"origin\":\"created\"}],\n"
        "  \"elements\": [\n"
        "    {\"elementType\":\"Event\",\"id\":\"start_event\",\"name\":\"프로세스 시작\",\"role\":\"100센터\",\"type\":\"StartEvent\",\"description\":\"\",\"trigger\":\"\"},\n"
        "    {\"elementType\":\"Activity\",\"id\":\"task1\",\"name\":\"청약접수\",\"role\":\"100센터\",\"type\":\"UserActivity\",\"source\":\"start_event\",\"description\":\"\",\"instruction\":\"\",\"inputData\":[],\"outputData\":[\"청약접수 결과\"],\"checkpoints\":[],\"duration\":\"5\"},\n"
        "    {\"elementType\":\"Sequence\",\"id\":\"seq_start_task1\",\"name\":\"\",\"source\":\"start_event\",\"target\":\"task1\",\"condition\":\"\"},\n"
        "    {\"elementType\":\"Event\",\"id\":\"end_event\",\"name\":\"프로세스 종료\",\"role\":\"100센터\",\"type\":\"EndEvent\",\"description\":\"\",\"trigger\":\"\"}\n"
        "  ],\n"
        "  \"subProcesses\": [],\n"
        "  \"participants\": []\n"
        "}\n"
        "\n"
        "### (역할/태스크 사용 규칙)\n"
        "- extracted.roles가 있으면 roles는 최대한 extracted.roles를 반영\n"
        "- extracted.tasks가 있으면 Activity는 extracted.tasks의 이름/순서를 최대한 보존\n"
        "- role이 불명확하면 extracted.roles 중 가장 근접한 역할을 선택하거나, 최소 1개의 역할을 생성하여 모든 Activity/Event/Gateway에 role을 부여\n"
        "\n"
        "### (시퀀스/게이트웨이)\n"
        "- extracted.gateways / extracted.sequence_flows 정보가 있으면 반영\n"
        "- gateway를 만드는 경우 outgoing이 2개 이상이어야 하며, ExclusiveGateway면 각 outgoing condition을 한글로 채우세요\n"
    )


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
    # IMPORTANT:
    # - 이 백엔드는 "무조건 생성(create)"만 수행합니다.
    # - 따라서 askProcessDef/modifications 같은 다중 모드 규칙을 프롬프트에 포함하면 모델이 혼란을 느끼고
    #   {"error":"cannot_comply"} 같은 폴백을 선택할 수 있습니다.
    # - 최종 입력은 "컨설팅 초안 + extracted 전체"만 포함합니다. (user_request는 컨설팅 단계에서 이미 반영됨)

    user_prompt = (
        ((f"컨설팅 기반 프로세스 초안:\n{consulting_outline}\n\n") if consulting_outline else "")
        + "추출된 프로세스 정보(extracted 전체):\n"
        + f"{json.dumps(extracted_summary, ensure_ascii=False)}\n"
    )

    return [
        {"role": "system", "content": create_only_process_definition_system_instructions()},
        {"role": "system", "content": process_quality_system_instructions()},
        {"role": "system", "content": strict_json_only_no_error_system_instructions()},
        {"role": "user", "content": user_prompt},
    ]

