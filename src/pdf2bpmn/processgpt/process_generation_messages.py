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
        "- 프로세스 정의는 창의적으로 새로 쓰는 것이 아니라, extracted에 있는 구조를 **안정적이고 일관된 BPMN 구조로 정리하는 것**이 목표입니다.\n"
        "- extracted에 명시된 이름/순서/역할/분기 의미를 최대한 유지하고, 사소한 표현 차이만으로 구조를 바꾸지 마세요.\n"
        "- 명시적 근거가 없는 새 프로세스/새 역할/새 게이트웨이/새 단계는 만들지 마세요.\n"
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
        "- Activity.instruction은 extracted.tasks.instruction 원문을 최우선으로 사용하세요.\n"
        "- instruction은 요약/재작성/축약하지 말고, 원문 문장과 줄바꿈을 최대한 유지하세요.\n"
        "- extracted에 instruction이 비어있을 때만 instruction을 빈 문자열로 두세요(임의 생성 금지).\n"
        "- role이 불명확하면 extracted.roles 중 가장 근접한 역할을 선택하거나, 최소 1개의 역할을 생성하여 모든 Activity/Event/Gateway에 role을 부여\n"
        "- 같은 business action으로 보이는 extracted.tasks는 이름을 임의로 바꾸지 말고 대표 이름을 안정적으로 유지하세요.\n"
        "- extracted.tasks의 순서를 특별한 근거 없이 재배열하지 마세요.\n"
        "- extracted에 이미 있는 역할명과 유사한 새 역할명을 임의로 만들지 마세요.\n"
        "\n"
        "### (시작 task 결정 - CRITICAL)\n"
        "- extracted.tasks 리스트의 첫 번째 항목(index=0)은 **반드시 BPMN의 첫 Activity** 입니다. \n"
        "  StartEvent의 outgoing Sequence는 그 첫 번째 task로 향해야 합니다.\n"
        "- 종결 의미가 강한 task('통보', '완료', '종료', '결과 발송', '마감')를 절대 BPMN의 시작점으로 만들지 마세요.\n"
        "  이런 task 는 거의 항상 EndEvent 직전에 위치합니다.\n"
        "- 만약 extracted.sequence_flows가 불완전해서 첫 task로부터 흐름이 보이지 않더라도,\n"
        "  extracted.tasks의 (index=0, 1, 2, ..., N-1) 순서를 기본 backbone 으로 사용하세요.\n"
        "  그 위에 extracted.sequence_flows / extracted.gateways 의 분기 정보를 덧붙이는 방식으로 BPMN 을 구성하세요.\n"
        "- extracted.tasks 의 마지막 항목은 보통 결과 통보/시스템 반영/이력 기록 등 종결 task 이므로\n"
        "  반드시 BPMN 의 마지막 Activity 위치(EndEvent 직전 또는 합류 후)로 배치하세요.\n"
        "\n"
        "### (시퀀스/게이트웨이)\n"
        "- extracted.gateways / extracted.sequence_flows 정보가 있으면 반영\n"
        "- gateway를 만드는 경우 outgoing이 2개 이상이어야 하며, ExclusiveGateway면 각 outgoing condition을 한글로 채우세요\n"
        "- gateway name은 절대로 '분기 1/분기 2' 같은 placeholder를 쓰지 마세요. gateway description/의사결정 의미를 반영한 이름으로 작성하세요.\n"
        "- ExclusiveGateway가 2개 분기라면 condition을 true/false 의미가 드러나게 작성하세요.\n"
        "  예: '<판단주제>인 경우' / '<판단주제>이 아닌 경우'\n"
        "- condition은 절대로 '조건 1/조건 2/분기 1' 같은 자리표시자(placeholder)로 쓰지 마세요.\n"
        "- extracted.sequence_flows.condition이 비어있는 경우에도, target activity name/description/instruction과 프로세스 문맥을 근거로 자연어 조건을 생성하세요.\n"
        "- condition으로 'HAS_TASK', 'HAS_INSTRUCTION', 'PERFORMED_BY' 같은 그래프 관계명은 절대 사용하지 마세요.\n"
        "- extracted에서 확인되는 instruction/description/role/task/flow는 누락하지 말고 가능한 범위에서 모두 반영하세요.\n"
        "- 분기 판단 내용은 dmn_rules가 아니라 반드시 Sequence.condition에 직접 작성하세요.\n"
        "- StartEvent에서 시작해 EndEvent에서 종료되도록 단절 없이 단일 방향 흐름을 보장하세요.\n"
        "- EndEvent를 제외한 모든 Activity/Gateway는 최소 1개의 outgoing Sequence를 가져야 합니다.\n"
        "- 마지막 업무 노드(terminal activity/gateway)는 반드시 EndEvent로 연결하세요.\n"
        "- 게이트웨이/조건/시퀀스는 extracted의 의미를 안정적으로 보존하는 방향으로만 보정하세요. 근거 없는 새 분기를 만들지 마세요.\n"
        "\n"
        "### (안정성 규칙)\n"
        "- 판단은 아래 규칙만 사용하세요: extracted의 이름, 순서, 역할, 흐름, 조건, 설명.\n"
        "- 단순한 문체 차이, 표현 차이, 길이 차이 때문에 구조를 바꾸지 마세요.\n"
        "- 애매한 경우에는 새 구조를 창작하기보다 extracted 구조를 최대한 유지하세요.\n"
        "- 출력 전 스스로 점검하세요:\n"
        "  - extracted.tasks의 순서가 유지되었는가?\n"
        "  - 같은 역할/같은 단계가 불필요하게 새 이름으로 바뀌지 않았는가?\n"
        "  - 게이트웨이와 조건이 extracted 의미와 일치하는가?\n"
        "  - 명시적 근거 없이 새 구조를 만들지 않았는가?\n"
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
        "- Gateway 이름은 description과 일치하는 의사결정명이어야 하며, '분기1/2' 같은 이름은 금지입니다.\n"
        "- condition은 '조건 1/조건 2' 같은 자리표시자 금지. 반드시 비즈니스 의미(승인/반려/요건충족/미충족 등)를 담으세요.\n"
        "- ParallelGateway: condition은 비워두고, 분기/병합 구조가 자연스럽게 되도록 설계하세요.\n"
        "- Gateway가 있는데 분기가 1개 뿐이라면 Gateway를 만들지 말고 직선 흐름으로 연결하세요.\n"
        "- 모든 Activity/Event/Gateway는 startEvent에서 시작해 endEvent로 끝나도록 **끊김 없이** 연결되어야 합니다.\n"
        "- Sequence는 항상 source/target을 가져야 하며, 고아 노드(연결되지 않은 노드)를 만들지 마세요.\n"
        "- EndEvent는 반드시 incoming Sequence를 1개 이상 가져야 하며, 흐름이 중간에서 끊기면 안 됩니다.\n"
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

