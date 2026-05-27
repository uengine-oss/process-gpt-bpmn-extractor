"""
Consulting -> extracted 변환 메시지 빌더 (백엔드 전용, 순수 함수).

용도:
- pdf2bpmn 은 원래 "문서(PDF) → 메멘토 청크 → 섹션 분리 → Neo4j 그래프 추출" 을 거쳐
  `extracted`(process/tasks/roles/gateways/events/sequence_flows) 구조를 만든 뒤
  그 구조로 ProcessGPT 프로세스 정의 JSON 을 생성한다.
- "컨설팅 기반 생성" 모드에서는 업로드 문서가 없으므로 위 앞단(메멘토/섹션/그래프)을 건너뛰고,
  사용자의 자연어 요청 + 컨설팅 초안 + 사용자 답변 + 이미지 분석 결과를
  동일한 `extracted` 구조로 변환한다.
- 즉 이 모듈은 "문서 추출" 을 "컨설팅 추출" 로 대체하기 위한 변환 단계만 담당하며,
  이후 JSON 생성 로직(`process_generation_messages.build_process_definition_messages`)은
  파일 모드와 100% 동일하게 재사용된다.

주의:
- 네트워크/DB 접근 없는 순수 헬퍼만 둔다 (process_generation_messages.py 와 동일 원칙).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def consulting_to_extracted_system_instructions() -> str:
    """컨설팅 내용을 extracted 구조로 변환하는 규칙."""
    return (
        "### (목표) 컨설팅 내용 → 추출 구조(extracted) 변환 전용\n"
        "- 당신의 작업은 1가지: 제공된 컨설팅 내용을 분석해 **하나의 업무 프로세스 추출 구조(JSON)**를 만드는 것입니다.\n"
        "- 이 결과는 곧바로 BPMN 프로세스 정의 생성의 입력(extracted)으로 사용됩니다.\n"
        "- 창작이 아니라, 컨설팅으로 합의된 흐름을 **구조화**하는 작업입니다. 컨설팅에 없는 단계/역할/분기를 임의로 추가하지 마세요.\n"
        "- 컨설팅 초안에 단계가 번호 목록으로 정리되어 있으면 그 순서를 그대로 tasks 의 순서로 사용하세요.\n"
        "- 사용자의 추가 답변/의견이 있으면 그 내용을 반영해 단계/역할/분기를 보정하세요.\n"
        "- 이미지 분석 내용이 있으면 그 안의 절차/흐름 정보도 함께 반영하세요.\n"
        "\n"
        "### (필수) 출력 스키마 — 이 형태로 100% 출력\n"
        "{\n"
        '  "process": {"name": "<한글 프로세스명>", "description": "<한글 한 줄 설명>"},\n'
        '  "tasks": [\n'
        '    {"task_id": "task_1", "name": "<한글 단계명>", "instruction": "<해당 단계 수행 지침(한글, 가능한 한 구체적으로)>",'
        ' "description": "<한글 보조 설명>", "role": "<수행 역할명>", "task_order": 1}\n'
        "  ],\n"
        '  "roles": [{"role_id": "role_1", "name": "<역할명>"}],\n'
        '  "gateways": [\n'
        '    {"gateway_id": "gw_1", "name": "<의사결정명>", "gateway_type": "ExclusiveGateway",'
        ' "condition": "", "description": "<무엇을 판단하는지>", "role": "<역할명>"}\n'
        "  ],\n"
        '  "events": [\n'
        '    {"event_id": "start_event", "event_type": "StartEvent", "name": "프로세스 시작"},\n'
        '    {"event_id": "end_event", "event_type": "EndEvent", "name": "프로세스 종료"}\n'
        "  ],\n"
        '  "sequence_flows": [\n'
        '    {"source": "start_event", "target": "task_1", "condition": ""},\n'
        '    {"source": "task_1", "target": "task_2", "condition": ""}\n'
        "  ]\n"
        "}\n"
        "\n"
        "### (tasks 규칙)\n"
        "- tasks 는 1개 이상이어야 하며, 실제 업무 흐름 순서대로 task_order 를 1,2,3,... 으로 부여하세요.\n"
        "- task_id 는 'task_1','task_2',... 처럼 순서대로 부여하세요.\n"
        "- name 은 짧은 한글 동작명(예: '청약 접수', '서류 검토')으로 작성하세요.\n"
        "- instruction 은 해당 단계에서 담당자가 무엇을 해야 하는지 컨설팅 내용 기반으로 구체적으로 작성하세요. 비우지 마세요.\n"
        "- 종결 의미가 강한 단계('통보','완료','마감','결과 발송')는 절대 첫 task 로 두지 말고 마지막 부근에 배치하세요.\n"
        "\n"
        "### (roles 규칙)\n"
        "- 컨설팅 내용에서 드러나는 역할/담당자를 roles 로 정리하세요. 최소 1개는 있어야 합니다.\n"
        "- 역할이 명시되지 않았으면 업무 맥락상 가장 자연스러운 역할명 1개를 만들어 모든 task 에 부여하세요.\n"
        "- 모든 task/gateway 의 role 값은 roles[].name 중 하나와 정확히 일치해야 합니다.\n"
        "\n"
        "### (gateways / sequence_flows 규칙)\n"
        "- 컨설팅 내용에 분기(승인/반려, 조건 분기 등)가 있으면 gateways 로 만들고, 없으면 gateways 는 빈 배열로 두세요.\n"
        "- gateway 를 만들면 그 gateway 에서 나가는 sequence_flows 가 2개 이상이어야 하며, 각 flow 의 condition 을 한글로 채우세요.\n"
        "- gateway name 은 '분기1' 같은 자리표시자 금지. 의사결정 의미를 담은 이름을 쓰세요.\n"
        "- sequence_flows 는 start_event → 첫 task → ... → 마지막 task → end_event 로 끊김 없이 단일 방향으로 연결하세요.\n"
        "- 분기가 없는 일반 흐름의 condition 은 빈 문자열(\"\")로 두세요.\n"
        "- source/target 에는 task_id, gateway_id, event_id 만 사용하세요.\n"
        "\n"
        "### (events 규칙)\n"
        "- events 에는 StartEvent 1개(event_id='start_event'), EndEvent 1개(event_id='end_event')를 반드시 포함하세요.\n"
        "\n"
        "### (안정성)\n"
        "- 출력 전 스스로 점검: tasks 순서가 업무 흐름과 맞는가? 모든 노드가 흐름으로 연결되는가? role 이 roles 와 일치하는가?\n"
    )


def consulting_to_extracted_output_format_instructions() -> str:
    """JSON-only 강제."""
    return (
        "### (최우선) 출력 형식 강제\n"
        "- 출력은 **반드시 단 하나의 JSON 객체**여야 합니다.\n"
        "- 마크다운/설명문/코드블록(```)/주석을 절대 출력하지 마세요. JSON 외 문자는 실패입니다.\n"
        "- 입력이 일부 불완전하더라도 컨설팅 내용 기반으로 **최선의 추출 구조 JSON 을 반드시 생성**하세요.\n"
        '- 절대로 {"error": ...} 형태로 응답하지 마세요.\n'
    )


def build_consulting_to_extracted_messages(
    *,
    user_request: str,
    consulting_outline: str,
    user_answer: str,
    image_analysis: str,
) -> List[Dict[str, str]]:
    """
    컨설팅 내용을 extracted 구조로 변환하기 위한 LLM 메시지 배열을 만든다.

    Args:
        user_request: 사용자의 원본 자연어 프로세스 생성 요청
        consulting_outline: 컨설팅 단계에서 합의된 프로세스 초안(마크다운)
        user_answer: 컨설팅 결과 패널에 대한 사용자의 선택/자유 의견
        image_analysis: 첨부 이미지 분석 결과 텍스트(없으면 빈 문자열)
    """
    payload = {
        "user_request": (user_request or "").strip(),
        "consulting_outline": (consulting_outline or "").strip(),
        "user_answer": (user_answer or "").strip(),
        "image_analysis": (image_analysis or "").strip(),
    }
    user_prompt = (
        "아래는 사용자와의 컨설팅 결과입니다. 이 내용을 토대로 추출 구조(extracted) JSON 을 생성하세요.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": consulting_to_extracted_system_instructions()},
        {"role": "system", "content": consulting_to_extracted_output_format_instructions()},
        {"role": "user", "content": user_prompt},
    ]
