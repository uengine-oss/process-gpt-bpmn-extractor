"""영어 초안·영어 문서로 만들면 정의의 이름·조건도 영어로 쓰라는 지시가 붙는지."""
from src.pdf2bpmn.processgpt.process_generation_messages import (
    build_process_definition_messages,
    source_language,
)


def test_source_language():
    outline = "## Process name\nSales Proposal\n## Steps\n1. [Customer] Submit request\n2. [Sales Representative] Write and send proposal"
    assert source_language("Create a sales proposal process.\n[ExecutionPolicy]\n- 한국어 지시", outline, {}) == "en"
    assert source_language("영업 제안서 프로세스를 만들어 줘", "## 프로세스명\n영업 제안서 작성", {}) == "ko"
    # 추출 결과의 키(tasks, name …)는 영어라도 값이 한국어면 한국어
    assert source_language("", "", {"extracted": {"tasks": [{"name": "견적 요청 접수", "description": "영업 담당자가 요청을 받는다"}]}}) == "ko"


def test_messages_carry_language_rule_only_for_english():
    kw = dict(base_system_prompt="", hints_simplified={}, extracted_summary={})
    en = build_process_definition_messages(consulting_outline="1. [Customer] Submit the request form today", user_request="Create a sales proposal process", **kw)
    ko = build_process_definition_messages(consulting_outline="1. [고객] 요청 입력", user_request="영업 제안서 프로세스를 만들어 줘", **kw)
    assert "Output language" in en[0]["content"]
    assert "Output language" not in ko[0]["content"]
