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
    # 영어면 스키마 자리표시자도 English, 언어 규칙은 system 맨 끝, 사용자 메시지에도 한 줄
    assert "<English 프로세스명>" in en[0]["content"] and "<한글 프로세스명>" not in en[0]["content"]
    assert en[0]["content"].find("### Output language") > en[0]["content"].find("### (최우선) 출력 형식 강제 (생성 전용)") > 0
    assert "in English" in en[-1]["content"]
    assert "<한글 프로세스명>" in ko[0]["content"] and "in English" not in ko[-1]["content"]


def test_worker_wrapper_is_not_counted():
    """MCP 가 만든 고정 한국어 틀([Description] …)은 언어 판단에서 빼고, 사용자 말만 본다."""
    import json as _json

    def wrap(data, extra=""):
        q = ("[Description]\n컨설팅 결과를 바탕으로 BPMN 프로세스를 생성합니다. (업로드 문서 없음 — 컨설팅 기반 생성)\n\n"
             "[Instruction]\n1. 컨설팅 내용을 분석하세요.\n2. 업무 프로세스 구조를 추출하세요.\n3. BPMN 형식의 프로세스 정의를 생성하세요.\n\n"
             f"[InputData]\n{_json.dumps(data, ensure_ascii=False)}")
        return q + (f"\n\n[추가 요청사항]\n{extra}" if extra else "")

    outline = "## Process name\nSales Proposal Creation\n\n## Steps\n1. [Customer] Submit request"
    en = wrap({"input_mode": "consulting", "user_request": "Create a sales proposal process.\n[ExecutionPolicy]\n- 한국어 지시",
               "user_answer": "Create the process as is", "tool_settings": {"pdf2bpmnLevel": "standard"}})
    assert source_language(en, outline, {"process_name": "Process 1", "extracted": {}}) == "en"
    ko = wrap({"input_mode": "consulting", "user_request": "영업 제안서 프로세스를 만들어 줘", "user_answer": "이대로 프로세스 생성"})
    assert source_language(ko, "## 프로세스명\n영업 제안서 작성\n1. [고객] 요청 입력", {}) == "ko"
    pdf = wrap({"files": [{"name": "Hanbit-Office-Quote-Approval-Policy.pdf"}]}, extra="Let agents handle the stock check.")
    extracted = {"extracted": {"tasks": [{"name": "Check stock", "description": "The warehouse checks the stock level"}]}}
    assert source_language(pdf, None, extracted) == "en"


def test_consulting_outline_blocks_are_not_counted():
    """실행기가 합친 컨설팅 입력 — 요청·답변마다 [ExecutionPolicy](한국어) 가 붙고 칸 이름도 한국어다."""
    policy = "\n[ExecutionPolicy]\n- ask_user는 마지막 수단입니다. 입력/문맥만으로 처리 가능한 경우 절대 호출하지 마세요.\n- 정말 필수 정보가 없어 진행 불가능한 경우에만 ask_user를 1회 호출하고, 호출 즉시 추가 도구 호출/추론을 중단하세요."
    combined = "\n\n".join([
        "[사용자 요청]\nCreate a sales proposal process. The customer enters their request." + policy,
        "[컨설팅 초안]\n## Process name\nSales Proposal Creation\n\n## Steps\n1. [Customer] Submit request",
        "[사용자 답변]\nCreate the process as is" + policy,
    ])
    assert source_language("Create a sales proposal process." + policy, combined, {"process_name": "Process 1", "extracted": {}}) == "en"
