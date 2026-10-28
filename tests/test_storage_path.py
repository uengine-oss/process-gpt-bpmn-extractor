"""채팅 첨부 경로에서 서명 주소의 쿼리를 떼는지 — 메멘토 청크는 쿼리 없는 저장 경로로 찾는다."""
from pdf2bpmn_agent_executor import _storage_path


def test_signed_url_query_is_removed():
    assert _storage_path("files/7d6eea37.pdf?token=eyJhbGciOiJIUzI1NiJ9.e30.x") == "files/7d6eea37.pdf"


def test_plain_path_and_empty_values():
    assert _storage_path(" files/a.pdf? ") == "files/a.pdf"
    assert _storage_path("files/a.pdf") == "files/a.pdf"
    assert _storage_path(None) == ""
