"""
HITL (Human In The Loop) 헬퍼 — pdf2bpmn 워커가 처리 중에 사용자에게 질문/승인을 요청하기 위한 모듈.

흐름 (pause / resume):
  1. emit_waiting_for_user(event_queue, ...) — 프론트에 질문 UI 표시
  2. pause_for_hitl(supabase, todo_id, checkpoint, question_ids)
     → output.hitl_checkpoint 저장, draft_status=HUMAN_ASKED, consumer 해제
     → execute() 종료 (파드는 todo 를 붙잡지 않음)
  3. 사용자 응답 → todolist.output.hitl_feedbacks + draft_status=FB_REQUESTED (프론트)
  4. fetch_pending_task 가 todo 재 claim → execute() 재진입
  5. read_batch_responses(...) 로 응답 읽고 checkpoint 기준으로 이어서 실행

(레거시) wait_for_batch_responses 는 인프로세스 폴링용 — 신규 경로에서는 사용하지 않음.

응답 dict 형태:
  {
    "question_id": str,
    "action":      "select_items" | "approve" | "reject" | ...
    "selected_ids":   [str, ...]    # select_items 시
    "selected_items": [{id,label,description}, ...]
    "custom_text":    str           # allow_other 자유 입력
    "answer":         str           # 표시용
    "reason":         str
    "submitted_at":   ISO timestamp
  }

UI 컨트랙트는 process-gpt-vue3/docs/HUMAN_FEEDBACK_PANEL.md 참고.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HitlPauseException(Exception):
    """HITL 대기로 execute() 를 정상 종료할 때 사용."""


# =========================================================================
# 1) waiting_for_user 이벤트 발송
# =========================================================================

def make_question_id(prefix: str = "q") -> str:
    """고유 question_id 생성. 같은 task 안에서 dedupe 용."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_question_payload(
    *,
    question: str,
    feedback_type: str = "select_items",
    items: Optional[List[Dict[str, Any]]] = None,
    suggestions: Optional[List[str]] = None,
    context: Optional[str] = None,
    evidence_spans: Optional[List[str]] = None,
    impact_preview: Optional[List[str]] = None,
    allow_multiple: bool = False,
    min_select: int = 1,
    allow_other: bool = True,
    allow_skip: bool = False,
    target_type: str = "",
    target_id: str = "",
    question_id: Optional[str] = None,
    option_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """프론트의 HumanFeedbackPanel 이 그대로 렌더할 수 있는 question payload 를 만든다.

    이 payload 는 events 테이블의 data.questions[0] 으로 들어간다.
    """
    qid = question_id or make_question_id()
    q: Dict[str, Any] = {
        "question_id": qid,
        "prompt": question,
        "feedback_type": feedback_type,
        "allow_multiple": bool(allow_multiple),
        "min_select": int(min_select),
        "allow_other": bool(allow_other),
        "allow_skip": bool(allow_skip),
        "target_type": target_type or "",
        "target_id": target_id or "",
    }
    if context:
        q["context"] = context
    if items:
        q["items"] = items
    if suggestions:
        q["choices"] = suggestions  # 프론트 addPdf2BpmnHumanQuestionMessage 가 choices 키를 본다
        q["suggestions"] = suggestions
    if evidence_spans:
        q["evidence_spans"] = evidence_spans
    if impact_preview:
        q["impact_preview"] = impact_preview
    if option_meta:
        q["option_meta"] = option_meta
    return q


def emit_waiting_for_user(
    *,
    event_queue,
    context_id: str,
    task_id: str,
    job_id: str,
    main_loop: asyncio.AbstractEventLoop,
    question_payload: Optional[Dict[str, Any]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    progress: int = 72,
    message_text: Optional[str] = None,
) -> None:
    """waiting_for_user 이벤트를 큐에 넣어 SDK 가 events 테이블에 INSERT 하도록 한다.

    - 단일 질문: question_payload 만 전달 (하위 호환)
    - 다중 질문: questions 배열 전달 → 프론트가 한 패널에 여러 섹션으로 렌더,
      한 번의 "응답 제출" 로 모든 질문 답변 batch 전송 (사용자 개입 1회)
    """
    # 지연 import: SDK 가 없는 환경(테스트 등) 에서도 모듈 로드는 가능하도록
    try:
        from a2a.types import TaskStatusUpdateEvent, TaskState  # type: ignore
        from a2a.utils import new_agent_text_message  # type: ignore
    except Exception as exc:
        logger.warning(f"[HITL] a2a SDK 임포트 실패 — emit 스킵: {exc}")
        return

    # questions 인자가 우선. 없으면 question_payload 단독을 배열로 감싼다.
    qlist: List[Dict[str, Any]]
    if isinstance(questions, list) and questions:
        qlist = list(questions)
    elif isinstance(question_payload, dict):
        qlist = [question_payload]
    else:
        logger.warning("[HITL] emit_waiting_for_user: questions 또는 question_payload 가 비어있음 — 스킵")
        return

    primary_prompt = (qlist[0].get("prompt") or "") if isinstance(qlist[0], dict) else ""
    msg_text = message_text or (
        f"[HITL] 사용자 확인 대기 중: {primary_prompt}"
        if len(qlist) == 1
        else f"[HITL] 사용자 확인 대기 중: {len(qlist)}개의 질문에 한 번에 응답해 주세요"
    )
    event_data = {
        "message": msg_text,
        "status": "waiting_for_user",
        "progress": int(progress),
        "job_id": job_id,
        "questions": qlist,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    evt = TaskStatusUpdateEvent(
        status={
            "state": TaskState.working,
            "message": new_agent_text_message(
                json.dumps(event_data, ensure_ascii=False),
                context_id, task_id,
            ),
        },
        final=False,
        contextId=context_id,
        taskId=task_id,
        metadata={
            "crew_type": "pdf2bpmn",
            "event_type": "waiting_for_user",
            "job_id": job_id,
            "progress": int(progress),
        },
    )
    try:
        main_loop.call_soon_threadsafe(event_queue.enqueue_event, evt)
    except Exception:
        # 마지막 수단 — 이벤트 루프가 없으면 직접 enqueue (안전한 fallback)
        try:
            event_queue.enqueue_event(evt)
        except Exception as exc:
            logger.error(f"[HITL] waiting_for_user 이벤트 발송 실패: {exc}")


def emit_human_feedback_received(
    *,
    event_queue,
    context_id: str,
    task_id: str,
    job_id: str,
    main_loop: asyncio.AbstractEventLoop,
    question_id: str,
    summary: str,
    progress: int = 74,
) -> None:
    """사용자 응답을 수신했음을 프론트에 알린다 (작업 재개 직전)."""
    try:
        from a2a.types import TaskStatusUpdateEvent, TaskState  # type: ignore
        from a2a.utils import new_agent_text_message  # type: ignore
    except Exception:
        return

    event_data = {
        "message": summary,
        "status": "human_feedback_submitted",
        "progress": int(progress),
        "job_id": job_id,
        "question_id": question_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    evt = TaskStatusUpdateEvent(
        status={
            "state": TaskState.working,
            "message": new_agent_text_message(
                json.dumps(event_data, ensure_ascii=False),
                context_id, task_id,
            ),
        },
        final=False,
        contextId=context_id,
        taskId=task_id,
        metadata={
            "crew_type": "pdf2bpmn",
            "event_type": "human_feedback_submitted",
            "job_id": job_id,
            "progress": int(progress),
        },
    )
    try:
        main_loop.call_soon_threadsafe(event_queue.enqueue_event, evt)
    except Exception:
        try:
            event_queue.enqueue_event(evt)
        except Exception:
            pass


# =========================================================================
# 2) todolist.output.hitl_feedbacks 폴링
# =========================================================================

def _read_todolist_output(supabase_client, todo_id: str) -> Dict[str, Any]:
    """todolist.output JSON dict 를 읽어 반환. 실패 시 빈 dict."""
    try:
        res = (
            supabase_client.table("todolist")
            .select("output")
            .eq("id", todo_id)
            .limit(1)
            .execute()
        )
        rows = getattr(res, "data", None) or []
        if not rows:
            return {}
        out = rows[0].get("output")
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                out = {}
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _to_jsonable(obj: Any) -> Any:
    """dataclass / pydantic / set / 기타 비-JSON 객체를 supabase 가 받을 수 있는 형태로 변환.

    이 함수는 hitl_checkpoint 의 workflow_state.dmn_decisions(=DMNDecision dataclass) 같은
    객체를 dict 으로 풀어줘서 update 가 'Object of type X is not JSON serializable' 로
    통째로 실패하는 것을 방지한다. update 가 실패하면 hitl_checkpoint 가 저장되지 않아
    재개 시 워커가 처음부터(메멘토/섹션분리/PASS1) 다시 실행되는 치명적인 회귀가 발생한다.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in obj]
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _to_jsonable(dataclasses.asdict(obj))
    except Exception:
        pass
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _to_jsonable(fn())
            except Exception:
                pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def _write_todolist_output(supabase_client, todo_id: str, output: Dict[str, Any]) -> bool:
    try:
        safe_output = _to_jsonable(output)
    except Exception as exc:
        logger.error(f"[HITL] todolist.output 직렬화 실패: {exc}")
        safe_output = output
    try:
        supabase_client.table("todolist").update({"output": safe_output}).eq("id", todo_id).execute()
        return True
    except Exception as exc:
        logger.error(f"[HITL] todolist.output 업데이트 실패: {exc}")
        return False


def prepare_hitl_wait(
    supabase_client,
    todo_id: str,
    question_ids: List[str],
) -> str:
    """HITL 대기 시작 전 todolist.output 을 정리하고 wait 마커를 기록한다.

    - 동일 question_id 의 이전 응답(재시도/새로고침 잔여)을 제거해
      wait_for_batch_responses 가 즉시 완료되는 것을 방지한다.
    - hitl_wait_started_at 을 기록해 이후 들어오는 응답만 유효하게 한다.
    """
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if supabase_client is None or not todo_id:
        return started_at

    qid_set = {str(q) for q in question_ids if str(q or "").strip()}
    out = _read_todolist_output(supabase_client, todo_id)
    if qid_set:
        existing = read_batch_responses(
            supabase_client, todo_id, list(qid_set), None
        )
        if all(existing.get(q) for q in qid_set):
            prev_started = str(out.get("hitl_wait_started_at") or "").strip()
            if prev_started:
                started_at = prev_started
            out["hitl_pending_question_ids"] = list(qid_set)
            _write_todolist_output(supabase_client, todo_id, out)
            logger.info(
                "[HITL] prepare_hitl_wait — 기존 응답 유지 (재질문/재pause 스킵) todo_id=%s",
                todo_id,
            )
            return started_at
    feedbacks = out.get("hitl_feedbacks") or []
    if not isinstance(feedbacks, list):
        feedbacks = []
    if qid_set:
        feedbacks = [
            fb for fb in feedbacks
            if isinstance(fb, dict) and str(fb.get("question_id") or "") not in qid_set
        ]
    out["hitl_feedbacks"] = feedbacks
    out["hitl_wait_started_at"] = started_at
    out["hitl_pending_question_ids"] = list(qid_set)
    _write_todolist_output(supabase_client, todo_id, out)
    return started_at


def read_batch_responses(
    supabase_client,
    todo_id: str,
    question_ids: List[str],
    wait_started_at: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """todolist.output.hitl_feedbacks 에서 question_id 별 최신 응답을 한 번에 읽는다 (폴링 없음)."""
    result: Dict[str, Optional[Dict[str, Any]]] = {
        qid: None for qid in question_ids if qid
    }
    if not result or supabase_client is None or not todo_id:
        return result

    wait_ts = None
    if wait_started_at:
        try:
            wait_ts = datetime.fromisoformat(
                str(wait_started_at).replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            wait_ts = None

    def _entry_is_valid(fb: Dict[str, Any]) -> bool:
        if not wait_ts:
            return True
        submitted = str(fb.get("submitted_at") or "")
        if not submitted:
            return False
        try:
            return (
                datetime.fromisoformat(submitted.replace("Z", "+00:00")).timestamp()
                >= wait_ts
            )
        except Exception:
            return False

    try:
        entries = _read_all_feedback_entries(supabase_client, todo_id)
        for fb in entries:
            if not isinstance(fb, dict):
                continue
            qid = str(fb.get("question_id") or "")
            if qid in result and result[qid] is None and _entry_is_valid(fb):
                result[qid] = fb
    except Exception as exc:
        logger.warning(f"[HITL] read_batch_responses 실패: {exc}")
    return result


def pause_for_hitl(
    supabase_client,
    todo_id: str,
    checkpoint: Dict[str, Any],
    question_ids: List[str],
) -> str:
    """HITL 대기: checkpoint 저장 후 todo 를 HUMAN_ASKED 로 두고 consumer 를 해제한다."""
    started_at = prepare_hitl_wait(supabase_client, todo_id, question_ids)
    if supabase_client is None or not todo_id:
        return started_at

    cp = dict(checkpoint or {})
    cp["wait_started_at"] = started_at
    out = _read_todolist_output(supabase_client, todo_id)
    out["hitl_checkpoint"] = cp
    out["hitl_paused"] = True
    out["pdf2bpmn_phase"] = "awaiting_hitl"
    _write_todolist_output(supabase_client, todo_id, out)

    try:
        supabase_client.table("todolist").update({
            "draft_status": "HUMAN_ASKED",
            "consumer": None,
        }).eq("id", todo_id).execute()
        logger.info(
            "[HITL] pause_for_hitl — todo_id=%s draft_status=HUMAN_ASKED consumer cleared",
            todo_id,
        )
    except Exception as exc:
        logger.warning(f"[HITL] pause_for_hitl draft_status 업데이트 실패: {exc}")
    return started_at


def stable_hitl_question_id(task_id: str, key: str) -> str:
    """재폴링/재실행 시에도 동일한 question_id 를 유지 (응답 매칭용)."""
    tid = str(task_id or "").strip()
    k = str(key or "").strip()
    return f"{tid}-{k}" if tid and k else make_question_id(k or "q")


def mark_hitl_process_resolved(
    supabase_client,
    todo_id: str,
    process_index: int,
) -> None:
    """해당 process_index 의 HITL 이 완료됐음을 기록 — 재실행 시 동일 질문 반복 방지."""
    if supabase_client is None or not todo_id:
        return
    out = _read_todolist_output(supabase_client, todo_id)
    idxs = out.get("hitl_resolved_process_indexes")
    if not isinstance(idxs, list):
        idxs = []
    if int(process_index) not in [int(x) for x in idxs if str(x).isdigit() or isinstance(x, int)]:
        idxs.append(int(process_index))
    out["hitl_resolved_process_indexes"] = idxs
    out.pop("hitl_paused", None)
    out["pdf2bpmn_phase"] = "post_hitl_generate"
    _write_todolist_output(supabase_client, todo_id, out)


def clear_hitl_checkpoint(supabase_client, todo_id: str) -> None:
    """재개 직후 checkpoint 마커 제거 (응답 기록은 유지)."""
    if supabase_client is None or not todo_id:
        return
    out = _read_todolist_output(supabase_client, todo_id)
    out.pop("hitl_checkpoint", None)
    out.pop("hitl_pending_question_ids", None)
    _write_todolist_output(supabase_client, todo_id, out)


async def wait_for_batch_responses(
    *,
    supabase_client,
    todo_id: str,
    question_ids: List[str],
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 2.0,
    wait_started_at: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """주어진 question_ids 모두에 응답이 들어올 때까지 폴링.

    프론트는 각 스텝마다 todolist.output.hitl_feedbacks 에 question_id 별로
    응답을 추가할 수 있고, 마지막에 통합 제출로 한꺼번에 보낼 수도 있다.

    반환:
        {question_id: entry|None, ...}
        timeout 시 일부만 채워지거나 모두 None 일 수 있음.
    """
    result: Dict[str, Optional[Dict[str, Any]]] = {qid: None for qid in question_ids if qid}
    if not result:
        return result
    if supabase_client is None or not todo_id:
        logger.error(
            "[HITL] wait_for_batch_responses: supabase_client 또는 todo_id 없음 — "
            "응답 대기 불가 (todo_id=%s)", todo_id
        )
        return result

    wait_ts = None
    if wait_started_at:
        try:
            wait_ts = datetime.fromisoformat(
                str(wait_started_at).replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            wait_ts = None

    def _entry_is_valid(fb: Dict[str, Any]) -> bool:
        if not wait_ts:
            return True
        submitted = str(fb.get("submitted_at") or "")
        if not submitted:
            return False
        try:
            return (
                datetime.fromisoformat(submitted.replace("Z", "+00:00")).timestamp()
                >= wait_ts
            )
        except Exception:
            return False

    deadline = asyncio.get_event_loop().time() + max(timeout_sec, 5.0)
    while True:
        try:
            entries = await asyncio.to_thread(
                _read_all_feedback_entries, supabase_client, todo_id
            )
            if entries:
                for fb in entries:
                    if not isinstance(fb, dict):
                        continue
                    qid = str(fb.get("question_id") or "")
                    if qid in result and result[qid] is None and _entry_is_valid(fb):
                        result[qid] = fb
                if all(v is not None for v in result.values()):
                    return result
        except Exception as exc:
            logger.warning(f"[HITL] batch 폴링 중 예외: {exc}")

        if asyncio.get_event_loop().time() >= deadline:
            missing = [qid for qid, v in result.items() if v is None]
            logger.warning(
                f"[HITL] batch timeout — todo_id={todo_id} answered="
                f"{len([v for v in result.values() if v])}/{len(result)} "
                f"missing={missing}"
            )
            return result
        await asyncio.sleep(max(poll_interval_sec, 0.5))


def _read_all_feedback_entries(supabase_client, todo_id: str) -> List[Dict[str, Any]]:
    """todolist.output.hitl_feedbacks 전체 배열 반환."""
    try:
        res = (
            supabase_client.table("todolist")
            .select("output")
            .eq("id", todo_id)
            .limit(1)
            .execute()
        )
        rows = getattr(res, "data", None) or []
        if not rows:
            return []
        out = rows[0].get("output")
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                out = {}
        if not isinstance(out, dict):
            return []
        feedbacks = out.get("hitl_feedbacks") or []
        return feedbacks if isinstance(feedbacks, list) else []
    except Exception:
        return []


async def wait_for_user_response(
    *,
    supabase_client,
    todo_id: str,
    question_id: str,
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """todolist.output.hitl_feedbacks 에 question_id 응답이 들어올 때까지 폴링.

    - timeout 초과 시 None 반환 → 호출 측은 적절한 디폴트(스킵/자동승인) 처리.
    - 응답 entry 그대로 반환 (action/selected_ids/custom_text/reason 등).
    """
    if supabase_client is None or not todo_id or not question_id:
        return None

    deadline = asyncio.get_event_loop().time() + max(timeout_sec, 5.0)
    seen_at_start = _peek_question_ids(supabase_client, todo_id)
    if question_id in seen_at_start:
        # 이미 같은 question_id 응답이 있다면(재개 케이스) 그대로 반환
        return _read_feedback_entry(supabase_client, todo_id, question_id)

    while True:
        try:
            entry = await asyncio.to_thread(
                _read_feedback_entry, supabase_client, todo_id, question_id
            )
            if entry:
                return entry
        except Exception as exc:
            logger.warning(f"[HITL] todolist.output 폴링 중 예외: {exc}")

        if asyncio.get_event_loop().time() >= deadline:
            logger.warning(
                f"[HITL] timeout — question_id={question_id} todo_id={todo_id} "
                f"after {timeout_sec}s"
            )
            return None
        await asyncio.sleep(max(poll_interval_sec, 0.5))


def _peek_question_ids(supabase_client, todo_id: str) -> set:
    """현재까지 todolist.output.hitl_feedbacks 에 들어있는 question_id 집합."""
    try:
        res = (
            supabase_client.table("todolist")
            .select("output")
            .eq("id", todo_id)
            .limit(1)
            .execute()
        )
        rows = getattr(res, "data", None) or []
        if not rows:
            return set()
        out = rows[0].get("output")
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                out = {}
        if not isinstance(out, dict):
            return set()
        feedbacks = out.get("hitl_feedbacks") or []
        if not isinstance(feedbacks, list):
            return set()
        return {
            str(fb.get("question_id") or "")
            for fb in feedbacks
            if isinstance(fb, dict) and fb.get("question_id")
        }
    except Exception:
        return set()


def _read_feedback_entry(supabase_client, todo_id: str, question_id: str) -> Optional[Dict[str, Any]]:
    """주어진 question_id 의 응답 entry 를 todolist.output 에서 찾아 반환."""
    try:
        res = (
            supabase_client.table("todolist")
            .select("output")
            .eq("id", todo_id)
            .limit(1)
            .execute()
        )
        rows = getattr(res, "data", None) or []
        if not rows:
            return None
        out = rows[0].get("output")
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                out = {}
        if not isinstance(out, dict):
            return None
        feedbacks = out.get("hitl_feedbacks") or []
        if not isinstance(feedbacks, list):
            return None
        for fb in feedbacks:
            if not isinstance(fb, dict):
                continue
            if str(fb.get("question_id") or "") == question_id:
                return fb
    except Exception:
        return None
    return None


# =========================================================================
# 3) 응답 entry 의 편의 헬퍼
# =========================================================================

def is_skipped(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry:
        return True
    action = str(entry.get("action") or "").lower()
    return action in ("skip", "skipped", "")


def selected_ids(entry: Optional[Dict[str, Any]]) -> List[str]:
    if not entry:
        return []
    ids = entry.get("selected_ids") or entry.get("selectedIds") or []
    if not isinstance(ids, list):
        return []
    return [str(x) for x in ids if str(x or "").strip()]


def custom_text(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return ""
    return str(entry.get("custom_text") or entry.get("customText") or entry.get("reason") or "").strip()


def is_approved(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry:
        return False
    action = str(entry.get("action") or "").lower()
    return action == "approve"
