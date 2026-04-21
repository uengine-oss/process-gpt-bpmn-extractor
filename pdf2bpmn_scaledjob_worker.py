#!/usr/bin/env python3
"""
KEDA ScaledJob worker (SDK 유지): 1 todo = 1 Job/Pod

요구사항:
- 기존 pdf2bpmn 에이전트의 "ProcessGPT SDK 기반 구조"는 그대로 유지
  - fetch_pending_task(RPC)로 todo claim
  - 이벤트 기록/결과 저장/FAILED 마킹 등은 process-gpt-agent-sdk의 경계 정책을 그대로 사용
- 단지 "서버(무한 폴링)"가 아니라 "배치(단건 처리 후 종료)"로만 동작을 바꿈

동작:
1) SDK initialize_db()로 Supabase 연결
2) SDK polling_pending_todos()로 todo 1개 claim
3) ProcessGPTAgentServer.process_todolist_item(row)로 처리 (SDK 로직 그대로)
4) 종료 (Pod Completed)
"""

import asyncio
import os
import logging

from pdf2bpmn_agent_executor import PDF2BPMNAgentExecutor

try:
    from processgpt_agent_sdk import ProcessGPTAgentServer
    from processgpt_agent_sdk.database import initialize_db, polling_pending_todos, get_consumer_id
except Exception as e:
    raise RuntimeError("process-gpt-agent-sdk가 필요합니다") from e


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf2bpmn_scaledjob_worker")


async def main() -> int:
    agent_orch = os.getenv("AGENT_ORCH", "pdf2bpmn")

    # 1) SDK DB 초기화 (SUPABASE_URL + SERVICE_ROLE_KEY(권장) 또는 SUPABASE_KEY 필요)
    initialize_db()

    # 2) todo 1개 claim (없으면 정상 종료)
    row = await polling_pending_todos(agent_orch, get_consumer_id())
    if not row:
        logger.info("claim 가능한 todo가 없습니다. (race/스케일 트리거 오차 가능) -> exit 0")
        return 0

    todo_id = row.get("id")
    logger.info("claimed todo: id=%s", str(todo_id))

    # 3) executor + server 구성 (서버 run-loop는 돌리지 않고 단건 처리만 수행)
    executor = PDF2BPMNAgentExecutor(
        config={
            "pdf2bpmn_url": os.getenv("PDF2BPMN_URL", "http://pdf2bpmn-api:8001"),
            "timeout": int(os.getenv("TASK_TIMEOUT", "3600")),
            "supabase_url": os.getenv("SUPABASE_URL"),
            "supabase_key": os.getenv("SERVICE_ROLE_KEY"),
        }
    )

    server = ProcessGPTAgentServer(agent_executor=executor, agent_type=agent_orch)

    # 4) SDK 경계(process_todolist_item) 그대로 실행
    await server.process_todolist_item(row)

    # 5) 종료 직전, SDK 내부 create_task 기반 이벤트/결과 저장 태스크들이 실행될 시간을 약간 제공
    # (완전 보장보다는 완화 목적. 필요하면 환경변수로 조절 가능)
    grace = float(os.getenv("PDF2BPMN_BATCH_EXIT_GRACE_SEC", "2.0"))
    if grace > 0:
        await asyncio.sleep(grace)

    # 남아있는 태스크 최대한 정리
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

