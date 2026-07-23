# Project Context — pdf2bpmn

## Purpose
pdf2bpmn turns documents and consulting requests into BPMN process definitions.
It polls the `todolist` table for work items created by `work-assistant-agent`,
runs the generation pipeline, emits human-in-the-loop questions through the
`events` table, and writes the resulting definition back.

This `openspec/` covers behavior this service owns alone. Cross-service
contracts (the tenant/auth chain, one-turn-one-render) live in the system-root
`../../openspec/`.

## Tech stack
- Python; A2A server (`a2a_server.py`, `src/pdf2bpmn/a2a/`), executor
  (`pdf2bpmn_agent_executor.py`), ScaledJob worker
  (`pdf2bpmn_scaledjob_worker.py`).
- Supabase for `todolist`, `events`, and `proc_def`.

## Conventions
- The tenant of a work item comes from its `todolist` row. There is **no default
  tenant** — an untenanted row fails loudly rather than running somewhere else.
  See `specs/tenant-scoped-task-execution/spec.md`.
- HITL is signalled through `events` only; pdf2bpmn does not write assistant rows
  to `chats`.
