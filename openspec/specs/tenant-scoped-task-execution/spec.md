# tenant-scoped-task-execution Specification

## Purpose
pdf2bpmn polls `todolist` and executes work items produced by
work-assistant-agent. One worker deployment serves all tenants, so define how
the tenant of a work item — never a default — governs its execution, and how its
human-in-the-loop signals stay a single chat turn.

Related system requirements:
`../../../../openspec/changes/tenant-isolation-and-hitl-duplicate-fix/specs/tenant-scoped-agent-streaming/spec.md`,
`../../../../openspec/changes/tenant-isolation-and-hitl-duplicate-fix/specs/chat-hitl-single-render/spec.md`.

## Requirements

### Requirement: A task executes under the tenant of its work item
The executor SHALL take `tenant_id` from the `todolist` row it picked up. It MUST
NOT substitute a default tenant when the field is missing or empty.

#### Scenario: Work item carries its tenant
- **GIVEN** a `todolist` row with `tenant_id = "acme"`
- **WHEN** the executor runs it
- **THEN** all reads, writes, and emitted events are scoped to `acme`

#### Scenario: Untenanted work item fails loudly
- **GIVEN** a `todolist` row whose `tenant_id` is null or empty
- **WHEN** the executor picks it up
- **THEN** it SHALL raise an error identifying the task
- **AND** it MUST NOT execute against any default tenant
- **AND** the failure is logged with the task id

#### Scenario: A2A input without a tenant is not defaulted
- **GIVEN** an A2A request whose `input_data` has no `tenant_id`
- **WHEN** the mock request context is constructed
- **THEN** the resulting row's `tenant_id` is empty rather than a literal tenant name
- **AND** the executor's untenanted-task rule then applies

### Requirement: HITL events belong to one turn
Each emitted human-in-the-loop question SHALL carry a stable batch key so the
frontend can render it exactly once. pdf2bpmn signals HITL steps through the
`events` table only; it MUST NOT write assistant rows to `chats`.

#### Scenario: A repeated HITL emission does not produce a second card
- **GIVEN** a HITL question already rendered in the chat for a task
- **WHEN** an equivalent event for the same task and batch is received again
- **THEN** the frontend keeps a single panel for it
