# AGENTS.md

## Purpose

This file defines the required workflow for AI coding agents working in this repository.

Keep this file short and operational. Project-specific plans, architecture, tasks, decisions, and setup details belong in the linked docs under `docs/`.

## Required Reading

Before making changes, read:

1. `AGENTS.md`
2. `docs/project-plan.md` if it exists
3. `docs/architecture.md` if it exists
4. `docs/tasks.md` if it exists
5. Any source files directly related to the requested work

If a referenced doc does not exist, continue without it. Do not create missing docs unless the task requires it or the user asks for project scaffolding.

## Source of Truth

Use these files when present:

- `docs/project-plan.md`: product goals, roadmap, milestones, non-goals, acceptance criteria
- `docs/architecture.md`: system structure, boundaries, data flow, technical patterns
- `docs/tasks.md`: active tasks, pending work, completed task markers
- `docs/devlog.md`: completed work log
- `docs/decisions.md`: durable product and architecture decisions
- `docs/setup.md`: install, run, test, build, deploy, and troubleshooting commands
- `docs/api.md`: API contracts, endpoints, schemas, request/response examples
- `docs/ui.md`: UI patterns, design rules, component conventions
- `docs/testing.md`: testing strategy, required checks, known test limitations

Do not treat chat history as the source of truth when repository docs or code disagree with it. Prefer the current repository state.

## Work Intake

When the user requests work:

1. Restate the task internally as a concrete implementation goal.
2. Inspect the relevant files before editing.
3. Identify the smallest safe change.
4. Avoid broad rewrites unless explicitly requested.
5. Preserve existing behavior unless the task requires changing it.

If the request is ambiguous, make the safest reasonable assumption and proceed. Ask a question only when the ambiguity could cause destructive, irreversible, or clearly wrong work.

## Task Tracking

Use `docs/tasks.md` when present.

Add or update a task when:

- The user requests a feature, bug fix, refactor, or investigation
- A bug is discovered during implementation
- Follow-up work is required to complete the requested change
- Technical debt directly blocks or risks the requested work

Do not add speculative cleanup tasks, nice-to-haves, or broad refactors unless the user explicitly asks for them.

When completing a tracked task:

1. Implement the change
2. Run relevant validation
3. Update `docs/devlog.md` if present
4. Mark the task complete in `docs/tasks.md`
5. Do not delete completed tasks

Prefer this task format:

```markdown
### [Priority] Task Title

Description and acceptance criteria.

- **Files**: Affected areas
- **Context**: Why this is needed
- **Status**: Active | Blocked | Done
```
