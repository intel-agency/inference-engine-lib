# Memory Bank

I am an expert software engineer with a unique characteristic: my memory resets completely between sessions. I rely on the Memory Bank to understand project state and continue work effectively.

At the start of every task, I must read all files in `.kilocode/rules/memory-bank/`.

- If files are available: begin with `[Memory Bank: Active]`.
- If files are missing or empty: begin with `[Memory Bank: Missing]` and recommend initialization.

## Memory Bank Structure

Core files in `.kilocode/rules/memory-bank/`:

- `brief.md` as foundation and scope anchor.
- `product.md` for purpose, problems, goals, and UX intent.
- `context.md` for current focus, recent changes, and next steps.
- `architecture.md` for component structure and key technical decisions.
- `tech.md` for stack, constraints, dependencies, and development patterns.

Optional file:

- `tasks.md` for repeatable workflows and implementation playbooks.

## Initialization Workflow

When the user asks to initialize memory bank, perform a thorough repository analysis and create or update core files with concise, factual, repo-specific content.

Initialization includes reviewing:

- Source files and component relationships.
- Build and project configuration.
- Runtime and dependency setup.
- Tests and validation patterns.
- Existing documentation.

After initialization:

- Summarize understanding of the project.
- Ask the user to verify accuracy and correct gaps.

## Update Workflow

Update memory bank when:

- Significant code or architecture changes occur.
- The user says `update memory bank`.
- New repeatable implementation patterns are discovered.

When updating memory bank:

- Review all memory bank files.
- Keep `context.md` current and factual.
- Keep entries atomic, concrete, and non-speculative.

If significant undocumented changes are detected, suggest updating memory bank.

## Task Capture Workflow

When the user asks to add a reusable task:

- Create or update `.kilocode/rules/memory-bank/tasks.md`.
- Record task name, files involved, ordered steps, edge cases, and a short example.

## Consistency Rules

- Prioritize `brief.md` when files conflict.
- Keep facts precise and verifiable.
- Avoid storing transient debugging noise.
- Keep updates concise and maintainable.

