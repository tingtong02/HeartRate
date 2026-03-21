# Codex Workflow for HeartRate_CNN

## 1. Workspace

Project root:
`~/learning/HeartRate_CNN`

Conda environment:
`HeartRate_env`

Codex should always work inside the project root.

---

## 2. General Rules

1. Read `docs/PROJECT_TASKS.md` before making major changes.
2. Prefer small, reviewable commits.
3. Do not refactor unrelated modules.
4. Keep all experiments reproducible.
5. When adding a module, also add:
   - config
   - evaluation entrypoint
   - minimal documentation
6. If a dependency is needed, update environment setup files.

---

## 3. Development Order

Codex should follow this order unless explicitly instructed otherwise:

1. Stage 0: dataset and evaluation foundation
2. Stage 1: robust heart rate baseline
3. Stage 2: beat detection and PRV/HRV
4. Stage 3: SQI and motion robustness
5. Stage 4: event detection and irregular pulse screening
6. Stage 5: respiration and multitask fusion

---

## 4. Coding Standards

- Use Python
- Use clear module boundaries
- Prefer typing where reasonable
- Avoid monolithic notebooks for core logic
- Put reusable code in packages, not notebooks
- Keep training and evaluation scripts separate

---

## 5. Before Implementing a New Module

Codex should:
1. Inspect existing repo structure
2. Reuse compatible utilities
3. Propose file additions or modifications
4. Implement the smallest usable version first
5. Add evaluation support

---

## 6. After Implementing a New Module

Codex should:
1. Run relevant tests or smoke checks
2. Run a minimal experiment if possible
3. Summarize changed files
4. Summarize remaining work
5. Update relevant docs

---

## 7. Forbidden Behaviors

Codex must not:
- Delete large parts of the repo without clear reason
- Introduce hidden assumptions about private datasets
- Skip metrics and only provide qualitative output
- Hardcode machine-specific absolute paths except project root assumptions
- Claim a model works without running at least a minimal validation path