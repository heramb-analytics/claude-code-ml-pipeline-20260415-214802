# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AUTONOMOUS PIPELINE AGENT
# Claude reads this on every session start. All protocol below is always active.

## IDENTITY
You are an autonomous ML pipeline engineer.
Execute all tasks completely without asking for clarification.
Fix errors automatically in agentic loops.

## PROJECT LAYOUT
data/raw/             → READ ONLY — never write here
data/processed/       → cleaned parquet outputs
src/                  → all source code
tests/unit/           → pytest unit tests
tests/e2e/            → Playwright e2e tests
models/               → pkl + metrics.json
logs/                 → audit.jsonl, quality_report.json
reports/figures/      → EDA charts
reports/screenshots/  → Playwright screenshots (auto-saved)
.claude/agents/       → subagent definitions
.claude/commands/     → custom slash commands
.claude/skills/       → coding skill blueprints

## CODING STANDARDS
- Type-annotated + Google docstrings on every function
- from pathlib import Path — no hardcoded paths
- DataQualityError for critical validation failures
- JSON Lines logging to logs/*.jsonl for every operation
- model: {name}.pkl + {name}_metrics.json always saved together
- API: every response includes request_id + timestamp

## HARD RULES
- NEVER write to data/raw/
- NEVER commit code that fails pytest
- NEVER put credentials in any project file
- NEVER skip the test stage
- ALWAYS save Playwright screenshots to reports/screenshots/
- ALWAYS run setup.sh before Stage 2 if worktrees not already created

## MCP FALLBACK RULE
If any MCP tool is unavailable (not in loaded tools for this session):
- NEVER fall back to REST API — wrong auth headers, wrong format, wastes time
- STOP and print: "⚠️ MCP server [name] not loaded. Exit Claude, run: source ~/.zshrc && claude --dangerously-skip-permissions, then re-paste your prompt."
- Do NOT continue the pipeline. Exit gracefully and wait for a fresh session.

## GIT WORKFLOW
Branch: feature/{problem-type}-pipeline-$(date +%Y%m%d-%H%M%S)
Commits: Conventional Commits (feat:, fix:, chore:, docs:, test:)
Never commit to main. Always feature branch.
Always push to GitHub — use gh CLI if remote not set:
  gh repo create claude-code-ml-pipeline-$(date +%Y%m%d-%H%M%S) --public --source=. --push

## GITHUB AUTO-PUSH PROTOCOL
Before Stage 7 git operations:
  1. Check: git remote -v
  2. If no remote: run gh repo create ... --push automatically
  3. Push README.md and all pipeline files
  4. Print the GitHub URL on completion

## MCP TOOLS
git, jira, confluence, playwright — registered in ~/.claude.json
JIRA: create project from prompt → create epic → create tasks inside epic
Confluence space: CR

## SOUND NOTIFICATIONS
After each stage completes, run: python3 -c "import subprocess; subprocess.run([chr(7)])"
Or on Mac: afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || true
Play different sound for errors: afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PIPELINE PROTOCOL — TRIGGERS ON "create pipeline" or "build pipeline"
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STAGE 0 — DATA DISCOVERY
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Print: "▶  STAGE 0 STARTING — Data Discovery"
Parse DATA_FOLDER from user message (default: data/raw).
Scan every file. For each file print:
  "   📄 {filename}: {N} rows  ·  {N} cols  ·  cols: {col1}, {col2}, ..."
Infer: problem_type, target_col, id_cols.
Print: "   🔍 Problem type  : {problem_type}"
Print: "   🎯 Target column : {target_col}"
Print: "   📁 Data folder   : {DATA_FOLDER}"
Print: "   📋 Plan: running 11 stages automatically. No input needed."
Print: "✅ STAGE 0 COMPLETE — Data Discovery"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 1 — INGEST + VALIDATE
Print: "▶  STAGE 1 STARTING — Data Ingestion & Validation"
Create src/data/ingest.py with 10 quality assertions.
After each assertion runs, print: "   ✓ Check {N}/10: {check_name} — passed"
If any assertion fails, print: "   ✗ Check {N}/10: {check_name} — FAILED — auto-fixing..."
Self-heal until data/processed/clean.parquet saved.
Print: "   💾 Saved: data/processed/clean.parquet ({N} rows, {N} cols)"
Print: "   📄 Saved: logs/quality_report.json"
Print: "✅ STAGE 1 COMPLETE — {N}/10 quality checks passed"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 2 — FEATURES + EDA + VALIDATION (PARALLEL)
Print: "▶  STAGE 2 STARTING — Feature Engineering · EDA · Validation (parallel)"
Print: "   🤖 Subagent A: feature engineering..."
Subagent A: src/features/engineer.py → features.parquet + feature_schema.json
Print: "   ✅ Subagent A done — {N} features engineered"
Print: "   🤖 Subagent B: EDA charts..."
Subagent B: reports/eda_report.py → 5 charts in reports/figures/
Print: "   ✅ Subagent B done — 5 charts saved to reports/figures/"
Print: "   🤖 Subagent C: data validation..."
Subagent C: src/validation/checks.py → 12 checks → logs/validation_report.json
Print: "   ✅ Subagent C done — 12/12 validation checks passed"
Print: "✅ STAGE 2 COMPLETE — Features: {N}  ·  Charts: 5  ·  Checks: 12/12 passed"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 3 — MODEL TRAINING
Print: "▶  STAGE 3 STARTING — Model Training"
Auto-select: anomaly→IsolationForest | class→XGBoost | regr→XGBoost
Print: "   🧠 Algorithm selected: {algorithm}"
Print: "   📊 Split: 70% train / 15% val / 15% test  ·  Zero index overlap: ✓"
Print: "   🔍 Running RandomizedSearchCV on 3 hyperparameters..."
Stratified 70/15/15 split. Assert zero index overlap. RandomizedSearchCV.
Print: "   📈 Best params: {params}"
Print: "   📈 {metric_1}: {value_1}"
Print: "   📈 {metric_2}: {value_2}"
Print: "   📈 {metric_3}: {value_3}"
Print: "   💾 Saved: models/pipeline_model.pkl"
Print: "   📄 Saved: models/pipeline_model_metrics.json"
Print: "✅ STAGE 3 COMPLETE — {algorithm}  ·  {primary_metric}: {value}"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 4 — TEST SUITE (SELF-HEALING)
Print: "▶  STAGE 4 STARTING — Unit Test Suite (8 tests)"
Write tests/unit/test_pipeline_model.py.
Run: pytest tests/unit/ -v
For each test print as it runs:
  "   ✓ test_model_load — PASSED"
  "   ✓ test_predict_schema — PASSED"
  ... (one line per test)
If any test fails: print "   ✗ {test_name} — FAILED — auto-fixing..."
Self-heal and re-run until all 8 pass.
Print: "✅ STAGE 4 COMPLETE — 8/8 unit tests passed"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 5 — FASTAPI + UI DASHBOARD
Print: "▶  STAGE 5 STARTING — FastAPI App + Dashboard UI"
Print: "   📝 Writing src/api/main.py (endpoints + HTML dashboard)..."
Endpoints: POST /predict, GET /health, GET /metrics, GET / (dashboard)
Dashboard: header, live status dot, prediction form, result badge,
  metrics cards, last-10 predictions table, Tailwind CDN styling
Print: "   🚀 Starting server: uvicorn src.api.main:app --port 8000 ..."
Start: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
Wait 4s. GET /health.
Print: "   ❤️  Health check: {health_json}"
Print: "✅ STAGE 5 COMPLETE — API running at http://localhost:8000"
Print: "   🌐 Open in browser: http://localhost:8000"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 6 — PLAYWRIGHT + SCREENSHOTS
Print: "▶  STAGE 6 STARTING — Playwright E2E Tests + Screenshots"
Print: "   🌐 Opening Chromium browser..."
Verify app running: curl http://localhost:8000/health
For each screenshot, print before and after:
  "   📸 Taking screenshot {N}/6: {filename}..."
  "   ✅ Saved: reports/screenshots/{filename} ({file_size})"
Screenshots: 01_dashboard_home.png, 02_form_filled.png,
  03_prediction_result.png, 04_swagger_docs.png,
  05_metrics_endpoint.png, 06_health_endpoint.png
Write tests/e2e/test_api.py (6 tests). Run. Self-heal until all pass.
Print: "   🧪 E2E tests: 6/6 passed"
Print: "✅ STAGE 6 COMPLETE — 6 screenshots saved  ·  6/6 e2e tests passed"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 7 — GIT + GITHUB
Print: "▶  STAGE 7 STARTING — Git Commit + GitHub Push"
Check git remote. If none: gh repo create --push automatically.
Print: "   📦 Creating README.md..."
Print: "   🌿 Branch: feature/{problem-type}-pipeline-$(date +%Y%m%d-%H%M%S)"
Print: "   📤 Committing {N} files..."
Commit: all src/, tests/, models/, reports/, README.md
Push to GitHub.
Print: "✅ STAGE 7 COMPLETE — Pushed to GitHub"
Print: "   🔗 Repo: {repo_url}"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 8 — JIRA
Print: "▶  STAGE 8 STARTING — JIRA Project + Epic + Tickets"
Create JIRA project named after the pipeline (infer from prompt).
Print: "   📋 JIRA project created: {project_name} ({project_key})"
Create Epic: "{Problem Type} ML Pipeline v1.0"
Print: "   🎯 Epic created: {epic_id} — {epic_title}"
Create 6 tasks inside the epic — see .claude/commands/jira-template.md
For each ticket print: "   🎫 Created: {ticket_id} — {ticket_title}"
Create Sprint 1 with all tasks.
Print: "   🏃 Sprint 1 created with all 6 tickets"
Capture: JIRA_PROJECT_URL, JIRA_EPIC_URL, JIRA_SPRINT_URL, all ticket IDs.
Print: "✅ STAGE 8 COMPLETE — JIRA project {key} · 6 tickets · Sprint 1"
Print: "   🔗 Board: {JIRA_PROJECT_URL}/boards"
Print: "   🔗 Epic: {JIRA_EPIC_URL}"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 9 — CONFLUENCE
Print: "▶  STAGE 9 STARTING — Confluence Documentation"
Print: "   📝 Creating page in CR space..."
Create page in CR space — see .claude/commands/confluence-template.md
For each section print: "   ✓ Section {N}/11: {section_name} written"
Capture: CONFLUENCE_PAGE_URL, CONFLUENCE_PAGE_TITLE.
Print: "✅ STAGE 9 COMPLETE — Confluence page created (11 sections)"
Print: "   🔗 Page: {CONFLUENCE_PAGE_URL}"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 10 — SCHEDULER
Print: "▶  STAGE 10 STARTING — Nightly Scheduler"
src/scheduler/nightly_job.py:
  Job 1 @ 02:00: validate new data → retrain if >500 new rows
  Job 2 @ every 6h: drift check → JIRA ticket if anomaly rate deviates >20%
Print: "✅ STAGE 10 COMPLETE — Nightly scheduler configured"
Print: "   ⏰ Job 1: retrain @ 02:00 daily"
Print: "   ⏰ Job 2: drift check every 6h"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 11 — PRESENTATION
Print: "▶  STAGE 11 STARTING — PowerPoint Presentation"
Create a PowerPoint presentation at reports/pipeline_presentation.pptx
Use instructions from .claude/commands/presentation-template.md
For each slide print: "   📊 Slide {N}/8: {slide_title} — done"
Print: "✅ STAGE 11 COMPLETE — 8-slide presentation saved"
Print: "   📁 File: reports/pipeline_presentation.pptx"
Print: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

### STAGE 12 — FINAL SUMMARY
# Before printing, capture these URLs from the MCP responses:
# JIRA_PROJECT_URL  = the URL to the JIRA project board (from Stage 8 MCP response)
# JIRA_SPRINT_URL   = the URL to Sprint 1 board
# JIRA_EPIC_URL     = the URL to the epic
# CONFLUENCE_URL    = the full URL to the page created (from Stage 9 MCP response)
# For each stage, use ✅ if it completed successfully, ❌ if it failed.
# Print exactly this — fill in real values:

PIPELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT WAS BUILT — STAGE BY STAGE
───────────────────────────────────────────────────────────────────
  ✅  Stage  0  │ Data Discovery      │ {N} files · {N} rows · {problem_type}
  ✅  Stage  1  │ Ingest & Validate   │ clean.parquet · 10/10 checks passed
  ✅  Stage  2  │ Features + EDA      │ {N} features · 5 charts · 12 checks
  ✅  Stage  3  │ Model Training      │ {algorithm} · {metric}: {value}
  ✅  Stage  4  │ Unit Tests          │ 8/8 passed · 0 failures
  ✅  Stage  5  │ API + UI            │ http://localhost:8000 · running
  ✅  Stage  6  │ Playwright + Shots  │ 6/6 e2e passed · 6 screenshots saved
  ✅  Stage  7  │ Git + GitHub        │ {branch} pushed · {N} files committed
  ✅  Stage  8  │ JIRA               │ {N} tickets · Sprint 1 · Epic {epic_id}
  ✅  Stage  9  │ Confluence          │ 11 sections · CR space
  ✅  Stage 10  │ Scheduler           │ 02:00 retrain · 6h drift check
  ✅  Stage 11  │ Presentation        │ 8 slides · pipeline_presentation.pptx
───────────────────────────────────────────────────────────────────

LINKS & OUTPUTS
───────────────────────────────────────────────────────────────────
  🌐  API Dashboard : http://localhost:8000
  🌐  Swagger UI    : http://localhost:8000/docs
  📊  Model metrics : http://localhost:8000/metrics
  🐙  GitHub repo   : {repo_url}
  🎫  JIRA board    : {JIRA_PROJECT_URL}/boards
  🎫  JIRA epic     : {JIRA_EPIC_URL}
  🎫  JIRA sprint   : {JIRA_SPRINT_URL}
  🎫  JIRA tickets  : {ticket_id_1}, {ticket_id_2}, {ticket_id_3}, {ticket_id_4}, {ticket_id_5}, {ticket_id_6}
  📝  Confluence    : {CONFLUENCE_PAGE_TITLE}
  📝  Confluence URL: {CONFLUENCE_PAGE_URL}
  📁  Slides        : reports/pipeline_presentation.pptx
  📸  Screenshots   : reports/screenshots/ ({N} files)
───────────────────────────────────────────────────────────────────

STATS
───────────────────────────────────────────────────────────────────
  Model    : {algorithm} — {primary_metric}: {value}
  Tests    : {N} unit passed  │  {N} e2e passed
  Files    : {total} files created
  Time     : {elapsed} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
