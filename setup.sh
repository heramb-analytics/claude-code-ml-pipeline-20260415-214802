#!/bin/bash
# setup.sh — pre-pipeline setup, called automatically by Claude before Stage 2
# Location: project root

echo "=== PRE-PIPELINE SETUP ==="

# 1. Ensure git is initialised
if [ ! -d ".git" ]; then
  git init && git checkout -b main
  git config user.name "${GIT_NAME:-Claude Code Bot}"
  git config user.email "${GIT_EMAIL:-claude@pipeline.local}"
  git add . && git commit -m "chore: auto-init by setup.sh"
  echo "  ✅ git initialised"
else
  echo "  ✅ git already initialised"
fi

# 2. Ensure GitHub remote exists
if ! git remote | grep -q origin; then
  if command -v gh &>/dev/null; then
    REPO_NAME=$(basename $(pwd))
    gh repo create "$REPO_NAME" --public --source=. --push --yes 2>/dev/null || true
    echo "  ✅ GitHub repo created: $REPO_NAME"
  else
    echo "  ⚠️  No GitHub remote and gh CLI not found. Push will fail in Stage 7."
  fi
else
  echo "  ✅ GitHub remote already set"
fi

# 3. Ensure all Python packages installed
python3 -c "import pandas, sklearn, fastapi, xgboost, playwright" 2>/dev/null || {
  echo "  Installing missing Python packages..."
  pip3 install -q pandas numpy scikit-learn xgboost fastapi uvicorn pytest \
    playwright pytest-playwright APScheduler matplotlib seaborn scipy requests httpx pyarrow
  python3 -m playwright install chromium
}
echo "  ✅ Python packages OK"

# 4. Create git worktrees for Stage 2 parallel processing
if git worktree list | grep -q "wt-features-$(date +%Y%m%d-%H%M%S)"; then
  echo "  ✅ Worktrees already exist"
else
  git worktree add ../wt-features-$(date +%Y%m%d-%H%M%S) -b feature/features-agent-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
  git worktree add ../wt-eda-$(date +%Y%m%d-%H%M%S) -b feature/eda-agent-$(date +%Y%m%d-%H%M%S)     2>/dev/null || true
  git worktree add ../wt-val-$(date +%Y%m%d-%H%M%S) -b feature/validation-agent-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
  echo "  ✅ Worktrees created: wt-features-DATE, wt-eda-DATE, wt-val-DATE"
fi

# 5. Ensure reports/screenshots folder exists
mkdir -p reports/screenshots reports/figures logs
echo "  ✅ Output folders ready"

echo "=== SETUP COMPLETE — Starting pipeline stages ==="
