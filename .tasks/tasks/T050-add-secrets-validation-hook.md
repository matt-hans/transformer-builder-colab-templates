---
id: T050
title: Add Secrets Validation Pre-commit Hook
status: pending
priority: 1
agent: infrastructure
dependencies: []
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [security, infrastructure, phase1, refactor, quick-win]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - CLAUDE.md

est_tokens: 1500
actual_tokens: null
---

## Description

Add a Git pre-commit hook that scans staged files for common secrets patterns (API keys, tokens, passwords) and blocks commits containing them. This provides **defense-in-depth** security alongside T049's .gitignore protection.

Current state: No automated secret detection. Users can accidentally commit credentials via `git add -f config_*.json` (bypassing .gitignore) or embed API keys directly in notebook cells or Python code.

Target state: `.git/hooks/pre-commit` script automatically runs before each commit, scans for patterns like `WANDB_API_KEY=`, `sk-...` (OpenAI), `ghp_...` (GitHub), and rejects commits with detected secrets. User-friendly error message guides remediation.

**Integration Points:**
- Works alongside T049's .gitignore (dual protection)
- Scans training.ipynb notebook cells for embedded credentials
- Checks Python files in utils/ for hardcoded keys
- Simple regex patterns, no external dependencies (works in Colab Git)

## Business Context

**User Story:** As an ML practitioner, I want automated protection against committing secrets, so that even if I make a mistake (using `git add -f`), the system prevents credential leaks.

**Why This Matters:**
.gitignore is passive protection—it can be bypassed with `git add -f` or by embedding secrets directly in code. Pre-commit hooks provide active scanning, catching credentials before they enter repository history.

**What It Unblocks:**
- Professional security posture (industry best practice)
- Safe collaboration on public repositories
- Reduced risk of account compromise and compute abuse

**Priority Justification:**
P1 (Critical) - Security issue. Takes 30 minutes to implement but catches the 20% of leaks that bypass .gitignore. Essential for production ML workflows.

## Acceptance Criteria

- [ ] `.git/hooks/pre-commit` script created with executable permissions (`chmod +x`)
- [ ] Script scans staged files for secrets using regex patterns (W&B keys, HF tokens, OpenAI keys)
- [ ] Detects patterns: `WANDB_API_KEY=<40-char-hex>`, `sk-[A-Za-z0-9]{32,}`, `ghp_[A-Za-z0-9]{36,}`, `hf_[A-Za-z0-9]{34,}`
- [ ] Rejects commit if secrets detected, prints filename + line number + pattern matched
- [ ] User-friendly error message guides remediation: "Remove secret from file or use environment variables"
- [ ] Allows commit with `--no-verify` flag for emergencies (documents risk in output)
- [ ] Validation: Create dummy file with `WANDB_API_KEY=abc123`, attempt `git commit` - blocked
- [ ] Validation: Commit normal code without secrets - passes
- [ ] Documentation: Add setup instructions to CLAUDE.md for collaborators to install hook

## Test Scenarios

**Test Case 1: Block W&B API Key**
- Given: File `config.json` with `"wandb_api_key": "1234567890abcdef1234567890abcdef12345678"`
- When: Run `git add config.json && git commit -m "test"`
- Then: Commit blocked, error shows `config.json:1: WANDB_API_KEY detected`, suggests using env vars

**Test Case 2: Block HuggingFace Token**
- Given: Python file with `HF_TOKEN = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"`
- When: Attempt commit
- Then: Blocked with message: `HF_TOKEN detected in utils/export.py:12`

**Test Case 3: Allow Normal Code**
- Given: Python file with `config = {"learning_rate": 5e-5, "batch_size": 4}`
- When: Commit file
- Then: Passes pre-commit hook, commit succeeds

**Test Case 4: Bypass with --no-verify Flag**
- Given: File with detected secret, user needs emergency commit
- When: Run `git commit --no-verify -m "urgent fix"`
- Then: Commit succeeds, warning printed: "⚠️ Secret detection bypassed - ensure no credentials committed"

**Test Case 5: Notebook Cell with Embedded Key**
- Given: training.ipynb cell: `wandb.login(key="abc123def456...")`
- When: Commit notebook
- Then: Blocked, message suggests using `wandb.login()` without key (prompts for input) or env vars

**Test Case 6: False Positive Handling**
- Given: Comment in code: `# Example: WANDB_API_KEY=your_key_here`
- When: Commit file
- Then: Hook flags it (conservative detection), user reviews and commits with `--no-verify` if safe, or removes example

## Technical Implementation

**Required Components:**

1. **Create `.git/hooks/pre-commit` script:**
```bash
#!/bin/bash
# Pre-commit hook: Detect secrets in staged files (T050)

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Secret patterns (regex)
declare -A PATTERNS=(
    ["WANDB_API_KEY"]='WANDB_API_KEY\s*=\s*["\047]?[a-f0-9]{40}["\047]?'
    ["HF_TOKEN"]='hf_[A-Za-z0-9]{34,}'
    ["OPENAI_API_KEY"]='sk-[A-Za-z0-9]{32,}'
    ["GITHUB_TOKEN"]='ghp_[A-Za-z0-9]{36,}'
    ["AWS_SECRET"]='aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}'
)

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

SECRETS_FOUND=0

# Scan each staged file
for FILE in $STAGED_FILES; do
    if [[ -f "$FILE" ]]; then
        for PATTERN_NAME in "${!PATTERNS[@]}"; do
            PATTERN="${PATTERNS[$PATTERN_NAME]}"
            MATCHES=$(grep -nE "$PATTERN" "$FILE" 2>/dev/null)

            if [[ -n "$MATCHES" ]]; then
                echo -e "${RED}❌ SECRET DETECTED: $PATTERN_NAME in $FILE${NC}"
                echo "$MATCHES" | while read -r LINE; do
                    echo -e "   Line: $LINE"
                done
                SECRETS_FOUND=1
            fi
        done
    fi
done

# Block commit if secrets detected
if [[ $SECRETS_FOUND -eq 1 ]]; then
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}🔒 COMMIT BLOCKED: Secrets detected in staged files${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Remediation options:"
    echo "  1. Remove secrets from files and use environment variables"
    echo "     Example: os.getenv('WANDB_API_KEY') instead of hardcoded key"
    echo "  2. Add files to .gitignore if they contain configs"
    echo "  3. Bypass hook (risky): git commit --no-verify"
    echo ""
    exit 1
fi

# Warn if using --no-verify
if [[ "$*" == *"--no-verify"* ]]; then
    echo -e "${YELLOW}⚠️  Secret detection bypassed - ensure no credentials committed${NC}"
fi

exit 0
```

2. **Make hook executable:**
```bash
chmod +x .git/hooks/pre-commit
```

3. **Add setup instructions to CLAUDE.md:**
```markdown
## Security Notes

- The template fetches arbitrary code from GitHub Gists—review before execution
- **Never commit config_*.json files**—they may contain API keys (auto-ignored via .gitignore)
- **Pre-commit hook**: Install secret detection hook for collaborators:
  ```bash
  # Copy pre-commit hook to your local repo (one-time setup)
  cp .github/hooks/pre-commit .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit
  ```
- Use environment variables for credentials in production: `os.getenv('WANDB_API_KEY')`
```

4. **Create shareable hook template in `.github/hooks/pre-commit`:**
```bash
# Same script as above, but in version-controlled .github/hooks/ directory
# Users copy this to .git/hooks/pre-commit since .git/hooks cannot be committed
```

**Validation Commands:**

```bash
# Test 1: Create file with secret and verify block
echo 'WANDB_API_KEY="1234567890abcdef1234567890abcdef12345678"' > test_secret.txt
git add test_secret.txt
git commit -m "test secret detection"
# Expected: Commit blocked with error message

# Test 2: Commit normal file
echo 'learning_rate = 5e-5' > test_normal.py
git add test_normal.py
git commit -m "test normal code"
# Expected: Commit succeeds

# Test 3: Bypass with --no-verify
git add test_secret.txt
git commit --no-verify -m "emergency commit"
# Expected: Commit succeeds with warning

# Cleanup
git reset --soft HEAD~2
rm test_secret.txt test_normal.py
```

**Code Patterns:**
- Bash script for portability (works on Linux/Mac/WSL)
- Regex patterns balance sensitivity (catch real secrets) vs. specificity (minimize false positives)
- Exit code 1 blocks commit, exit code 0 allows
- Colored output for visibility

## Dependencies

**Hard Dependencies** (must be complete first):
- None

**Soft Dependencies** (nice to have):
- [T049] .gitignore update complements this (belt-and-suspenders approach)

**External Dependencies:**
- Git (standard on all development machines)
- Bash (standard on Linux/Mac, WSL on Windows)
- grep with -E flag (standard GNU grep)

**Blocks Future Tasks:**
- None (independent security enhancement)

## Design Decisions

**Decision 1: Bash Script vs. Python Hook**
- **Rationale:** Bash has no external dependencies, runs instantly. Python would require interpreter check, slower startup.
- **Alternatives:**
  - Python with better regex - adds dependency
  - Third-party tool (e.g., detect-secrets) - requires installation
- **Trade-offs:**
  - Pro: Zero dependencies, fast execution
  - Con: Regex in bash less readable (acceptable for simple patterns)

**Decision 2: Regex Patterns (Conservative Detection)**
- **Rationale:** Err on side of false positives—better to flag safe code than miss real secrets.
- **Alternatives:**
  - Strict patterns only - may miss obfuscated secrets
  - Entropy-based detection - complex, prone to false positives on hashes
- **Trade-offs:**
  - Pro: Catches common secret formats reliably
  - Con: May flag comments with example patterns (user reviews and uses `--no-verify` if safe)

**Decision 3: Allow --no-verify Bypass**
- **Rationale:** Emergency escape hatch for legitimate cases (e.g., committing example patterns in documentation).
- **Alternatives:**
  - No bypass - users frustrated, disable hook entirely
  - Whitelist file (e.g., .secretsignore) - adds complexity
- **Trade-offs:**
  - Pro: Flexibility for edge cases
  - Con: Users can abuse bypass (acceptable, warned in output)

**Decision 4: Shareable Hook in `.github/hooks/`**
- **Rationale:** `.git/hooks/` cannot be version controlled. Users must manually copy from `.github/hooks/` to `.git/hooks/`.
- **Alternatives:**
  - Pre-commit framework (python package) - requires installation
  - Setup script that auto-copies hook - adds step to onboarding
- **Trade-offs:**
  - Pro: Simple, clear instructions in CLAUDE.md
  - Con: Manual setup for each collaborator (acceptable, one-time step)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| False positives block legitimate commits (e.g., example code) | Medium | Medium | Document `--no-verify` bypass with warning. Review flagged code before bypassing. Consider whitelist file in future iteration. |
| Users don't install hook (only works if present) | Medium | High | Add setup instructions to CLAUDE.md and README. Consider setup script in future that auto-copies hook. |
| Regex patterns don't catch obfuscated secrets (e.g., base64-encoded) | High | Low | Document limitations. Educate users that hook is not foolproof, still need careful review. Consider entropy detection in future. |
| Hook conflicts with other Git workflows (e.g., rebase, merge) | Low | Low | Hook only runs on `git commit`, doesn't interfere with other commands. Test with common workflows (rebase, cherry-pick). |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** User approved comprehensive refactor plan - Phase 1, Task 3 of 18. Defense-in-depth security to complement T049's .gitignore protection.
**Dependencies:** None (independent, complements T049)
**Estimated Complexity:** Simple (30-minute Bash script with high security ROI)

## Completion Checklist

**Code Quality:**
- [ ] Bash script follows shellcheck best practices (no SC warnings)
- [ ] Regex patterns tested against known secret formats
- [ ] Colored output works on common terminals (Linux/Mac/Windows WSL)

**Testing:**
- [ ] Created dummy file with W&B API key, verified commit blocked
- [ ] Tested HF token, OpenAI key, GitHub token patterns - all detected
- [ ] Committed normal code without secrets - passed
- [ ] Bypass flag `--no-verify` works and prints warning

**Documentation:**
- [ ] CLAUDE.md updated with setup instructions for collaborators
- [ ] .github/hooks/pre-commit created as shareable template
- [ ] Comments in script explain each pattern's purpose

**Integration:**
- [ ] Hook executable: `chmod +x .git/hooks/pre-commit`
- [ ] Tested with notebook commits (training.ipynb cells scanned)
- [ ] Tested with Python files (utils/*.py scanned)

**Definition of Done:**
Task is complete when pre-commit hook exists, detects common secrets, blocks commits with clear error messages, and CLAUDE.md documents setup for collaborators.
