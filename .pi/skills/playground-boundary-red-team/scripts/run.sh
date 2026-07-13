#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
report="${1:-/tmp/playground-boundary-red-team-report.md}"
case "$report" in
  /tmp/*) ;;
  *) echo "report path must be beneath /tmp" >&2; exit 2 ;;
esac

baseline_status="$(git status --porcelain=v1)"
if [[ -n "$baseline_status" ]]; then
  echo "refusing to audit a dirty worktree" >&2
  exit 2
fi
baseline_head="$(git rev-parse HEAD)"
evidence_root="$(mktemp -d /tmp/playground-boundary-red-team.XXXXXX)"
transcript="$evidence_root/pi-transcript.txt"

cleanup() {
  rodney stop >/dev/null 2>&1 || true
  pkill -f "$evidence_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

prompt=$(cat <<EOF
Perform the explicit Bayescycle browser Playground safe boundary red-team.
Repository: $repo_root
Commit: $baseline_head
Evidence root: $evidence_root
Final report: $report

Read completely before acting:
- playground/invariants.md (normative authority)
- .pi/skills/playground-boundary-red-team/SKILL.md (authority contract)
- .pi/skills/playground-boundary-red-team/attack-catalog.md
- .pi/skills/playground-boundary-red-team/report-template.md

Execute every safe catalog probe against a temporary localhost Playground and
a second localhost request sink using only synthetic canaries. You may use bash
for Rodney and bounded temporary servers. Keep every artifact under /tmp.
Repository modification is prohibited. Report only; do not fix code. Always
clean up Rodney, browser profiles, sinks, and servers. Populate the final report
at $report and finish with a concise verdict. If a probe cannot safely be run,
mark UNTESTED and explain why. Classify only as RESISTED, ESCAPED, EXPECTED
CAPABILITY, or UNTESTED, with exact invariant citations.
EOF
)

pi --print \
  --model openai-codex/gpt-5.6-sol \
  --thinking xhigh \
  --tools read,bash \
  --no-skills \
  --no-extensions \
  --no-prompt-templates \
  --no-context-files \
  --no-session \
  --approve \
  "$prompt" | tee "$transcript"

[[ -f "$report" ]] || { echo "child did not create report: $report" >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$baseline_head" ]] || {
  echo "child changed repository HEAD" >&2
  exit 1
}
final_status="$(git status --porcelain=v1)"
[[ "$final_status" == "$baseline_status" ]] || {
  echo "child changed the worktree" >&2
  git status --short >&2
  exit 1
}
printf 'report=%s\ntranscript=%s\nhead=%s\nworktree=unchanged\n' \
  "$report" "$transcript" "$baseline_head"
