#!/bin/bash
# PreToolUse hook: block `git push` invocations per CLAUDE.md policy.

set -e

input=$(cat)
tool_name=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_name",""))' 2>/dev/null || echo "")
[ "$tool_name" = "Bash" ] || exit 0

cmd=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null || echo "")

# Match `git push` as a token (avoid matching e.g. `git push-test-branch`)
if printf '%s' "$cmd" | grep -qE '(^|[[:space:]&|;])git[[:space:]]+push([[:space:]]|$)'; then
  {
    echo "BLOCKED: 'git push' is forbidden by CLAUDE.md policy (local-only repo)."
    echo "If you really need to push, run it from your terminal directly."
  } >&2
  exit 2
fi

exit 0
