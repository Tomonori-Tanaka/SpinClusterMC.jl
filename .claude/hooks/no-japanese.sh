#!/bin/bash
# PostToolUse hook: block Edit/Write that introduces Japanese characters into
# src/, test/, or benchmark/ .jl files.
#
# Policy: feedback_comments_in_english.md (English-only in source).
# docs/ is exempt.

set -e

input=$(cat)
tool_name=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_name",""))' 2>/dev/null || echo "")

case "$tool_name" in
  Edit|Write|MultiEdit) ;;
  *) exit 0 ;;
esac

file_path=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null || echo "")
[ -z "$file_path" ] && exit 0
[ -f "$file_path" ] || exit 0

# Only enforce on .jl under src/test/benchmark
case "$file_path" in
  *.jl) ;;
  *) exit 0 ;;
esac
cwd=$(pwd)
rel="${file_path#$cwd/}"
case "$rel" in
  src/*|test/*|benchmark/*) ;;
  *) exit 0 ;;
esac

hits=$(perl -CSD -ne 'print "  line $.: $_" if /[\x{3040}-\x{30FF}\x{4E00}-\x{9FFF}]/' "$file_path" || true)
if [ -n "$hits" ]; then
  {
    echo "BLOCKED: Japanese characters detected in $rel"
    echo "(policy: feedback_comments_in_english.md — source/.jl files must be English-only)"
    echo "$hits"
    echo "Re-edit the file to replace the Japanese text with English."
  } >&2
  exit 2
fi

exit 0
