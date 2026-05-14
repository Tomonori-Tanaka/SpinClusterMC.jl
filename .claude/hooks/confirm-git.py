#!/usr/bin/env python3
"""PreToolUse hook: require confirmation for mutating git operations.

Read-only git commands (status, diff, log, show, branch/tag listing, ...)
pass through silently. Mutating operations (push, pull, commit, merge,
rebase, reset, checkout, clean, rm, branch -D, ...) return
permissionDecision=ask so Claude Code prompts the user before running them.
"""
import json
import shlex
import sys

# git subcommands that always mutate repo / working tree / remote state.
MUTATING = {
    "push", "pull", "commit", "merge", "rebase", "reset", "checkout",
    "switch", "restore", "cherry-pick", "revert", "am", "apply", "clean",
    "rm", "mv", "gc", "prune", "filter-branch", "filter-repo",
    "update-ref", "fast-import",
}
# global options that consume the following token as their value, so the
# subcommand sits one token further (e.g. `git -C /path push`).
OPT_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def git_subcommand(tokens, i):
    """Given tokens[i] == 'git', return (subcommand, rest_args)."""
    j = i + 1
    while j < len(tokens):
        tok = tokens[j]
        if tok in OPT_WITH_VALUE:
            j += 2
            continue
        if tok.startswith("-"):
            j += 1
            continue
        return tok, tokens[j + 1:]
    return None, []


def is_mutating(sub, rest):
    if sub in MUTATING:
        return True
    if sub == "branch":
        return any(a in ("-d", "-D", "-m", "-M", "--delete", "--move") for a in rest)
    if sub == "tag":
        return any(a in ("-d", "--delete", "-f", "--force") for a in rest)
    if sub == "stash":
        # bare `git stash` == `git stash push`; list/show are read-only.
        return not rest or rest[0] in ("push", "save", "pop", "drop", "clear", "apply")
    if sub == "remote":
        # `git remote` / `git remote -v` list; add/remove/set-* mutate config.
        return bool(rest) and rest[0] in (
            "add", "remove", "rm", "rename", "set-url", "set-head",
            "set-branches", "prune", "update",
        )
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    cmd = data.get("tool_input", {}).get("command", "")
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    for i, tok in enumerate(tokens):
        if tok == "git":
            sub, rest = git_subcommand(tokens, i)
            if sub and is_mutating(sub, rest):
                print(json.dumps({
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": (
                            "Mutating git operation '%s' — confirm before "
                            "running (CLAUDE.md: git 操作は必ず"
                            "確認を取る)." % sub
                        ),
                    }
                }))
                sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
