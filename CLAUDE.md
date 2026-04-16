# CLAUDE.md

## Project mission
This repository develops scientific computing software for electronic-structure / spin-model / finite-temperature simulations.
Prioritize numerical correctness, reproducibility, and physical consistency over stylistic refactors.

## Core rules
- Never change numerical conventions silently.
- Before editing algorithms, identify the relevant equations and current sign/unit conventions.
- Any change that can alter numerical results must be accompanied by:
  1. a brief explanation of why results may change,
  2. a regression or validation test,
  3. an update to docs/examples if user-facing.

  ## Implementation conventions
  - Avoid hidden global state.
  - Use explicit type annotations/docstrings for exported APIs.
  - For performance work, benchmark before and after.