---
alwaysApply: true
---

# Code Quality

## Anti-defaults (counter common Claude tendencies)

- No premature abstractions. Three similar lines beats a helper used once.
- Don't add features or improvements beyond what was asked.
- Don't refactor adjacent code while fixing a bug.
- No dead code or commented-out blocks. Git has history.
- WHY comments, never WHAT. If code needs a "what" comment, rename instead.
- API docs at module boundaries only, not every internal function.

## Naming

- Functions and variables: `snake_case`. Types, classes, and type aliases: `PascalCase`. Macros and compile-time constants: `SCREAMING_SNAKE`.
- Namespaces: `snake_case` short names (`fexp`, `detail`). Template parameters: single uppercase letter or `PascalCase`.
- Files: `snake_case.cpp` / `snake_case.hpp`. Headers use `#pragma once`.

## Code Markers

`TODO(author): desc (#issue)` for planned work. `FIXME(author): desc (#issue)` for known bugs. `HACK(author): desc (#issue)` for ugly workarounds (explain the proper fix). `NOTE: desc` for non-obvious context. Owner and issue link required. Never `XXX`, `TEMP`, `REMOVEME`.

## File Organization

- Includes: standard library headers, then third-party, then project headers. Blank line between groups.
- Function order: public API first, then `static` helpers below in call order.
