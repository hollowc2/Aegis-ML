# Contributing

Thanks for helping improve Aegis-ML.

## Local setup

```bash
uv sync --locked --extra dev
uv run ruff check app demo tests
uv run pytest
```

Keep pull requests focused. Add or update tests for behavior changes, and avoid
committing large transformer artifacts. The small Phase 1 model is intentionally
bundled so the default demo remains runnable.

Security fixes should be reported privately as described in
`docs/THREAT_MODEL.md`.
