# Contributing

Thanks for contributing to `proofgrade`.

This repository now has a narrow public scope:

- supported product surface: the `proofgrade` package, CLI, API, Docker path, and frozen `imo_grading` reproducibility line
- research-only material: legacy HyperAgents scaffolding and non-product benchmark history

Please keep pull requests aligned with that scope.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` if you want to run live grading or live benchmark reproduction.

## Common commands

```bash
make test
make lint
make cli-smoke
```

## Pull request expectations

1. Keep the frozen benchmark line frozen unless the change is explicitly a reproducibility or packaging fix.
2. Do not widen public claims beyond the documented product positioning.
3. Add or update tests for runtime-facing behavior.
4. Update docs if the CLI, API, config, or release story changes.
5. Prefer small, reviewable changes over broad refactors.

## Code style

- Python 3.12
- keep logs and error messages human-readable
- keep product behavior explicit rather than hidden in environment magic
- preserve the current frozen grading-policy variants and their names

## Issues

- Use GitHub issues for bugs, UX problems, and documentation gaps.
- Use GitHub security reporting for vulnerabilities. See [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under [Apache-2.0](LICENSE).
