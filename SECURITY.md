# Security Policy

## Supported release line

Security fixes are only guaranteed for the current public release line:

- `v0.1.x`

Legacy research artifacts retained in this repository are provided for provenance and reproducibility, not as hardened production surfaces.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub private vulnerability reporting for this repository:

1. Open the repository on GitHub.
2. Go to the `Security` tab.
3. Use `Report a vulnerability`.
4. Include the affected version, reproduction steps, impact, and any suggested mitigation.

If private reporting is not yet enabled, enable it before the public release goes live and treat that as a release blocker in [PUBLISH_CHECKLIST.md](PUBLISH_CHECKLIST.md).

## Scope notes

- The public runtime surface is the `proofgrade` CLI/API package.
- Frozen benchmark scripts and curated result artifacts are in scope for disclosure if the issue affects reproducibility, data handling, or provider credentials.
- Legacy research code outside the supported runtime is lower-priority and may be fixed on a best-effort basis.

