# Publish Checklist

- [ ] Release audit complete and reviewed: `docs/release_audit.md`
- [ ] Public scope decision complete and reviewed: `docs/public_scope_decision.md`
- [ ] Repo scrub complete
- [ ] Secrets scan complete
- [ ] Large artifact review complete
- [ ] Old root archive fragments removed
- [ ] README, QUICKSTART, docs, and community files complete
- [ ] Apache-2.0 license present as `LICENSE`
- [ ] `proofgrade` package installs locally
- [ ] CLI verified locally
- [ ] API verified locally
- [ ] Docker build verified locally
- [ ] CI workflows committed and green
- [ ] Benchmark docs and curated result package verified
- [ ] GitHub private vulnerability reporting enabled
- [ ] Release notes ready: `RELEASE_NOTES_v0.1.0.md`
- [ ] Tag command prepared
- [ ] GitHub release steps prepared

## Local verification commands

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest tests.test_final_imo_lock tests.test_final_imo_release tests.test_proofgrade_runtime tests.test_proofgrade_api
python -m proofgrade.cli version
PYTHONPATH=. python analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml
PYTHONPATH=. python analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml
docker build -t proofgrade:0.1.0 .
```

## Manual release commands

```bash
git checkout -b release/v0.1.0
git add .
git commit -m "Prepare public v0.1.0 release candidate"
git push origin release/v0.1.0
git checkout main
git pull --ff-only
git merge --ff-only release/v0.1.0
git tag v0.1.0
git push origin main
git push origin v0.1.0
gh release create v0.1.0 --title "proofgrade v0.1.0" --notes-file RELEASE_NOTES_v0.1.0.md
```
