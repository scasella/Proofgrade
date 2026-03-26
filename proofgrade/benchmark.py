from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from proofgrade.exceptions import ConfigurationError


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke instead
        raise ConfigurationError(
            "Benchmark workflows require the repo checkout dependencies. Install with `pip install -e .` from the repo root."
        ) from exc
    with path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Benchmark config must be a YAML mapping: {path}")
    return loaded


def _run_script(script_path: str, config_path: Path) -> None:
    env = os.environ.copy()
    cwd = str(Path.cwd())
    env["PYTHONPATH"] = cwd if not env.get("PYTHONPATH") else f"{cwd}:{env['PYTHONPATH']}"
    subprocess.run(
        [sys.executable, script_path, "--config", str(config_path)],
        check=True,
        env=env,
    )


def run_benchmark(config_path: str) -> None:
    path = Path(config_path)
    config = _load_yaml(path)
    study_type = config.get("study_type")
    if study_type == "final_imo_lock":
        _run_script("analysis/run_final_imo_ablation.py", path)
        _run_script("analysis/build_final_imo_remaining_error_atlas.py", path)
        return
    if study_type == "final_imo_lockbox_test":
        _run_script("analysis/run_final_imo_lockbox_test.py", path)
        return
    if study_type in {"fresh_generalization_eval", "fresh_imo_generalization"}:
        _run_script("analysis/run_fresh_generalization_eval.py", path)
        return
    if study_type == "final_imo_release":
        _run_script("analysis/build_imo_result_tables.py", path)
        _run_script("analysis/build_imo_casebook.py", path)
        return
    raise ConfigurationError(
        f"Unsupported benchmark config '{config_path}'. "
        "Expected one of final_imo_lock, final_imo_lockbox_test, fresh_generalization_eval, or final_imo_release."
    )
