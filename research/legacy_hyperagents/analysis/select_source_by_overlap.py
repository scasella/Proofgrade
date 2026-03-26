"""Select the best source domain for imo_grading by shared failure-mode overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(
    "/Users/scasella/Downloads/hyperagents/HyperAgents/configs/baseline_freeze/failure_overlap_selection.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_report(config: dict[str, Any], selection: dict[str, Any], atlas: dict[str, Any]) -> str:
    lines = [
        "# Failure-Overlap Source Selection",
        "",
        "## Candidate sources",
        "",
    ]
    for candidate in atlas["candidate_sources"]:
        lines.extend(
            [
                f"### {candidate['domain']}",
                "",
                f"- Task type: {candidate['task_type']}",
                f"- Approximate run cost: `{candidate['approximate_cost']}`",
                f"- Shared executed symbols with `{atlas['target']['domain']}`: `{len(candidate['shared_executed_symbols'])}`",
                f"- Weighted overlap score: `{candidate['overlap']['weighted_overlap_score']:.2f}`",
                f"- Plausibility: {candidate['plausibility_note']}",
                "",
            ]
        )

    if atlas.get("excluded_domains"):
        lines.extend(["## Excluded domains", ""])
        for item in atlas["excluded_domains"]:
            lines.append(f"- `{item['domain']}`: {item['reason']}")
        lines.append("")

    lines.extend(
        [
            "## Ranked overlap",
            "",
            "| source | score | raw overlap | penalty | top shared symptoms |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for candidate in selection["ranked_candidates"]:
        symptom_text = ", ".join(
            item["symptom"] for item in candidate["overlap"].get("contributions", [])[:3]
        ) or "none"
        lines.append(
            f"| `{candidate['domain']}` | `{candidate['overlap']['weighted_overlap_score']:.2f}` | "
            f"`{candidate['overlap']['raw_overlap_score']:.2f}` | `{candidate['overlap']['source_only_penalty']:.2f}` | {symptom_text} |"
        )

    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"- Is `paper_review` actually a good source for `imo_grading`? `{selection['answers']['paper_review_good_source']}`",
            f"- If not, which source is better and why? `{selection['answers']['better_source']}`",
            f"- If no source has meaningful overlap, say that clearly. `{selection['answers']['structural_conclusion']}`",
            "",
        ]
    )

    winner = selection.get("selected_source")
    if winner is None:
        lines.extend(
            [
                "## Selection decision",
                "",
                "No candidate source cleared the minimum overlap threshold.",
                "",
                "> No available source domain in the current repo shows enough overlap with `imo_grading` on the shared exercised surface to make the next transfer claim plausible.",
            ]
        )
    else:
        lines.extend(
            [
                "## Selection decision",
                "",
                f"Selected source: `{winner['domain']}`",
                f"- Allowlist: `{winner['allowlist_path']}`",
                f"- Reason: `{winner['reason']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def select_source(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    atlas = _read_json(Path(config["atlas_json"]))
    threshold = float(config["overlap"]["min_meaningful_score"])
    ranked = sorted(
        atlas["candidate_sources"],
        key=lambda item: (float(item["overlap"]["weighted_overlap_score"]), float(item["overlap"]["raw_overlap_score"])),
        reverse=True,
    )
    selected = ranked[0] if ranked and float(ranked[0]["overlap"]["weighted_overlap_score"]) >= threshold else None

    paper_review = next((item for item in ranked if item["domain"] == "paper_review"), None)
    better_source_text = "No candidate source beat the overlap threshold."
    if selected is not None:
        better_source_text = (
            f"`{selected['domain']}` had the highest overlap score because its shared patchable symptom counts best matched "
            f"`{atlas['target']['domain']}` on the live shared surface."
        )
    elif ranked:
        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        if runner_up is not None:
            better_source_text = (
                f"No source was actually better enough. `{top['domain']}` ranked first numerically with "
                f"`{top['overlap']['weighted_overlap_score']:.2f}`, but that still stayed below the minimum meaningful threshold "
                f"`{threshold:.2f}`. `{runner_up['domain']}` was worse because its main failures lived off the shared target path."
            )
        else:
            better_source_text = (
                f"`{top['domain']}` ranked first numerically, but its score `{top['overlap']['weighted_overlap_score']:.2f}` "
                f"still stayed below the minimum meaningful threshold `{threshold:.2f}`."
            )

    selection = {
        "config_path": str(config_path),
        "threshold": threshold,
        "ranked_candidates": ranked,
        "selected_source": (
            {
                "domain": selected["domain"],
                "allowlist_path": selected["allowlist_path"],
                "reason": better_source_text,
            }
            if selected is not None
            else None
        ),
        "answers": {
            "paper_review_good_source": (
                "No. Its dominant source-side failures do not overlap meaningfully with imo_grading."
                if paper_review is not None and float(paper_review["overlap"]["weighted_overlap_score"]) < threshold
                else "Yes."
            ),
            "better_source": better_source_text,
            "structural_conclusion": (
                "No available source domain in this repo shows enough shared, patchable failure overlap with imo_grading."
                if selected is None
                else f"{selected['domain']} is the best available source under the overlap score."
            ),
        },
    }
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    atlas = _read_json(Path(config["atlas_json"]))
    selection = select_source(args.config)
    _write_json(Path(config["selection_json"]), selection)

    report_path = Path(config["source_selection_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(config, selection, atlas), encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
