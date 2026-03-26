import argparse
import json
import os

from agent.llm import CLAUDE_MODEL
from meta_agent import MetaAgent
from utils.git_utils import diff_versus_commit, reset_paths_to_commit
from utils.meta_memory import build_meta_memory, render_meta_memory_text


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=CLAUDE_MODEL,
        help="Model to use for the agent",
    )
    parser.add_argument(
        "--chat_history_file",
        type=str,
        default="./outputs/chat_history.md",
        help="Path to chat history file",
    )
    parser.add_argument(
        "--repo_path", type=str, default="./", help="Path to the agent file"
    )
    parser.add_argument(
        "--evals_folder",
        type=str,
        default="./outputs/",
        help="Path to the folder containing the evaluation files",
    )
    parser.add_argument(
        "--iterations_left",
        type=int,
        default=None,
        help="The number of remaining iterations in which the meta agent will be invoked in future.",
    )
    parser.add_argument(
        "--git_dir", required=True, help="Path to git repository directory"
    )
    parser.add_argument(
        "--base_commit", required=True, help="Base commit hash to compare against"
    )
    parser.add_argument(
        "--outdir", required=False, default="./outputs/", help="Output directory"
    )
    parser.add_argument(
        "--use_meta_memory",
        action="store_true",
        default=False,
        help="Whether to condition the meta agent on compact structured memory from prior runs.",
    )
    parser.add_argument(
        "--meta_memory_format",
        choices=["json", "text", "both"],
        default="both",
        help="Which structured memory representation to inject into the prompt when enabled.",
    )
    parser.add_argument(
        "--meta_memory_window",
        type=int,
        default=5,
        help="How many recent generations to emphasize when summarizing prior runs.",
    )
    parser.add_argument(
        "--meta_memory_include_patch_labels",
        action="store_true",
        default=False,
        help="Whether to include deterministic patch taxonomy summaries in the structured memory.",
    )
    args = parser.parse_args()

    structured_memory = None
    if args.use_meta_memory:
        memory_payload = build_meta_memory(
            eval_path=args.evals_folder,
            window=args.meta_memory_window,
            include_patch_labels=args.meta_memory_include_patch_labels,
        )
        json_memory = json.dumps(memory_payload, indent=2)
        text_memory = render_meta_memory_text(memory_payload)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "meta_memory.json"), "w", encoding="utf-8") as f:
            f.write(json_memory + "\n")
        with open(os.path.join(args.outdir, "meta_memory.txt"), "w", encoding="utf-8") as f:
            f.write(text_memory)

        if args.meta_memory_format == "json":
            structured_memory = json_memory
        elif args.meta_memory_format == "text":
            structured_memory = text_memory
        else:
            structured_memory = f"JSON memory:\n{json_memory}\n\nText memory:\n{text_memory}"

    # Run meta agent
    meta_agent = MetaAgent(
        model=args.model,
        chat_history_file=args.chat_history_file,
    )
    meta_agent.forward(
        repo_path=args.repo_path,
        eval_path=args.evals_folder,
        iterations_left=args.iterations_left,
        structured_memory=structured_memory,
    )

    # Reset unwanted diffs
    reset_paths_to_commit(
        git_dname=args.git_dir, commit=args.base_commit, paths=["domains/"]
    )

    # Save git diff
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = (
        os.path.join(args.outdir, "model_patch.diff")
        if args.outdir
        else "model_patch.diff"
    )
    with open(model_patch_outfile, "w") as f:
        f.write(model_patch)


if __name__ == "__main__":
    main()
