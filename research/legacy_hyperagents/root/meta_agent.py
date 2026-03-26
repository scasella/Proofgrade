# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None, structured_memory=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
            structured_memory (str | None, optional): Compact structured run memory to include in the prompt.
        """
        instruction = (
            "You are improving a self-improving evaluation system.\n"
            f"Repository: `{repo_path}`\n"
            f"Previous runs and evaluation outputs: `{eval_path}`\n\n"
            "Goal: make one small, concrete code change that is likely to improve future iterations or evaluation reliability.\n"
            "Start from the available evaluation evidence, then inspect only the most relevant files.\n"
            "Prefer meta-level infrastructure and agent logic such as `meta_agent.py`, `generate_loop.py`, "
            "`task_agent.py`, `run_meta_agent.py`, and files under `utils/` or `domains/` that affect scoring or iteration.\n"
            "Do not spend many turns repeatedly listing directories. Inspect specific files, make a focused change, and stop.\n"
            "Avoid editing docs or tests unless they are necessary for the code change."
        )
        if structured_memory:
            instruction += (
                "\n\nUse this compact structured memory from prior runs and evaluations."
                " Prefer preserving proven gains and correcting specific regressions over broad rewrites.\n\n"
                f"{structured_memory}"
            )

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')
