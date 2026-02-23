"""
agent.py
--------
Helpers for running anthill workflow steps with any LLM agent.

An "agent" in anthill is any callable with the signature::

    (prompt: str) -> str

This means you can plug in any LLM — Anthropic Claude, local Ollama models,
OpenAI, or any HTTP API — without anthill having an opinion about which one
you use.

Example::

    import anthropic
    import anthill

    client = anthropic.Anthropic()

    def my_agent(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    wf = anthill.Workflow(task="Sentiment classifier for movie reviews")
    response = anthill.run_step(wf, my_agent)
    print(response)
    # Review the output, then mark the step complete:
    # wf["data.task_definition"].complete()

See the examples/ directory for ready-to-run scripts for Claude, Ollama,
OpenAI, and a minimal subprocess wrapper for Claude Code.
"""

from __future__ import annotations

from typing import Callable

# An agent is any callable that takes a prompt string and returns a response string.
AgentFn = Callable[[str], str]


def run_step(workflow, agent: AgentFn, *, extra_context: str = "") -> str:
    """Feed the next pending checklist step to an LLM agent and return its response.

    Builds a full context prompt containing the workflow state and the detailed
    instruction for the current step, then passes it to the agent callable.
    Does **not** automatically mark the step complete — that is the human's
    responsibility after reviewing the agent's output.

    Parameters
    ----------
    workflow : anthill.Workflow
        The current workflow object.
    agent : (str) -> str
        Any callable that accepts a prompt string and returns a response string.
        See ``examples/`` for Claude, Ollama, and OpenAI wrappers.
    extra_context : str, optional
        Additional context to include in the prompt — for example, relevant
        file contents, existing code snippets, or error messages from a
        previous step.

    Returns
    -------
    str
        The raw text response from the agent.

    Notes
    -----
    After reviewing the response and acting on it, mark the step complete::

        wf["data.task_definition"].complete()

    Then call ``run_step`` again to advance to the next step.
    """
    item = workflow.current_item()
    if item is None:
        return "All workflow phases are complete — nothing left to run."

    item.start()

    lines = [
        "You are helping build a custom PyTorch model.",
        f"Task: {workflow.task}",
        "",
        "Current workflow state:",
        workflow.to_markdown(),
        "",
    ]

    if extra_context:
        lines += [
            "Additional context:",
            extra_context,
            "",
        ]

    lines += [
        f"--- CURRENT STEP: {item.key} — {item.label} ---",
        "",
        item.agent_instruction,
    ]

    return agent("\n".join(lines))
