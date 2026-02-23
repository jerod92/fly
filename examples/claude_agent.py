"""
examples/claude_agent.py
------------------------
Use the Anthropic Claude API as your LLM agent with anthill.

Install:
    pip install anthropic

Set your API key:
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    python examples/claude_agent.py
"""

from __future__ import annotations

import anthill

try:
    import anthropic
except ImportError:
    raise SystemExit(
        "Install the Anthropic SDK:  pip install anthropic\n"
        "Then set:  export ANTHROPIC_API_KEY='sk-ant-...'"
    )


# ---------------------------------------------------------------------------
# 1. Create the Claude agent callable
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


def claude_agent(prompt: str) -> str:
    """Send a prompt to Claude and return its text response."""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# 2. Set up the anthill workflow
# ---------------------------------------------------------------------------

wf = anthill.Workflow(task="Sentiment classifier for short movie reviews")
wf.print_checklist()

# ---------------------------------------------------------------------------
# 3. Run the first step with Claude
# ---------------------------------------------------------------------------

print("\n--- Sending step to Claude ---\n")
response = anthill.run_step(wf, claude_agent)
print(response)

# ---------------------------------------------------------------------------
# 4. After reviewing the response and acting on it, mark the step complete
#    and move on to the next step.
# ---------------------------------------------------------------------------

# wf["data.task_definition"].complete()

# Pass the agent's output as context for the next step:
# response2 = anthill.run_step(
#     wf, claude_agent,
#     extra_context=f"From the previous step:\n{response}",
# )
# print(response2)
# wf["data.source_decision"].complete(notes="Using PGD — fully simulable")
