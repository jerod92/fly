"""
examples/ollama_agent.py
------------------------
Use a locally running Ollama model as your LLM agent with anthill.
No API key required — everything runs on your machine.

Install Ollama:
    https://ollama.com/download

Pull a code-capable model (pick one):
    ollama pull llama3.2          # fast general-purpose, 3B params
    ollama pull codellama         # fine-tuned for code generation
    ollama pull deepseek-coder    # strong Python code generation
    ollama pull qwen2.5-coder     # strong at code, multiple sizes

Install the Python client:
    pip install ollama

Run:
    python examples/ollama_agent.py
"""

from __future__ import annotations

import anthill

try:
    import ollama
except ImportError:
    raise SystemExit(
        "Install the Ollama Python client:  pip install ollama\n"
        "Install Ollama itself from:  https://ollama.com/download"
    )


# ---------------------------------------------------------------------------
# 1. Create the Ollama agent callable
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2"  # change to whichever model you have pulled


def ollama_agent(prompt: str) -> str:
    """Send a prompt to a local Ollama model and return its text response."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# 2. Set up the anthill workflow
# ---------------------------------------------------------------------------

wf = anthill.Workflow(task="Sentiment classifier for short movie reviews")
wf.print_checklist()

# ---------------------------------------------------------------------------
# 3. Run the first step with the local model
# ---------------------------------------------------------------------------

print(f"\n--- Sending step to {OLLAMA_MODEL} via Ollama ---\n")
response = anthill.run_step(wf, ollama_agent)
print(response)

# ---------------------------------------------------------------------------
# 4. After reviewing the response and acting on it, mark the step complete
#    and move on to the next step.
# ---------------------------------------------------------------------------

# wf["data.task_definition"].complete()

# Pass the agent's output as context for the next step:
# response2 = anthill.run_step(
#     wf, ollama_agent,
#     extra_context=f"From the previous step:\n{response}",
# )
# print(response2)
# wf["data.source_decision"].complete(notes="Using PGD — fully simulable")
