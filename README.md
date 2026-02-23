# Anthill

Need a quick ML model as part of your workflow? 

A framework for building custom PyTorch models using an LLM agent as the primary driver.

anthill does not train models for you.  It provides the structure that makes it practical for an LLM agent (Claude, GPT-4, Gemini, etc.) — guided by a human engineer — to take a task description from idea to a trained, exported model with minimal boilerplate.

---

## What it does

### A four-phase workflow checklist

Every project follows four phases.  anthill tracks your progress through them and provides the exact instruction to give your LLM agent at each step.

```
Phase 1 · Data Procurement
Phase 2 · Model Creation
Phase 3 · Training Loop
Phase 4 · Deployment & Maintenance
```

Each checklist item carries a machine-readable key, a short label for the user, and a detailed instruction block written for the LLM agent.

### Structured LLM prompts

`anthill.prompts` contains ready-to-use prompt templates for common decisions:

- Should I use existing data, scrape it, or generate it procedurally?
- Write a procedural data generator with structural consistency and distribution variability.
- Generate a text dataset (copy-paste into any LLM chat interface).
- Design an appropriate PyTorch architecture given input/output shape and parameter budget.
- Select a pretrained backbone from sensible defaults.
- Design a loss function and evaluation metrics for the task.
- Diagnose a training run from loss/metric curves.
- Write the inference wrapper and handoff document.

These are prompts, not automation.  You or your agent decides what to do with them.

### A thin agent integration layer

`anthill.run_step(workflow, agent)` connects any `(str) -> str` callable to the workflow — Claude, Ollama, OpenAI, or any HTTP API.  It builds the full context prompt and passes it to your agent.  See the [Connecting your LLM agent](#connecting-your-llm-agent) section below.

### A training loop with built-in feedback

`anthill.train.Trainer` is a supervised training loop that handles the scaffolding so the agent doesn't have to re-implement it each time:

- Pre-training sanity check: forward pass, loss plausibility, overfit-one-batch test, data loading speed, dataset validation.  Stops before a long run if something is wrong.
- Per-step logging: loss, gradient norm, samples/sec, ETA.
- Per-epoch validation: metrics, sample predictions printed to console.
- Checkpoint saving on validation loss improvement.
- Early stopping.
- NaN watchdog.
- Cosine / step learning rate scheduling.
- Callback system for extensibility.

### Export and inference wrapper

`anthill.deploy.export` saves a trained model in state_dict, TorchScript, or ONNX format alongside normalization statistics and metadata.  `anthill.deploy.ModelWrapper` is a base class for writing a clean `load() / predict()` interface around any exported model.

---

## What it does not do

- It does not automate architecture search or hyperparameter tuning.
- It does not manage datasets or download data on your behalf.
- It does not support TensorFlow, JAX, or any framework other than PyTorch.
- It does not support distributed training.
- Reinforcement learning is mentioned in the workflow and prompts but the Trainer only handles supervised learning.  RL training loops must be written by the agent.

---

## Connecting your LLM agent

An "agent" in anthill is any callable with the signature `(prompt: str) -> str`.  Pass it to `anthill.run_step(workflow, agent)` and anthill will build the full context prompt (workflow state + step instruction) and hand it off.

```python
import anthill

wf = anthill.Workflow(task="Sentiment classifier for movie reviews")

def my_agent(prompt: str) -> str:
    ...  # call any LLM here

response = anthill.run_step(wf, my_agent)
print(response)

# Review the output, then mark the step complete and move to the next:
wf["data.task_definition"].complete()
response = anthill.run_step(wf, my_agent)
```

You can also pass additional context (existing code, error messages, file contents) as `extra_context`:

```python
response = anthill.run_step(
    wf, my_agent,
    extra_context="My dataset is 5000 labelled CSV rows with columns: text, label",
)
```

### Claude (Anthropic API)

```python
import anthropic
import anthill

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

def claude_agent(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

wf = anthill.Workflow(task="Sentiment classifier for movie reviews")
response = anthill.run_step(wf, claude_agent)
print(response)
```

Install: `pip install anthill[claude]`

See [`examples/claude_agent.py`](examples/claude_agent.py) for a full walkthrough.

### Ollama (local models)

```python
import ollama
import anthill

def ollama_agent(prompt: str) -> str:
    response = ollama.chat(
        model="llama3.2",  # or codellama, deepseek-coder, qwen2.5-coder, …
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

wf = anthill.Workflow(task="Sentiment classifier for movie reviews")
response = anthill.run_step(wf, ollama_agent)
print(response)
```

Install Ollama from [ollama.com](https://ollama.com), pull a model (`ollama pull llama3.2`), then: `pip install anthill[ollama]`

See [`examples/ollama_agent.py`](examples/ollama_agent.py) for a full walkthrough.

### OpenAI

```python
from openai import OpenAI
import anthill

client = OpenAI()  # reads OPENAI_API_KEY from environment

def openai_agent(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

wf = anthill.Workflow(task="Sentiment classifier for movie reviews")
response = anthill.run_step(wf, openai_agent)
print(response)
```

Install: `pip install anthill[openai]`

### Claude Code (agentic CLI)

If you use Claude Code as your agent, export the workflow checklist as a markdown string and include it in your `CLAUDE.md` or paste it directly into the session:

```python
import anthill

wf = anthill.Workflow(task="Sentiment classifier for movie reviews")

# Export the full checklist with instructions for Claude Code to read
print(wf.to_markdown())
```

Paste the output into your `CLAUDE.md`, or pipe it directly:

```bash
python -c "
import anthill
wf = anthill.Workflow(task='Sentiment classifier for movie reviews')
print(wf.to_markdown())
" >> CLAUDE.md
```

Claude Code will then have the full workflow checklist and step-by-step instructions in its context for the entire session.

### Any other provider

Any function that takes a string and returns a string works:

```python
import httpx
import anthill

def my_api_agent(prompt: str) -> str:
    response = httpx.post(
        "http://localhost:8080/generate",
        json={"prompt": prompt},
    )
    return response.json()["text"]

wf = anthill.Workflow(task="My task")
response = anthill.run_step(wf, my_api_agent)
```

---

## Installation

```bash
pip install anthill
```

Requires Python 3.10+ and PyTorch 2.0+.

Optional: install extras for your LLM provider of choice:

```bash
pip install anthill[claude]   # Anthropic Claude API
pip install anthill[ollama]   # Ollama local models
pip install anthill[openai]   # OpenAI API
pip install anthill[hf]       # HuggingFace transformers + pretrained backbones
```

---

## Quickstart

```python
import anthill

# Start a workflow for your task
wf = anthill.Workflow(task="Analogue clock reader")
wf.print_checklist()

# Get the instruction for the next pending step and paste it into your LLM
wf.print_next_instruction()

# Mark steps complete as you go
wf["data.task_definition"].complete()
wf["data.source_decision"].complete(notes="Using PGD — fully simulable")
wf.print_checklist()

# Get a prompt to generate text training data
from anthill.prompts import data as dp
print(dp.pgd_text_prompt.render(
    task="classify the sentiment of short movie reviews",
    input_description="a 1-4 sentence movie review",
    output_description="positive | negative | neutral",
    n_samples=300,
    ood_fraction="15%",
))

# When you have a model and data loaders:
from anthill.train import Trainer, TrainConfig
from anthill.train.sanity import pre_training_check
import torch.nn as nn

config = TrainConfig(epochs=20, lr=1e-3, device="auto")
loss_fn = nn.CrossEntropyLoss()

ok = pre_training_check(model, train_loader, val_loader, config, loss_fn=loss_fn)
if ok:
    trainer = Trainer(model, train_loader, val_loader, config, loss_fn=loss_fn)
    trainer.fit()
    trainer.load_best()

# Export
from anthill.deploy import export
export(model, "./clock_model", format="state_dict", norm_stats={"mean": [0.5], "std": [0.25]})
```

---

## Workflow reference

| Key | Phase | Description |
|-----|-------|-------------|
| `data.task_definition` | Data | Define input/output types and success criterion |
| `data.source_decision` | Data | Choose: existing data / scrape / procedural generation |
| `data.pgd_generator` | Data | Write a procedural generator with structure + variability + OOD val |
| `data.validate_existing` | Data | Validate shapes, check NaN, confirm label distribution |
| `data.normalize` | Data | Compute and apply normalization on training set only |
| `model.complexity_assessment` | Model | Estimate task difficulty; choose scratch vs. pretrained backbone |
| `model.architecture_design` | Model | Implement a PyTorch model; verify forward pass and parameter count |
| `model.loss_and_metrics` | Model | Choose loss function and implement compute_metrics() |
| `train.config` | Train | Set TrainConfig hyperparameters |
| `train.sanity_check` | Train | Run pre_training_check before any full run |
| `train.run` | Train | Run the training loop; monitor feedback |
| `train.evaluate` | Train | Evaluate best checkpoint on held-out test set |
| `train.iterate` | Train | Diagnose results; decide to iterate or proceed to deploy |
| `deploy.target` | Deploy | Identify deployment environment and latency budget |
| `deploy.export` | Deploy | Export model in appropriate format |
| `deploy.inference_wrapper` | Deploy | Write a clean Predictor class |
| `deploy.tests` | Deploy | Round-trip test, edge case test, latency test |
| `deploy.handoff` | Deploy | Write HANDOFF.md for the user |

---

## Examples

| File | What it shows |
|------|---------------|
| [`examples/analogue_clock.py`](examples/analogue_clock.py) | Complete end-to-end: procedural data, small CNN, Trainer, export, inference wrapper |
| [`examples/claude_agent.py`](examples/claude_agent.py) | Drive the workflow with the Anthropic Claude API |
| [`examples/ollama_agent.py`](examples/ollama_agent.py) | Drive the workflow with a local Ollama model |

---

## Status

Alpha.  The API may change between minor versions.  Tested on Python 3.11 and PyTorch 2.2.
