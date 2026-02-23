# anthill

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

- It does not call any LLM API.  Prompts are strings you paste into your interface of choice.
- It does not automate architecture search or hyperparameter tuning.
- It does not manage datasets or download data on your behalf.
- It does not support TensorFlow, JAX, or any framework other than PyTorch.
- It does not support distributed training.
- Reinforcement learning is mentioned in the workflow and prompts but the Trainer only handles supervised learning.  RL training loops must be written by the agent.

---

## Installation

```bash
pip install anthill
```

Requires Python 3.10+ and PyTorch 2.0+.

Optional: install `transformers` and `huggingface_hub` to use pretrained backbones:

```bash
pip install anthill[hf]
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

## Example

See [`examples/analogue_clock.py`](examples/analogue_clock.py) for a complete walkthrough of the analogue clock reading task: procedural data generation, a small CNN trained from scratch on CPU, and an inference wrapper.

---

## Status

Alpha.  The API may change between minor versions.  Tested on Python 3.11 and PyTorch 2.2.
