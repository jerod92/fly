"""
anthill.prompts.deploy
----------------------
Prompt templates for the Deployment & Maintenance phase.
"""

from anthill.prompts._base import PromptTemplate

# ---------------------------------------------------------------------------
# Inference wrapper design
# ---------------------------------------------------------------------------

inference_wrapper_prompt = PromptTemplate(
    name="inference_wrapper_prompt",
    required_keys=["task", "raw_input_description", "raw_output_description",
                   "deployment_target", "latency_budget_ms"],
    template="""
You are writing the inference wrapper for a trained PyTorch model.

## Task
{task}

## Raw input (what the caller provides)
{raw_input_description}

## Raw output (what the caller expects)
{raw_output_description}

## Deployment target
{deployment_target}

## Latency budget
{latency_budget_ms} ms per call (on the target hardware)

## Requirements

### Class interface
```python
class Predictor:
    def __init__(self, model_dir: str):
        \"\"\"Load model, normalization stats, and any preprocessing artifacts.\"\"\"
        ...

    def predict(self, raw_input) -> ...:
        \"\"\"
        Accept raw_input in the format the caller provides.
        Apply all preprocessing internally.
        Return output in the format the caller expects.
        Thread-safe (stateless).
        \"\"\"
        ...
```

### Additional requirements
- Must work without GPU if the deployment target is CPU-only.
- Must not import training-only dependencies (e.g. torchvision.transforms for training
  augmentations) — use only what is needed for inference.
- If the model requires normalization statistics (mean/std/vocab), load them from
  `model_dir/normalization.json` (written by anthill.deploy.export).
- Handle at least one graceful error case (wrong input shape, empty input, etc.)
  with a clear error message rather than a stack trace.

### Deliverables
1. Complete `Predictor` class.
2. `if __name__ == "__main__":` CLI demo block.
3. One example of the round-trip: raw input → predict() → raw output.
""",
)

# ---------------------------------------------------------------------------
# Handoff document
# ---------------------------------------------------------------------------

handoff_prompt = PromptTemplate(
    name="handoff_prompt",
    required_keys=["task", "test_metrics", "model_description", "failure_modes",
                   "retrain_instructions", "dependencies"],
    template="""
Write a concise handoff document for the following trained model.

## Task
{task}

## Model description
{model_description}

## Test set performance
{test_metrics}

## Known failure modes
{failure_modes}

## How to retrain
{retrain_instructions}

## External dependencies
{dependencies}

---

The document should be in Markdown, under 400 words, and cover:
1. **What this model does** — one paragraph, plain language.
2. **Performance** — key metrics, with context (what does 92% accuracy mean for this task?).
3. **How to use it** — minimal code snippet to load and run the Predictor.
4. **What it struggles with** — honest list of known limitations and edge cases.
5. **How to retrain** — one-paragraph description + the exact command to run.
6. **Dependencies** — any external services, datasets, or packages that must be maintained.

Be direct and honest.  Do not overstate the model's capabilities.
""",
)

# ---------------------------------------------------------------------------
# Export format selection
# ---------------------------------------------------------------------------

export_format_prompt = PromptTemplate(
    name="export_format_prompt",
    required_keys=["deployment_target", "caller_language", "model_has_dynamic_shapes"],
    template="""
Choose the best export format for this trained PyTorch model.

## Deployment target
{deployment_target}

## Caller language / runtime
{caller_language}

## Does the model use dynamic shapes (variable-length inputs)?
{model_has_dynamic_shapes}

## Options

### torch.save (state_dict + architecture code)
- Best for: Python-only, same codebase at inference time
- Pros: Simple, exact reproduction of training behavior
- Cons: Requires the model class definition at load time; not portable

### TorchScript (torch.jit.script or torch.jit.trace)
- Best for: C++ deployments, mobile, production Python services
- Pros: No Python dependency at runtime; slightly faster inference
- Cons: Some Python constructs not supported by TorchScript compiler

### ONNX
- Best for: Cross-platform, non-Python callers, inference optimizers (TensorRT, ORT)
- Pros: Widely supported; can be optimized for various hardware
- Cons: Export can fail for complex models; dynamic shapes require care

## Your recommendation
Given the above options and the deployment requirements, which format should be used?
Justify in 2-3 sentences, then provide the exact export code (10-20 lines).
""",
)
