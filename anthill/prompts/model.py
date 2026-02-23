"""
anthill.prompts.model
---------------------
Prompt templates for the Model Creation phase.
"""

from anthill.prompts._base import PromptTemplate

# ---------------------------------------------------------------------------
# Architecture design
# ---------------------------------------------------------------------------

architecture_design_prompt = PromptTemplate(
    name="architecture_design_prompt",
    required_keys=["task", "input_shape", "output_description", "max_params_m", "must_run_on_cpu"],
    template="""
You are a PyTorch model architect.  Design and implement a model for the following task.

## Task
{task}

## Input tensor shape
{input_shape}
(batch dimension excluded — e.g. "(3, 224, 224)" for a single RGB image)

## Output description
{output_description}

## Hard constraints
- PyTorch only.  No TensorFlow, JAX, or Keras.
- Parameter budget: ≤ {max_params_m}M trainable parameters.
- CPU training viable: {must_run_on_cpu}
  (If True, a full epoch on a 1000-sample dataset should complete in < 5 minutes on a
   modern laptop CPU.  Keep the model small enough to achieve this.)
- The model must not require internet access at runtime.

## Architecture guidelines
- Prefer depth over width.
- Use SotA micro-patterns where appropriate:
    • Depthwise-separable convolutions instead of large conv stacks
    • Residual / skip connections for networks deeper than 3 layers
    • LayerNorm (not BatchNorm) when batch sizes may be small (< 16)
    • SiLU or GELU activations
    • Scaled dot-product attention for sequence or 2D spatial reasoning
- Do NOT add unnecessary components.  A simpler model that trains faster is preferred.

## Your response

### 1. Design rationale (3-5 sentences)
Why is this architecture appropriate for the task?  What design decisions did you make?

### 2. PyTorch implementation
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    ...
```

### 3. Verification block
```python
if __name__ == "__main__":
    model = Model()
    x = torch.randn(4, {input_shape_flat})   # batch of 4
    y = model(x)
    print("Output shape:", y.shape)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {{params:,}}")
```

### 4. One-sentence description of what each major module does.
""",
)

# ---------------------------------------------------------------------------
# Pretrained backbone selection
# ---------------------------------------------------------------------------

backbone_selection_prompt = PromptTemplate(
    name="backbone_selection_prompt",
    required_keys=["task", "modality", "max_params_m"],
    template="""
You are selecting a pretrained backbone for a machine learning task.

## Task
{task}

## Input modality
{modality}
(e.g. "natural images", "natural language", "audio", "tabular", "image + text")

## Parameter budget for the full model
≤ {max_params_m}M parameters (backbone + head combined)

## Your job
Recommend the best pretrained backbone from the list of defaults below, or suggest an
alternative if none fit well.  Justify your choice.

### Default backbones by modality

IMAGE:
  - torchvision.models.mobilenet_v3_small  (~2.5M params, fast on CPU)
  - torchvision.models.efficientnet_b0     (~5.3M params, good accuracy/speed tradeoff)
  - torchvision.models.resnet18            (~11M params, widely understood baseline)

TEXT:
  - sentence-transformers/all-MiniLM-L6-v2  (384-dim, ~22M, great sentence embeddings)
  - prajjwal1/bert-tiny                      (128-dim, ~4.4M, fast and small)
  - distilbert-base-uncased                  (768-dim, ~66M, strong NLP baseline)

AUDIO:
  - facebook/wav2vec2-base                   (~95M, use only if GPU available)
  - openai/whisper-tiny.en                   (~39M, good for speech)

IMAGE + TEXT:
  - openai/clip-vit-base-patch32             (~150M, use feature extraction only)

## Response format
1. Recommended backbone (name, parameter count, output dimension).
2. How to load it and extract features (code snippet, 10-15 lines max).
3. What to freeze vs. fine-tune, and why.
4. What head to attach on top (architecture + parameter count).
5. Total model parameter count estimate.
""",
)

# ---------------------------------------------------------------------------
# HuggingFace model search
# ---------------------------------------------------------------------------

hf_model_search_prompt = PromptTemplate(
    name="hf_model_search_prompt",
    required_keys=["task", "modality", "max_params_m"],
    template="""
Search HuggingFace for a suitable pretrained model for the following task.

## Task
{task}

## Modality
{modality}

## Constraints
- Prefer models with > 1 000 downloads.
- Total parameter count ≤ {max_params_m}M (for the full model including any head).
- Model must be loadable without GPU if CPU training is required.
- Apache 2.0 or MIT license preferred.

## Search strategy
1. Go to https://huggingface.co/models?search=<keyword>&sort=downloads
   (replace <keyword> with relevant terms for this task and modality).
2. Filter by task type using the HuggingFace task taxonomy.
3. Check the model card for: parameter count, license, input/output format.

## Response format
Return a ranked list of up to 3 candidate models:
  - Model ID
  - Parameter count
  - License
  - Why it fits this task
  - One-line loading snippet

Then recommend ONE and explain why.
""",
)
