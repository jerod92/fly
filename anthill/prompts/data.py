"""
anthill.prompts.data
--------------------
Prompt templates for the Data Procurement phase.
"""

from anthill.prompts._base import PromptTemplate

# ---------------------------------------------------------------------------
# PGD: ask an LLM to generate text data in a structured format
# ---------------------------------------------------------------------------

pgd_text_prompt = PromptTemplate(
    name="pgd_text_prompt",
    required_keys=["task", "input_description", "output_description", "n_samples"],
    template="""
You are a data generation assistant.  Your job is to produce a high-quality synthetic
dataset for the following machine learning task.

## Task
{task}

## Input description
{input_description}

## Output (label) description
{output_description}

## Requirements
1. Generate exactly {n_samples} labelled examples.
2. FORMAT — Return the data as a JSON array of objects, one object per sample:
   [{{"input": "<the input text>", "label": "<the label>"}}]
   Do not include any other text outside the JSON array.
3. VARIABILITY — Ensure wide diversity in phrasing, length, topic, style, and vocabulary.
   Do NOT repeat sentence structures or use predictable patterns.
4. BALANCE — Distribute labels evenly across all classes unless I specify otherwise.
5. DIFFICULTY — Include easy, medium, and hard examples.
   Hard examples should be ones where a human might plausibly be uncertain.
6. OUT-OF-DISTRIBUTION — Mark the last {ood_fraction} of examples with "split": "val_ood"
   and make them deliberately different in some dimension (different domain, longer text,
   unusual vocabulary, different cultural context, etc.).
   All other examples should have "split": "train".

Begin generating the dataset now.  Output only valid JSON.
""",
)

# ---------------------------------------------------------------------------
# PGD: ask an LLM to write a Python data generator function
# ---------------------------------------------------------------------------

pgd_generator_prompt = PromptTemplate(
    name="pgd_generator_prompt",
    required_keys=["task", "input_description", "output_description"],
    template="""
You are a machine learning engineer.  Write a Python data generator for the following task.

## Task
{task}

## Input
{input_description}

## Output (label)
{output_description}

## Requirements

### Code requirements
- Return a `torch.utils.data.Dataset` subclass named `SyntheticDataset`.
- Constructor signature: `__init__(self, split: str, n_samples: int, seed: int = 42)`
  - `split` is one of: 'train', 'val', 'test'
  - `val` and `test` should be **out-of-distribution** relative to `train` in at least one
    dimension (described below).
- `__len__` returns `n_samples`.
- `__getitem__` returns `(x, y)` where `x` is a `torch.Tensor` and `y` is a `torch.Tensor`.
- Must run entirely on CPU without internet access.
- One call to `__getitem__` must complete in under 10 ms.

### Distribution requirements — TRAINING split
Cover ALL of the following sources of variation (adjust to the specific task):
{variation_dimensions}

### Distribution requirements — VAL/TEST split (out-of-distribution)
Hold out at least one dimension that is NOT present in training:
{ood_dimensions}

### Output format
Return ONLY a Python code block with:
1. All necessary imports at the top.
2. The `SyntheticDataset` class.
3. A `if __name__ == "__main__":` block that:
   - Creates a train and a val dataset with n_samples=100.
   - Prints a few sample (x.shape, y) pairs.
   - Confirms both splits can be iterated in under 5 seconds.
""",
)

# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------

data_source_selection_prompt = PromptTemplate(
    name="data_source_selection_prompt",
    required_keys=["task", "input_description", "output_description"],
    template="""
You are planning the data strategy for a machine learning project.

## Project task
{task}

## Model input
{input_description}

## Model output
{output_description}

## Your job
Decide which data strategy to use.  Evaluate all three options and recommend one:

### Option A: Use existing data
- Does the user already have labelled data?
- If yes, what format is it in and what preprocessing is needed?
- Rough estimate of sample count needed for this task.

### Option B: Procedural generation
- Can the task be fully simulated in code?  (No real-world ambiguity in the labels?)
- What are the axes of variation that must be covered?
- What makes a good out-of-distribution validation set for this task?

### Option C: Web scraping / public datasets
- Are there relevant public datasets (Hugging Face, Kaggle, UCI, etc.)?
  Search using: https://huggingface.co/datasets?search=<keyword>
- Note any licensing restrictions (CC-BY, non-commercial, etc.).
- Estimate scraping complexity if needed.

## Your response format
1. Brief analysis of each option (2-3 sentences each).
2. Your recommendation and the key reason for it.
3. The first concrete action to take.
""",
)

# ---------------------------------------------------------------------------
# Normalization / preprocessing strategy
# ---------------------------------------------------------------------------

preprocessing_prompt = PromptTemplate(
    name="preprocessing_prompt",
    required_keys=["task", "input_description", "dataset_sample"],
    template="""
You are designing the preprocessing pipeline for a machine learning model.

## Task
{task}

## Input description
{input_description}

## Sample data (first few examples)
{dataset_sample}

## Your job
Design the complete preprocessing pipeline:

1. **Normalization strategy**
   - For images: per-channel mean/std, computed on training set only.
   - For text: tokenization scheme (character / BPE / whitespace), vocabulary size.
   - For numeric: standardization or min-max, computed on training set only.
   - For other: describe your approach.

2. **Augmentation strategy (training only)**
   - List 3-5 augmentations that are semantically valid (won't change the label).
   - List any augmentations to AVOID (those that would change the label).

3. **Preprocessing code**
   Implement a `preprocess_train(x)` and `preprocess_val(x)` function (or torchvision
   transform pipeline).  `preprocess_val` must NOT include random augmentations.

4. **Normalization artifact**
   Show how to compute and save the normalization statistics so they can be loaded
   at inference time.
""",
)
