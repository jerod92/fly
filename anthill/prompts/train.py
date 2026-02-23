"""
anthill.prompts.train
---------------------
Prompt templates for the Training Loop phase.
"""

from anthill.prompts._base import PromptTemplate

# ---------------------------------------------------------------------------
# Loss & metric design
# ---------------------------------------------------------------------------

loss_design_prompt = PromptTemplate(
    name="loss_design_prompt",
    required_keys=["task", "output_description", "class_distribution"],
    template="""
You are designing the loss function and evaluation metrics for a model.

## Task
{task}

## Model output
{output_description}

## Training label distribution
{class_distribution}

## Your job

### 1. Loss function
Choose and implement the loss function.  Consider:
- Output type (logits vs. probabilities vs. continuous scalar vs. sequence)
- Class imbalance (if any class has < 10% of samples, consider weighted loss)
- Whether the task has natural ordinal structure (e.g. rating 1-5: MSE may be better than CE)
- Whether multiple losses are needed (e.g. auxiliary task for regularization)

Implement as a `compute_loss(logits, targets) -> torch.Tensor` function.

### 2. Evaluation metrics
Implement `compute_metrics(preds, targets) -> dict[str, float]`:
- PRIMARY metric (one number that defines success, matches the success criterion)
- 2-3 SECONDARY metrics that provide additional diagnostic information

Metrics must be:
- Computable on GPU tensors without converting to numpy (until the final aggregation)
- Meaningful to a non-expert (include human-readable keys, e.g. "accuracy_pct" not "acc")

### 3. Sanity check values
For random model weights, what is the EXPECTED value of:
- Training loss: ___
- Primary metric: ___
(These are used in the pre-training sanity check to confirm the loss is plausible.)
""",
)

# ---------------------------------------------------------------------------
# Training failure diagnosis
# ---------------------------------------------------------------------------

training_diagnosis_prompt = PromptTemplate(
    name="training_diagnosis_prompt",
    required_keys=["train_loss_curve", "val_loss_curve", "primary_metric_curve",
                   "gradient_norm_curve", "task", "model_description"],
    template="""
You are diagnosing a training run for a machine learning model.

## Task
{task}

## Model
{model_description}

## Training loss curve (epoch: loss)
{train_loss_curve}

## Validation loss curve (epoch: loss)
{val_loss_curve}

## Primary metric curve (epoch: metric_value)
{primary_metric_curve}

## Gradient norm curve (step: grad_norm)
{gradient_norm_curve}

## Your diagnosis

### Pattern recognition
Identify which pattern best describes this training run:
  A. HEALTHY — loss decreasing, val follows train, metrics improving
  B. UNDERFITTING — both losses high and flat from early on
  C. OVERFITTING — train loss drops but val loss plateaus or rises
  D. DIVERGING — loss increasing or exploding
  E. OSCILLATING — high variance in loss / metrics, no clear trend
  F. STUCK — loss decreases then completely plateaus before reaching target

### Root cause analysis
For the identified pattern, list the 2-3 most likely root causes.

### Recommended interventions
List in priority order.  For each intervention:
  - What to change (exact hyperparameter or code change)
  - Why it addresses the root cause
  - Expected effect on the next training run

### Red flags (if any)
Note any unusual observations in the gradient norms or loss curves that need
immediate attention before re-running.
""",
)

# ---------------------------------------------------------------------------
# RL reward design
# ---------------------------------------------------------------------------

reward_design_prompt = PromptTemplate(
    name="reward_design_prompt",
    required_keys=["task", "action_space", "state_description", "success_criterion"],
    template="""
You are designing the reward function for a reinforcement learning task.

## Task
{task}

## Action space
{action_space}

## State description
{state_description}

## Success criterion
{success_criterion}

## Requirements for a good reward function

1. DENSITY — A sparse reward (only at episode end) is usually too hard to learn from.
   Provide intermediate rewards that approximate progress toward the goal.

2. SCALE — Keep reward values in approximately [-1, 1] or [0, 1].
   Unnormalized rewards cause instability in most RL algorithms.

3. NO REWARD HACKING — Ensure the agent cannot achieve high reward without solving
   the actual task.  Check for obvious shortcuts.

4. SHAPING SIGNAL — If the task has a clear "distance to goal" metric, use it as a
   dense shaping term added to the sparse task reward.

## Your response

### 1. Reward function implementation
```python
def compute_reward(state, action, next_state, done) -> float:
    ...
```

### 2. Reward breakdown
List each reward component and its magnitude and sign.

### 3. Episode termination conditions
List all conditions that end an episode (success, failure, timeout).

### 4. Random policy baseline
What reward does a uniformly random policy achieve on average?
(This is your lower bound — the learned policy must beat it within 100 episodes.)

### 5. Suggested RL algorithm
Based on the action space type and state dimensionality:
  - Discrete actions, low-dim state: DQN or PPO
  - Continuous actions: PPO or SAC
  - Note: anthill.train.Trainer supports supervised learning by default.
    For RL, provide a custom training loop using the reward function above.
""",
)
