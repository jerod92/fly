"""
workflow.py
-----------
Defines the four-phase workflow checklist for LLM-assisted model development.

The checklist is designed to be consumed by an LLM agent AND printed for the
human user.  Each CheckItem carries:
  - a machine-readable key
  - a short description for the user
  - a detailed instruction block aimed at the LLM agent
  - an optional feedback hook (callable that receives the project state)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()


class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


_STATUS_ICONS = {
    Status.PENDING:     "[ ]",
    Status.IN_PROGRESS: "[~]",
    Status.DONE:        "[x]",
    Status.SKIPPED:     "[-]",
    Status.BLOCKED:     "[!]",
}

_STATUS_COLORS = {
    Status.PENDING:     "dim",
    Status.IN_PROGRESS: "yellow",
    Status.DONE:        "green",
    Status.SKIPPED:     "blue",
    Status.BLOCKED:     "red",
}


@dataclass
class CheckItem:
    key: str
    label: str
    agent_instruction: str
    phase: str
    status: Status = Status.PENDING
    notes: str = ""
    on_complete: Optional[Callable] = field(default=None, repr=False)

    def complete(self, notes: str = "") -> None:
        self.status = Status.DONE
        if notes:
            self.notes = notes
        if self.on_complete:
            self.on_complete(self)

    def start(self) -> None:
        self.status = Status.IN_PROGRESS

    def skip(self, reason: str = "") -> None:
        self.status = Status.SKIPPED
        self.notes = reason

    def block(self, reason: str = "") -> None:
        self.status = Status.BLOCKED
        self.notes = reason


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES = ["data", "model", "train", "deploy"]

PHASE_TITLES = {
    "data":   "Phase 1 · Data Procurement",
    "model":  "Phase 2 · Model Creation",
    "train":  "Phase 3 · Training Loop",
    "deploy": "Phase 4 · Deployment & Maintenance",
}

PHASE_DESCRIPTIONS = {
    "data": (
        "Identify, obtain, and validate the training data.  "
        "Before any model is designed, the data pipeline must be solid."
    ),
    "model": (
        "Design and implement a PyTorch model that fits the task.  "
        "Start small, verify shapes, then scale up only if needed."
    ),
    "train": (
        "Run a well-structured training loop with continuous feedback.  "
        "Catch bugs and data issues on the first batch, not at epoch 50."
    ),
    "deploy": (
        "Export, wrap, and serve the trained model in a way that matches "
        "the real-world environment it will run in."
    ),
}


def build_checklist() -> list[CheckItem]:
    """Return the canonical ordered list of CheckItems for all four phases."""
    items: list[CheckItem] = []

    # ------------------------------------------------------------------ DATA
    items += [
        CheckItem(
            key="data.task_definition",
            phase="data",
            label="Define the task inputs and outputs precisely",
            agent_instruction=textwrap.dedent("""\
                Ask the user (or infer from context):
                  1. What is the model's INPUT?  (image, text, numeric vector, time-series, …)
                     Specify dtype, shape, and domain (e.g. "RGB image 224×224", "sentence up to 512 tokens").
                  2. What is the model's OUTPUT?  (class label, scalar, sequence, action distribution, …)
                     Specify dtype, shape, and semantics (e.g. "integer 0-11 = hour hand position").
                  3. What does 'correct' look like?  Write a one-sentence success criterion that can be
                     measured on a held-out test set.
                Document the answers before proceeding.  All downstream decisions depend on them.
            """),
        ),
        CheckItem(
            key="data.source_decision",
            phase="data",
            label="Decide on data source: existing / scrape / procedural generation",
            agent_instruction=textwrap.dedent("""\
                Evaluate the three options in this order:
                  A. EXISTING DATA — ask the user if they have labelled data already.
                     If yes, jump to data.validate_existing.
                  B. PROCEDURAL GENERATION (PGD) — preferred when the task can be fully
                     simulated (game states, synthetic images, rule-based text, …).
                     Choose PGD if the domain is fully specifiable programmatically.
                  C. WEB SCRAPING — last resort; only if neither A nor B is feasible.
                     Identify 2-3 candidate public sources; note licensing constraints.
                State the chosen approach and brief justification before continuing.
            """),
        ),
        CheckItem(
            key="data.pgd_generator",
            phase="data",
            label="[PGD] Write a procedural data generator with structure + variability",
            agent_instruction=textwrap.dedent("""\
                Design a generator function (or class) that produces (input, label) pairs.

                Requirements:
                  STRUCTURE  — Every sample must have the same dtype/shape contract.
                               Labels must be deterministic given the generator parameters.
                  VARIABILITY — Cover the full distribution: vary every aesthetic/stylistic
                               dimension that should be invariant to (color, noise, rotation,
                               font, size, background, …).  The model must not be able to
                               short-cut on any single correlated feature.
                  OOD VALIDATION — Implement a second generator (or a mode flag) that holds
                               out one or more dimensions not seen in training.  For example,
                               if training on digit images: val uses a font family absent from
                               training.  This catches overfitting to spurious correlations.

                The generator must:
                  • Accept a random seed for reproducibility.
                  • Accept a split argument ('train' | 'val' | 'test').
                  • Return a torch.utils.data.Dataset or an iterable of (x, y).
                  • Be fast enough to generate a batch in < 50 ms on CPU.

                If text data is needed and a runtime generator is impractical, instead produce
                a copy-paste-ready prompt (see anthill.prompts.data.pgd_text_prompt) that the
                user can run in any LLM chat interface to generate labelled text samples in a
                specified JSON/CSV format.
            """),
        ),
        CheckItem(
            key="data.validate_existing",
            phase="data",
            label="Validate existing / scraped / generated dataset",
            agent_instruction=textwrap.dedent("""\
                Run anthill.train.sanity.check_dataset(dataset) and address every WARNING or ERROR it raises.

                Also manually verify:
                  1. Class / label balance — report counts per class.  If imbalance > 10×,
                     recommend oversampling, undersampling, or weighted loss.
                  2. Shape consistency — assert all samples share the declared input shape.
                  3. NaN / Inf / corrupt samples — scan and remove or flag.
                  4. Representative samples — print/display 4 random samples to the user so
                     they can confirm the data looks correct before any training begins.
                  5. Train / val / test split — confirm sizes and that val/test are held out
                     before any preprocessing statistics (mean, std) are computed.

                Report a one-paragraph data summary to the user.
            """),
        ),
        CheckItem(
            key="data.normalize",
            phase="data",
            label="Compute and apply dataset-level normalization / tokenization",
            agent_instruction=textwrap.dedent("""\
                For image data: compute per-channel mean and std on the TRAINING set only.
                For text data: choose a tokenizer (byte-pair, character, whitespace) and fit
                  vocabulary on the TRAINING set only.
                For numeric data: compute min/max or mean/std on TRAINING set only.

                Wrap normalization in a transform or Dataset.__getitem__ so it is applied
                consistently at runtime (including during inference).  Save the statistics
                alongside the model checkpoint so they can be reloaded at inference time.
            """),
        ),
    ]

    # ----------------------------------------------------------------- MODEL
    items += [
        CheckItem(
            key="model.complexity_assessment",
            phase="model",
            label="Assess task difficulty and choose scratch vs. pretrained backbone",
            agent_instruction=textwrap.dedent("""\
                Decide whether to train from scratch or use pretrained components:

                  TRAIN FROM SCRATCH when:
                    • Input dimensionality is low (e.g. < 10k pixels, short sequences)
                    • Domain is highly synthetic / unlike natural images or text
                    • A small model (< 1M parameters) is likely sufficient

                  USE PRETRAINED COMPONENTS when:
                    • Input is natural language → use a sentence-transformer or small BERT variant
                    • Input is natural images → use a small CNN backbone (MobileNetV3, EfficientNet-B0)
                    • Task requires joint image+text understanding → use a CLIP-style encoder
                    • HuggingFace model search: `huggingface_hub.list_models(task=<task>, sort="downloads")`
                      Pick the smallest model with > 1k downloads that fits the task.

                  DEFAULT PRETRAINED SHORTCUTS (use these unless there is a good reason not to):
                    • Image features:  torchvision.models.mobilenet_v3_small(pretrained=True)
                    • Text features:   sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~22M params)
                    • Image+text:      openai/clip-vit-base-patch32

                State the decision and parameter count estimate before writing any model code.
            """),
        ),
        CheckItem(
            key="model.architecture_design",
            phase="model",
            label="Design and implement the PyTorch model",
            agent_instruction=textwrap.dedent("""\
                Implement the model as a torch.nn.Module.

                Constraints:
                  • PyTorch only.  No TensorFlow, JAX, or Keras.
                  • Start SMALL.  Try to keep total parameters under 5M.  A model that trains
                    in < 5 minutes on CPU is strongly preferred over a larger one.
                  • Use SotA micro-architecture components where appropriate:
                      - Depthwise-separable convolutions instead of standard 3×3 convs
                      - LayerNorm / RMSNorm instead of BatchNorm when batch size is small
                      - SiLU / GELU activations
                      - Residual connections for networks deeper than 3 layers
                      - Attention (scaled dot-product) for sequence or spatial tasks
                  • The forward() method must accept exactly the shape produced by the dataset
                    and return exactly the shape required by the loss function.

                After implementing:
                  1. Run a single forward pass with a dummy batch and print the output shape.
                  2. Count parameters: sum(p.numel() for p in model.parameters() if p.requires_grad)
                  3. Estimate memory: anthill.train.sanity.estimate_memory(model, input_shape, batch_size)
                  4. Confirm CPU training is feasible (< 30 min for a reasonable epoch count).
            """),
        ),
        CheckItem(
            key="model.loss_and_metrics",
            phase="model",
            label="Choose loss function and evaluation metrics",
            agent_instruction=textwrap.dedent("""\
                Choose loss and metrics appropriate to the output type:

                  CLASSIFICATION (discrete labels):
                    Loss:    nn.CrossEntropyLoss (multi-class) or nn.BCEWithLogitsLoss (binary/multi-label)
                    Metrics: accuracy, top-k accuracy, confusion matrix
                  REGRESSION (continuous scalar/vector):
                    Loss:    nn.MSELoss or nn.HuberLoss (more robust to outliers)
                    Metrics: MAE, RMSE, R²
                  SEQUENCE (token-level):
                    Loss:    nn.CrossEntropyLoss with ignore_index for padding
                    Metrics: token accuracy, perplexity
                  REINFORCEMENT LEARNING:
                    Loss:    policy gradient / PPO / DQN depending on action space
                    Reward:  define the reward function explicitly; confirm it is dense enough
                             to give learning signal in the first 100 steps.

                Implement a compute_metrics(preds, targets) function that returns a dict.
                This will be called at the end of every validation epoch.
            """),
        ),
    ]

    # ----------------------------------------------------------------- TRAIN
    items += [
        CheckItem(
            key="train.config",
            phase="train",
            label="Define training configuration (hyperparameters)",
            agent_instruction=textwrap.dedent("""\
                Create an anthill.train.TrainConfig (dataclass) with at minimum:
                  • epochs          — start with 20; adjust after first run
                  • batch_size      — 32 for CPU, 128-256 for GPU
                  • learning_rate   — 1e-3 with AdamW is a solid default
                  • weight_decay    — 1e-4
                  • lr_schedule     — 'cosine' (recommended) | 'step' | 'none'
                  • warmup_steps    — 5% of total steps
                  • grad_clip       — 1.0
                  • checkpoint_dir  — './checkpoints'
                  • device          — 'auto' (will select cuda > mps > cpu)
                  • log_every_n     — how often to print/log metrics (default: 50 steps)
                  • val_every_epoch — validate every N epochs (default: 1)
                  • early_stop_patience — stop if val loss doesn't improve for N epochs (default: 10)

                Print the config to the user before training starts.
            """),
        ),
        CheckItem(
            key="train.sanity_check",
            phase="train",
            label="Run pre-training sanity check before any full training run",
            agent_instruction=textwrap.dedent("""\
                Run anthill.train.sanity.pre_training_check(model, train_loader, val_loader, config).

                This check will:
                  1. Forward + backward pass on a single batch — confirm no shape errors or NaNs.
                  2. Confirm the loss is in the expected range for random weights (e.g. ~log(n_classes)
                     for cross-entropy with balanced classes).
                  3. Overfit check — train for 20 steps on a single batch; confirm loss goes to near-zero.
                     If it does not, there is a bug in the model or data pipeline.  Stop and fix it.
                  4. Data loading speed — time 10 batches; warn if > 100 ms/batch on CPU.
                  5. Display one batch of samples to the user (images shown as ASCII or saved to file;
                     text printed; numeric data printed as a table).

                DO NOT proceed to full training until all checks pass.
                A failed overfit check means either: wrong loss function, wrong label format,
                missing gradient somewhere, or the model cannot express the target.
            """),
        ),
        CheckItem(
            key="train.run",
            phase="train",
            label="Run the training loop",
            agent_instruction=textwrap.dedent("""\
                Use anthill.train.Trainer(model, train_loader, val_loader, config).fit(config.epochs).

                The Trainer provides at every log step:
                  • step, epoch, loss (train), loss (val, at val steps)
                  • gradient norm (to detect exploding/vanishing gradients)
                  • learning rate (to confirm schedule is working)
                  • samples/sec (to monitor throughput)
                  • ETA for current epoch and full run

                At the end of every validation epoch:
                  • Print the metrics dict from compute_metrics
                  • Print or display a small sample of model predictions vs. ground truth
                    (2-4 examples), so the user can qualitatively assess learning progress.
                  • Save a checkpoint if val loss improved.

                Intervene (pause and report to user) if:
                  • Training loss is not decreasing after 3 epochs.
                  • Validation loss diverges upward (> 20% above best) after epoch 5.
                  • Any NaN in loss or gradients.
                  • GPU/CPU memory is growing each epoch (likely a memory leak).
            """),
        ),
        CheckItem(
            key="train.evaluate",
            phase="train",
            label="Evaluate the best checkpoint on the held-out test set",
            agent_instruction=textwrap.dedent("""\
                Load the best checkpoint (lowest val loss) and run one pass over the test set.
                Report:
                  • All metrics from compute_metrics
                  • A confusion matrix (for classification tasks)
                  • 5 random correct predictions and 5 random incorrect predictions (for
                    classification), or 5 samples with highest and lowest error (for regression).

                Compare test metrics to val metrics.  If test metrics are significantly worse
                (> 5% gap on the primary metric), the model may be overfitting to val; note this
                and recommend collecting more diverse validation data.

                Report final metrics to the user in a clearly formatted table.
            """),
        ),
        CheckItem(
            key="train.iterate",
            phase="train",
            label="Decide whether to iterate on data, model, or call it done",
            agent_instruction=textwrap.dedent("""\
                Assess the results against the success criterion defined in data.task_definition.

                If the criterion is met: proceed to deploy phase.

                If not met, diagnose using this decision tree:
                  HIGH TRAIN LOSS, HIGH VAL LOSS (underfitting):
                    → Increase model capacity, train longer, reduce regularization.
                  LOW TRAIN LOSS, HIGH VAL LOSS (overfitting):
                    → Add dropout/weight_decay, collect more data, augment more aggressively.
                  HIGH VARIANCE IN PREDICTIONS (unstable):
                    → Lower learning rate, add gradient clipping, check for label noise.
                  GOOD AVERAGE METRICS BUT POOR ON SPECIFIC SUBGROUPS:
                    → Investigate which subgroups fail; add targeted data or loss weighting.

                Present diagnosis and recommended next step to the user before re-running.
            """),
        ),
    ]

    # ---------------------------------------------------------------- DEPLOY
    items += [
        CheckItem(
            key="deploy.target",
            phase="deploy",
            label="Define deployment target and inference requirements",
            agent_instruction=textwrap.dedent("""\
                Ask the user:
                  1. Where will the model run?
                     (Python script, REST API, mobile app, embedded system, browser, …)
                  2. What is the latency budget?  (< 10 ms, < 100 ms, batch-only, …)
                  3. What hardware is available at inference time?  (CPU only, GPU, edge device)
                  4. Will the model be called by other code?  If so, what does the caller expect
                     (numpy array, JSON, torch.Tensor, …)?

                Document answers.  They determine export format and wrapper design.
            """),
        ),
        CheckItem(
            key="deploy.export",
            phase="deploy",
            label="Export the model in an appropriate format",
            agent_instruction=textwrap.dedent("""\
                Choose the export format based on the deployment target:
                  • PYTHON ONLY (scripting, APIs):  torch.save state_dict + architecture code.
                    Wrap with anthill.deploy.ModelWrapper for a clean load() / predict() interface.
                  • CROSS-LANGUAGE / PRODUCTION:  export via torch.onnx.export().
                    Verify the ONNX graph with onnxruntime.InferenceSession on a test input.
                  • EMBEDDED / MOBILE:  torch.jit.script() or torch.jit.trace().
                    Confirm the TorchScript model produces identical output to the original.

                Always save alongside the model:
                  • Normalization statistics (mean, std, vocab, …)
                  • Model config / hyperparameters
                  • The anthill version used
                  • A short description of what the model does

                Use anthill.deploy.export(model, path, format='auto') for a sensible default.
            """),
        ),
        CheckItem(
            key="deploy.inference_wrapper",
            phase="deploy",
            label="Write an inference wrapper with the same interface as the deployment target",
            agent_instruction=textwrap.dedent("""\
                Write a Python class (or function) named Predictor that:
                  • Loads the exported model and normalization statistics from a directory path.
                  • Exposes a predict(raw_input) method accepting the same raw format the caller
                    will provide (e.g. a PIL Image, a raw string, a numpy array).
                  • Applies all preprocessing (resize, normalize, tokenize) internally.
                  • Returns output in the format the caller expects.
                  • Is stateless — safe to call from multiple threads.

                Include a __main__ block with a usage example so the user can test it immediately:
                  python -m <your_module>.predictor --input <path_or_string>
            """),
        ),
        CheckItem(
            key="deploy.tests",
            phase="deploy",
            label="Write smoke tests for the exported model and inference wrapper",
            agent_instruction=textwrap.dedent("""\
                Write at least three test cases:
                  1. ROUND-TRIP TEST — run a training sample through the Predictor and confirm
                     the output matches what the trained model produced before export.
                  2. EDGE CASE TEST — test one or two inputs that are unusual but valid
                     (e.g. all-black image, empty string, extreme numeric value).
                  3. LATENCY TEST — time 100 calls to predict(); confirm it meets the
                     latency budget defined in deploy.target.

                These tests should be runnable with `pytest` with no GPU required.
                If any test fails, fix the issue before handing off to the user.
            """),
        ),
        CheckItem(
            key="deploy.handoff",
            phase="deploy",
            label="Write a handoff document for the user",
            agent_instruction=textwrap.dedent("""\
                Produce a short (< 1 page) HANDOFF.md that covers:
                  1. What the model does and its measured performance on the test set.
                  2. How to load and run the Predictor (code snippet).
                  3. Known limitations and failure modes (based on the error analysis in train.evaluate).
                  4. How to retrain if new data becomes available (which script to run, which
                     config to change).
                  5. Any external dependencies or data sources that must be maintained.

                This is the document the user keeps after the agent session ends.
            """),
        ),
    ]

    return items


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class Workflow:
    """
    Tracks the state of a project through the four phases.

    Usage::

        wf = anthill.Workflow(task="Analogue clock reader")
        wf.print_checklist()

        # Mark items as you go
        wf["data.task_definition"].complete()
        wf["data.source_decision"].complete(notes="Using PGD")
        wf.print_checklist()

        # Get the agent instruction for the current item
        print(wf.current_item().agent_instruction)
    """

    def __init__(self, task: str = ""):
        self.task = task
        self.items: list[CheckItem] = build_checklist()
        self._index: dict[str, CheckItem] = {item.key: item for item in self.items}

    def __getitem__(self, key: str) -> CheckItem:
        if key not in self._index:
            raise KeyError(f"No checklist item with key {key!r}. Valid keys: {list(self._index)}")
        return self._index[key]

    def current_item(self) -> CheckItem | None:
        """Return the first non-done, non-skipped item."""
        for item in self.items:
            if item.status in (Status.PENDING, Status.IN_PROGRESS, Status.BLOCKED):
                return item
        return None

    def phase_items(self, phase: str) -> list[CheckItem]:
        return [i for i in self.items if i.phase == phase]

    def print_checklist(self) -> None:
        """Print the full checklist as a Rich table."""
        title = f"anthill workflow"
        if self.task:
            title += f" — {self.task}"
        console.print()
        console.rule(f"[bold cyan]{title}[/bold cyan]")

        for phase in PHASES:
            items = self.phase_items(phase)
            done = sum(1 for i in items if i.status == Status.DONE)
            skipped = sum(1 for i in items if i.status == Status.SKIPPED)
            completed = done + skipped

            table = Table(
                box=box.SIMPLE,
                show_header=False,
                padding=(0, 1),
                expand=True,
            )
            table.add_column("icon", width=4)
            table.add_column("key", style="dim", width=30)
            table.add_column("label")
            table.add_column("notes", style="dim italic")

            for item in items:
                color = _STATUS_COLORS[item.status]
                icon = _STATUS_ICONS[item.status]
                table.add_row(
                    f"[{color}]{icon}[/{color}]",
                    f"[dim]{item.key}[/dim]",
                    f"[{color}]{item.label}[/{color}]",
                    item.notes or "",
                )

            phase_color = "green" if completed == len(items) else "yellow" if completed > 0 else "dim"
            console.print(
                Panel(
                    table,
                    title=f"[bold {phase_color}]{PHASE_TITLES[phase]}[/bold {phase_color}]"
                          f"  [{phase_color}]{completed}/{len(items)}[/{phase_color}]",
                    subtitle=f"[dim]{PHASE_DESCRIPTIONS[phase]}[/dim]",
                    border_style=phase_color,
                    padding=(0, 1),
                )
            )

        console.print()

    def print_next_instruction(self) -> None:
        """Print the agent instruction for the current pending item."""
        item = self.current_item()
        if item is None:
            console.print("[bold green]All phases complete.[/bold green]")
            return
        item.start()
        console.print(
            Panel(
                f"[bold]{item.label}[/bold]\n\n{item.agent_instruction}",
                title=f"[cyan]{item.key}[/cyan]",
                subtitle=f"[dim]{PHASE_TITLES[item.phase]}[/dim]",
                border_style="cyan",
            )
        )

    def to_markdown(self) -> str:
        """Return the full checklist as a Markdown string (for LLM context injection)."""
        lines = [f"# anthill workflow checklist — {self.task}\n"]
        for phase in PHASES:
            lines.append(f"\n## {PHASE_TITLES[phase]}\n")
            lines.append(f"{PHASE_DESCRIPTIONS[phase]}\n")
            for item in self.phase_items(phase):
                icon = "- [x]" if item.status == Status.DONE else \
                       "- [-]" if item.status == Status.SKIPPED else "- [ ]"
                lines.append(f"{icon} **{item.key}** — {item.label}")
                if item.notes:
                    lines.append(f"  - _{item.notes}_")
        return "\n".join(lines)

    def summary(self) -> dict:
        total = len(self.items)
        done = sum(1 for i in self.items if i.status == Status.DONE)
        skipped = sum(1 for i in self.items if i.status == Status.SKIPPED)
        return {
            "task": self.task,
            "total": total,
            "done": done,
            "skipped": skipped,
            "remaining": total - done - skipped,
            "percent_complete": round(100 * (done + skipped) / total),
        }
