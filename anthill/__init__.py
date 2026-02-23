"""
anthill
-------
An agentic framework for building custom deep learning models with your LLM.

anthill provides:
  - A structured four-phase workflow checklist (data → model → train → deploy)
  - Copy-paste-ready LLM prompt templates for each phase
  - A supervised training loop with rich feedback and pre-training sanity checks
  - Model export and inference wrapper utilities
  - A thin agent helper (run_step) to connect any LLM callable to the workflow

Quickstart::

    import anthill

    # Start a workflow for your task
    wf = anthill.Workflow(task="Analogue clock reader")
    wf.print_checklist()

    # Get the instruction for the next pending step (paste into your LLM)
    wf.print_next_instruction()

    # OR drive the workflow with an LLM agent callable:
    def my_agent(prompt: str) -> str:
        ...  # call Claude, Ollama, OpenAI, etc.

    response = anthill.run_step(wf, my_agent)
    print(response)
    wf["data.task_definition"].complete()

    # When you have a model and data loaders, use the Trainer:
    from anthill.train import Trainer, TrainConfig
    from anthill.train.sanity import pre_training_check

    config = TrainConfig(epochs=20, lr=1e-3, device="auto")
    pre_training_check(model, train_loader, val_loader, config, loss_fn=loss_fn)
    trainer = Trainer(model, train_loader, val_loader, config, loss_fn=loss_fn)
    trainer.fit()

    # Export the model
    from anthill.deploy import export
    export(model, "./my_model", format="state_dict", norm_stats=norm_stats)
"""

__version__ = "0.1.0"

from anthill.workflow import Workflow, CheckItem, Status, build_checklist
from anthill.train import Trainer, TrainConfig
from anthill.deploy import export, ModelWrapper
from anthill.agent import AgentFn, run_step
import anthill.prompts as prompts
import anthill.agent as agent

__all__ = [
    "Workflow",
    "CheckItem",
    "Status",
    "build_checklist",
    "Trainer",
    "TrainConfig",
    "export",
    "ModelWrapper",
    "AgentFn",
    "run_step",
    "agent",
    "prompts",
]
