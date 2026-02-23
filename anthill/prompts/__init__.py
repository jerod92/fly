"""
anthill.prompts
---------------
Copy-paste-ready prompt templates for each phase of the workflow.

Each template is a PromptTemplate instance.  Call .render(**kwargs) to fill
in the placeholders and get a string you can paste into any LLM chat interface.

Example::

    from anthill.prompts import data as dp
    print(dp.pgd_text_prompt.render(
        task="classify the sentiment of movie reviews",
        input_description="a short movie review (1-4 sentences)",
        output_description="sentiment label: positive | negative | neutral",
        n_samples=200,
    ))
"""

from anthill.prompts._base import PromptTemplate

__all__ = ["PromptTemplate"]
