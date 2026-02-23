"""Base PromptTemplate class."""
from __future__ import annotations
import textwrap


class PromptTemplate:
    """
    A string template with named placeholders.

    Parameters
    ----------
    template : str
        The prompt text.  Use {placeholder_name} for values to be filled in.
    name : str
        Short identifier, used in repr and error messages.
    required_keys : list[str] | None
        If given, .render() will raise if any of these keys are missing.
    """

    def __init__(self, template: str, name: str = "", required_keys: list[str] | None = None):
        self.template = textwrap.dedent(template).strip()
        self.name = name
        self.required_keys: list[str] = required_keys or []

    def render(self, **kwargs) -> str:
        missing = [k for k in self.required_keys if k not in kwargs]
        if missing:
            raise ValueError(
                f"PromptTemplate {self.name!r} is missing required keys: {missing}"
            )
        return self.template.format(**kwargs)

    def print(self, **kwargs) -> None:
        print(self.render(**kwargs))

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r})"
