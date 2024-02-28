from functools import partialmethod
from re import sub

import yaml


def get_yaml(fpath) -> dict:
    """Load the given .yaml file as a Python dictionary."""
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def to_snake_case(string: str) -> str:
    """Convert the given string to snake case."""
    return "_".join(
        sub("([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", string.replace("-", " "))).split()
    ).lower()


def partialclass(cls, *args, **kwargs):
    """Return a class constructor, partially-initialized with the given args and kwargs."""

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls
