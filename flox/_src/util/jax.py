""" Helper functions for jax-related things. """

from collections import ChainMap
from collections.abc import Callable, Generator
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax_dataclasses import (
    pytree_dataclass,
)  # pyright: reportGeneralTypeIssues=false
from jaxtyping import Bool, Integer  # pyright: reportPrivateImportUsage=false

Condition = Bool[Array, "*dims"]
BranchIndex = Integer[Array, "*dims"]

__all__ = ["Switch", "key_chain", "FrozenMap", "op_repeat"]


def _select_branch(conds, branches) -> BranchIndex:
    """chains multiple jnp.where clauses and executes them in reverse order"""

    accum = jnp.zeros_like(conds[0]) - 1
    for (cond, val) in reversed(tuple(zip(conds, branches, strict=True))):
        accum = jnp.where(cond, val, accum)
    return accum


class Switch:
    """Implements a convenience decorator for handling switch-case control flow"""

    def __init__(self):
        self.conditions = []
        self.branches = []
        self.has_default = False

    def default_branch(self):
        if self.has_default:
            raise ValueError("Cannot add more than one default branch!")
        self.has_default = True
        return self.add_branch(jnp.array(True))

    def add_branch(
        self, condition: Condition
    ) -> Callable[[Callable], Callable]:
        def decorator(branch):
            self.conditions.append(condition)
            self.branches.append(branch)
            return branch

        return decorator

    def __call__(self, *operands):
        if len(self.branches) == 0:
            raise ValueError("Need at least one branch!")
        index = _select_branch(self.conditions, jnp.arange(len(self.branches)))
        # checkify.check(jnp.all(index >= 0), "No branch matches condition!")
        return jax.lax.switch(index, self.branches, *operands)


def key_chain(
    seed: int | jnp.ndarray | jax.random.PRNGKeyArray,
) -> Generator[jax.random.PRNGKeyArray, None, None]:
    """returns an iterator that automatically splits jax.random.PRNGKeys"""

    if isinstance(seed, int) or seed.ndim == 0:
        key = jax.random.PRNGKey(int(seed))
    else:
        key, _ = jax.random.split(seed)
    while True:
        new, key = jax.random.split(key)
        yield new


@pytree_dataclass(frozen=True)
class FrozenMap(dict[str, Any]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(
            self, "__dict__", dict(ChainMap(self.__dict__, kwargs))
        )

    def __repr__(self):
        inner = ", ".join(f"{key}={value}" for (key, value) in self.items())
        return f"FrozenMap({inner})"


T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")

Fn = Callable[[T], S]
Op = Callable[Concatenate[Fn, P], Fn]


def op_repeat(
    op: Op, nreps: int = 1, /, *args: P.args, **kwargs: P.kwargs
) -> Fn:
    @wraps(op)
    def wrapper(fn: Fn) -> Fn:
        for _ in range(nreps):
            fn = op(fn, *args, **kwargs)
        return fn

    return wrapper
