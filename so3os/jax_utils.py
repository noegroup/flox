from typing import Callable

import jax
import jax.numpy as jnp

# from jax.experimental import checkify
from jaxtyping import Array, Bool, Integer


Condition = Bool[Array, "*dims"]
BranchIndex = Integer[Array, "*dims"]


def _select_branch(conds, branches) -> BranchIndex:
    """ chains multiple jnp.where clauses and executes them in reverse order """

    accum = jnp.zeros_like(conds[0]) - 1
    for (cond, val) in reversed(tuple(zip(conds, branches, strict=True))):
        accum = jnp.where(cond, val, accum)
    return accum


class Switch:
    """ Implements a convenience decorator for handling switch-case control flow"""

    def __init__(self):
        self.conditions = []
        self.branches = []
        self.has_default = False

    def default_branch(self):
        if self.has_default:
            raise ValueError("Cannot add more than one default branch!")
        self.has_default = True
        return self.add_branch(jnp.array(True))

    def add_branch(self, condition: Condition) -> Callable[[Callable], Callable]:
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


def key_chain(seed: int | jnp.ndarray):
    """ returns an iterator that automatically splits jax.random.PRNGKeys """

    if isinstance(seed, int) or seed.ndim == 0:
        key = jax.random.PRNGKey(seed)
    else:
        key, _ = jax.random.split(seed)
    while True:
        new, key = jax.random.split(key)
        yield new
