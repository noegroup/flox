from collections.abc import Sequence, Callable, Generator

import equinox as eqx
import jax.numpy as jnp

from .jax_utils import key_chain

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def dense(
    key: int | jnp.ndarray,
    units: Sequence[int],
    activation: Activation | None = None,
    norm: bool = True,
) -> Generator[eqx.Module, None, None]:
    chain = key_chain(key)
    num_layers = len(units) - 1
    for i, (inp, out) in enumerate(zip(units[:-1], units[1:])):
        # yield eqx.nn.Bat
        yield eqx.nn.Linear(inp, out, key=next(chain))
        if activation is not None and i < num_layers - 1:
            if norm:
                yield eqx.nn.LayerNorm(shape=(out,))
            yield eqx.nn.Lambda(activation)


def residual(fn):
    def with_residual(x):
        return x + fn(x)

    return with_residual
