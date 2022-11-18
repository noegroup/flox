from collections.abc import Generator

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # type: ignore

from flox._src.geom.quaternion import Quaternion
from flox._src.util.jax import key_chain

KeyArray = jax.random.PRNGKeyArray | jnp.ndarray

__all__ = ["gen_dense", "FlipInvariant", "BoxWrapped"]


def gen_dense(
    units, activation, key: KeyArray
) -> Generator[eqx.Module, None, None]:
    num_layers = len(units) - 1
    for i, (inp, out) in enumerate(zip(units[:-1], units[1:])):
        yield eqx.nn.Linear(inp, out, use_bias=True, key=key)
        if i < num_layers - 1:
            if activation is not None:
                if isinstance(activation, eqx.Module):
                    yield activation
                else:
                    yield eqx.nn.Lambda(activation)


class FlipInvariant(eqx.Module):

    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __init__(self, units: int, activation=jax.nn.silu, *, key: KeyArray):
        chain = key_chain(key)
        self.encoder = eqx.nn.Sequential(
            tuple(gen_dense(units, activation, next(chain)))
        )
        self.decoder = eqx.nn.Sequential(
            tuple(gen_dense(units, activation, next(chain)))
        )

    def __call__(self, quat: Quaternion) -> Float[Array, "..."]:
        qpos = quat.reshape(-1)
        qneg = quat.reshape(-1)
        ypos = self.encoder(qpos)
        yneg = self.encoder(qneg)

        # make sure gradients flow
        y = jnp.stack([ypos, yneg], axis=0)
        weight = jax.nn.softmax(y, axis=0)

        z = (weight * y).sum(0)

        return self.decoder(z)


class BoxWrapped(eqx.Module):

    flatten: bool = True

    def __init__(self, box: Float[Array, "D"], pos: Float[Array, "D"]):
        feats = jnp.concatenate(
            [jnp.sin(pos / box * 2 * jnp.pi), jnp.cos(pos / box * 2 * jnp.pi)],
            axis=-1,
        )
        if self.flatten:
            feats = jnp.reshape(feats, -1)
        return feats
