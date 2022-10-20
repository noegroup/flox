from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp
import optax
from jax_dataclasses import pytree_dataclass
from optax._src.base import GradientTransformation, OptState, Params

from flox.flow_api import Transformed
from flox.geometry import Scalar
from flox.haiku import MultiTransformedFlow
from flox.potential import Potential

T = TypeVar("T")
S = TypeVar("S")

@pytree_dataclass
class RunningMean:
    value: float
    count: int

    def update(self, new) -> "RunningMean":
        return RunningMean(
            (self.value * self.count + new) / (self.count + 1),
            self.count + 1,
        )


def kl_divergence(
    params: dict,
    samples: Transformed[T],
    flow: MultiTransformedFlow[T, S],
    target: Potential[S],
) -> Scalar:
    latent = flow.with_params(params).forward(samples)
    return target(latent.payload) - latent.logprob


def kl_loss(
    params: dict,
    samples: Transformed[T],
    flow: MultiTransformedFlow[T, S],
    target: Potential[S],
 ) -> Scalar:
    return jnp.mean(
        jax.vmap(
            kl_divergence,
            in_axes=(None, 0, None, None)
        )(
            params, samples, flow, target
        )
    )


@partial(jax.jit, static_argnames=("optim", "flow", "target"))
def update_step(
    params: Params,
    opt_state: OptState,
    optim: GradientTransformation,
    samples: Transformed[T],
    flow: MultiTransformedFlow[T, S],
    target: Potential[S],
):
    loss, grads = jax.value_and_grad(kl_loss, argnums=0)(
        params, samples, flow, target)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, opt_state, params