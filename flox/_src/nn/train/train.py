""" Everything related to training - this is still heavy WIP! """

from typing import Any, Callable, Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore
from optax import OptState
from optax._src.base import GradientTransformation, Params

from flox._src.flow.api import UnboundFlow
from flox._src.flow.potential import Potential, PullbackPotential
from flox._src.flow.sampling import PullbackSampler, PushforwardSampler, Sampler

T = TypeVar("T")
S = TypeVar("S")

Scalar = Float[Array, ""] | float


__all__ = [
    "Criterion",
    "RunningMean",
    "update_step",
    "MaximumLikelihoodLoss",
    "FreeEnergyLoss",
    "mle_step",
    "free_energy_step",
]


class Criterion(Protocol):
    def __call__(self, key: jax.random.PRNGKeyArray, params: Params) -> Scalar:
        ...


@pytree_dataclass
class RunningMean:
    value: float
    count: int
    discount: float = 1.0

    def update(self, new) -> "RunningMean":

        return RunningMean(
            (self.value * self.count * self.discount + new)
            / (self.count * self.discount + 1),
            self.count + 1,
        )


def update_step(
    key: jax.random.PRNGKeyArray,
    num_samples: int,
    params: Params,
    opt_state: OptState,
    optim: GradientTransformation,
    criterion: Criterion,
    aggregation: Callable[[Any], Scalar] = jnp.mean,
) -> tuple[Scalar, Params, OptState]:
    def batch_criterion(key: jax.random.PRNGKeyArray, params: Params) -> Scalar:
        return aggregation(
            jax.vmap(lambda key: criterion(key, params))(
                jax.random.split(key, num_samples)
            )
        )

    loss, grads = jax.value_and_grad(batch_criterion, argnums=1)(key, params)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


Input = TypeVar("Input")
Output = TypeVar("Output")


@pytree_dataclass(frozen=True)
class MaximumLikelihoodLoss(Generic[Input, Output]):

    base: Potential[Input]
    flow: UnboundFlow[Input, Output]
    target: Sampler[Output]

    def __call__(self, key: jax.random.PRNGKeyArray, params: Params) -> Scalar:
        latent = PullbackSampler(self.target, self.flow.with_params(params))(
            key
        )
        return self.base(latent.obj) - latent.ldj


@pytree_dataclass(frozen=True)
class VarGradLoss(Generic[Input, Output]):

    sampler: Sampler[Output]
    base: Potential[Input]
    flow: UnboundFlow[Input, Output]
    target: Potential[Output]

    def __call__(self, key: jax.random.PRNGKeyArray, params: Params) -> Scalar:
        inp = jax.lax.stop_gradient(self.sampler(key)).obj

        u_model = PullbackPotential(self.base, self.flow.with_params(params))(
            inp
        )
        u_target = self.target(inp)
        return u_target - u_model


@pytree_dataclass(frozen=True)
class FreeEnergyLoss(Generic[Input, Output]):

    target: Potential[Output]
    flow: UnboundFlow[Input, Output]
    base: Sampler[Input]

    def __call__(self, key: jax.random.PRNGKeyArray, params: Params) -> Scalar:
        sample = PushforwardSampler(self.base, self.flow.with_params(params))(
            key
        )
        return self.target(sample.obj) - sample.ldj


Input = TypeVar("Input")
Output = TypeVar("Output")


def mle_step(
    base: Potential[Input],
    flow: UnboundFlow[Input, Output],
    optim: GradientTransformation,
    source: Sampler[Output],
    num_samples: int,
):

    criterion = MaximumLikelihoodLoss(base, flow, source)

    @jax.jit
    def jitted_step(
        key: jax.random.PRNGKeyArray,
        params: Params,
        opt_state: OptState,
    ) -> tuple[Scalar, OptState, Params]:
        return update_step(
            key, num_samples, params, opt_state, optim, criterion
        )

    return jitted_step


def free_energy_step(
    target: Potential[Output],
    flow: UnboundFlow[Input, Output],
    optim: GradientTransformation,
    base: Sampler[Input],
    num_samples: int,
):

    criterion = FreeEnergyLoss(target, flow, base)

    @jax.jit
    def jitted_step(
        key: jax.random.PRNGKeyArray,
        params: Params,
        opt_state: OptState,
    ) -> tuple[Scalar, OptState, Params]:
        return update_step(
            key, num_samples, params, opt_state, optim, criterion
        )

    return jitted_step


def var_grad_step(
    sampler: Sampler[Output],
    base: Potential[Input],
    flow: UnboundFlow[Input, Output],
    target: Potential[Output],
    optim: GradientTransformation,
    num_samples: int,
):

    criterion = VarGradLoss(sampler, base, flow, target)

    @jax.jit
    def jitted_step(
        key: jax.random.PRNGKeyArray,
        params: Params,
        opt_state: OptState,
    ) -> tuple[Scalar, OptState, Params]:
        return update_step(
            key,
            num_samples,
            params,
            opt_state,
            optim,
            criterion,
            aggregation=jnp.var,
        )

    return jitted_step
