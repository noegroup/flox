""" Everything related to training - this is still heavy WIP! """

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # type: ignore
from optax._src.base import GradientTransformation

from flox._src.flow.api import Transform
from flox._src.flow.potential import Potential, PullbackPotential
from flox._src.flow.sampling import PullbackSampler, PushforwardSampler, Sampler

T = TypeVar("T")
S = TypeVar("S")


OptState: TypeAlias = Any
Scalar = Float[Array, ""] | float

KeyArray = jax.Array | jax.random.PRNGKeyArray

__all__ = [
    "Criterion",
    "RunningMean",
    "update_step",
    "MaximumLikelihoodLoss",
    "FreeEnergyLoss",
    "mle_step",
    "free_energy_step",
]

Model = TypeVar("Model", eqx.Module, Transform, contravariant=True)


@runtime_checkable
class Criterion(Protocol[Model]):
    def __call__(self, key: KeyArray, model: Model) -> Scalar:
        ...


@dataclass(frozen=True)
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


Input = TypeVar("Input")
Output = TypeVar("Output")


def update_step(
    key: KeyArray,
    num_samples: int,
    model: Transform[T, S],
    opt_state: OptState,
    optim: GradientTransformation,
    criterion: Criterion[Transform[T, S]],
    aggregation: Callable[[Any], Scalar] = jnp.mean,
):
    def batch_criterion(key: KeyArray, model: Transform[T, S]) -> Scalar:
        return aggregation(
            jax.vmap(lambda key: criterion(key, model))(
                jax.random.split(key, num_samples)
            )
        )

    loss, grads = eqx.filter_value_and_grad(partial(batch_criterion, key))(
        model
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = cast(Transform[T, S], eqx.apply_updates(model, updates))
    return loss, model, opt_state


@dataclass(frozen=True)
class MaximumLikelihoodLoss(Generic[Input, Output]):

    base: Potential[Input]
    target: Sampler[Output]

    def __call__(
        self, key: KeyArray, model: Transform[Input, Output]
    ) -> Scalar:
        latent = PullbackSampler(self.target, model)(key)
        return self.base(latent.obj) - latent.ldj


@dataclass(frozen=True)
class VarGradLoss(Generic[Input, Output]):

    sampler: Sampler[Output]
    base: Potential[Input]
    target: Potential[Output]

    def __call__(
        self, key: KeyArray, model: Transform[Input, Output]
    ) -> Scalar:
        inp = jax.lax.stop_gradient(self.sampler(key)).obj

        u_model = PullbackPotential(self.base, model)(inp)
        u_target = self.target(inp)
        return u_target - u_model


@dataclass(frozen=True)
class FreeEnergyLoss(Generic[Input, Output]):

    target: Potential[Output]
    base: Sampler[Input]

    def __call__(
        self, key: KeyArray, model: Transform[Input, Output]
    ) -> Scalar:
        sample = PushforwardSampler(self.base, model)(key)
        return self.target(sample.obj) - sample.ldj


def mle_step(
    base: Potential[Input],
    optim: GradientTransformation,
    source: Sampler[Output],
    num_samples: int,
) -> Callable[
    [KeyArray, Transform[Input, Output], Any],
    tuple[Scalar, Transform[Input, Output], Any],
]:

    criterion = MaximumLikelihoodLoss(base, source)

    @eqx.filter_jit
    def jitted_step(
        key: KeyArray,
        model: Transform[Input, Output],
        opt_state: Any,
    ) -> tuple[Scalar, Transform[Input, Output], Any]:
        return update_step(key, num_samples, model, opt_state, optim, criterion)

    return jitted_step


def free_energy_step(
    target: Potential[Output],
    optim: GradientTransformation,
    base: Sampler[Input],
    num_samples: int,
) -> Callable[
    [KeyArray, Transform[Input, Output], Any],
    tuple[Scalar, Transform[Input, Output], Any],
]:

    criterion = FreeEnergyLoss(target, base)

    @eqx.filter_jit
    def jitted_step(
        key: KeyArray,
        model: Transform[Input, Output],
        opt_state: Any,
    ) -> tuple[Scalar, Transform[Input, Output], Any]:
        return update_step(key, num_samples, model, opt_state, optim, criterion)

    return jitted_step


def var_grad_step(
    sampler: Sampler[Output],
    base: Potential[Input],
    target: Potential[Output],
    optim: GradientTransformation,
    num_samples: int,
) -> Callable[
    [KeyArray, Transform[Input, Output], Any],
    tuple[Scalar, Transform[Input, Output], Any],
]:

    criterion = VarGradLoss(sampler, base, target)

    @eqx.filter_jit
    def jitted_step(
        key: KeyArray,
        model: Transform[Input, Output],
        opt_state: Any,
    ) -> tuple[Scalar, Any, Any]:
        return update_step(
            key,
            num_samples,
            model,
            opt_state,
            optim,
            criterion,
            jnp.var,
        )

    return jitted_step
