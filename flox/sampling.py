from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

import jax
from jax_dataclasses import pytree_dataclass

from flox.flow_api import Transformed, Volume, VolumeAccumulator

T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class Sampler(Protocol[T]):

    def __call__(self, key: jax.random.PRNGKeyArray) -> Transformed[T]:
        ...


@pytree_dataclass(frozen=True)
class DatasetSampler(Generic[T]):
    size: int
    dataset: T

    def __call__(self, key: jax.random.PRNGKeyArray) -> Transformed[T]:
        sel = jax.random.randint(key, (), 0, self.size)
        sample: T = jax.tree_map(lambda x: x[sel], self.dataset)
        logprob: Volume = cast(Volume, 1. / self.size)
        return Transformed(sample, logprob)


@pytree_dataclass(frozen=True)
class PushforwardSampler(Generic[T, S]):
    base: Sampler[T]
    flow: VolumeAccumulator[T, S]

    def __call__(self, key: jax.random.PRNGKeyArray) -> Transformed[S]:
        return self.flow.forward(self.base(key))


@pytree_dataclass(frozen=True)
class PullbackSampler(Generic[T, S]):
    target: Sampler[S]
    flow: VolumeAccumulator[T, S]

    def __call__(self, key: jax.random.PRNGKeyArray) -> Transformed[T]:
        return self.flow.inverse(self.target(key))