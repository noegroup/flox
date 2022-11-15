""" Everything related to sampling. Still WIP! """

from typing import Generic, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass

from flox._src.flow.api import Inverted, Transform, Transformed, Volume, bind

T = TypeVar("T")
S = TypeVar("S")

KeyArray = jax.Array | jax.random.PRNGKeyArray

__all__ = [
    "Sampler",
    "DatasetSampler",
    "PushforwardSampler",
    "PullbackSampler",
]


@runtime_checkable
class Sampler(Protocol[T]):
    def __call__(self, key: KeyArray) -> Transformed[T]:
        ...


@pytree_dataclass(frozen=True)
class DatasetSampler(Generic[T]):
    data: T
    len: int
    ldj: Volume = jnp.zeros(())

    def __call__(self, key: KeyArray) -> Transformed[T]:
        idxs = jax.random.choice(key, self.len)
        sliced: T = jax.tree_map(lambda a: a[idxs], self.data)
        return Transformed(sliced, self.ldj)


@pytree_dataclass(frozen=True)
class PushforwardSampler(Generic[T, S]):
    base: Sampler[T]
    flow: Transform[T, S]

    def __call__(self, key: KeyArray) -> Transformed[S]:
        return bind(self.base(key), self.flow)


@pytree_dataclass(frozen=True)
class PullbackSampler(Generic[T, S]):
    target: Sampler[S]
    flow: Transform[T, S]

    def __call__(self, key: KeyArray) -> Transformed[T]:
        return bind(self.target(key), Inverted(self.flow))
