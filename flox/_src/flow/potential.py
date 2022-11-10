from typing import Generic, Protocol, TypeVar, runtime_checkable

from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float

from flox._src.flow.api import Transform, bind  # type: ignore

__all__ = ["Potential", "PullbackPotential", "PushforwardPotential"]

Scalar = Float[Array, ""] | float

T = TypeVar("T", contravariant=True)
S = TypeVar("S")


@runtime_checkable
class Potential(Protocol[T]):
    def __call__(self, inp: T) -> Scalar:
        ...


Input = TypeVar("Input")
Output = TypeVar("Output")


@pytree_dataclass(frozen=True)
class PullbackPotential(Generic[Input, Output], Potential[Output]):
    base: Potential[Input]
    flow: Transform[Input, Output]

    def __call__(self, inp: Output) -> Scalar:
        out = self.flow.inverse(inp)
        return self.base(out.obj) - out.ldj


@pytree_dataclass(frozen=True)
class PushforwardPotential(Generic[Input, Output], Potential[Input]):
    base: Potential[Output]
    flow: Transform[Input, Output]

    def __call__(self, inp: Input) -> Scalar:
        out = self.flow.forward(inp)
        return self.base(out.obj) - out.ldj
