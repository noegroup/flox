from collections import ChainMap
from collections.abc import Reversible
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    TypeVar,
    Protocol,
    Generic,
    cast,
    runtime_checkable,
)

import jax_dataclasses as jdc
from jaxtyping import Float, Array  #  type: ignore
import lenses
from lenses.ui.base import BaseUiLens

Volume = Float[Array, ""]

Input = TypeVar("Input")
Output = TypeVar("Output")
Context = TypeVar("Context")

T = TypeVar("T")


@dataclass
class Parameters(dict[str, Any]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "__dict__", dict(ChainMap(self.__dict__, kwargs)))

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise ValueError(f"Cannot change property `{__name}` of immutable Parameters.")

    def __setitem__(self, __key: str, __value: Any) -> None:
        raise ValueError(f"Cannot set key `{__key}`of immutable Parameters.")

    def __repr__(self):
        inner = ", ".join(f"{key}={value}" for (key, value) in self.items())
        return f"Parameters({inner})"


@jdc.pytree_dataclass
class Transformed(Generic[T]):
    payload: T
    logprob: Volume


@runtime_checkable
class Transform(Protocol[Input, Output]):
    def forward(self, input: Input) -> Transformed[Output]:
        ...

    def inverse(self, input: Output) -> Transformed[Input]:
        ...


def unpack_volume(lens: BaseUiLens = lenses.lens) -> BaseUiLens:
    def setter(state, update) -> Transformed[Any]:
        return Transformed(lens.set(update.payload)(state), update.logprob)

    return lenses.lens.Lens(lens.get(), setter)


@dataclass
class Coupling(Generic[Input, Output]):
    transform_cstr: Callable[..., Transform[Input, Output]]
    context: BaseUiLens[Any, Any, Parameters, Any]
    target: BaseUiLens[Any, Any, Any, Any]

    def forward(self, input: Input) -> Transformed[Output]:
        context = self.context.get()(input)
        bound = self.transform_cstr(**context)
        return unpack_volume(self.target).modify(bound.forward)(input)

    def inverse(self, input: Output) -> Transformed[Input]:
        context = self.context.get()(input)
        bound = self.transform_cstr(**context)
        return unpack_volume(self.target).modify(bound.inverse)(input)


@dataclass
class VolumeAccumulator(Generic[Input, Output]):
    transform: Transform[Input, Output]

    def forward(self, input: Transformed[Input]) -> Transformed[Output]:
        transformed = self.transform.forward(input.payload)
        return Transformed(transformed.payload, input.logprob + transformed.logprob)

    def inverse(self, input: Transformed[Output]) -> Transformed[Input]:
        transformed = self.transform.inverse(input.payload)
        return Transformed(transformed.payload, input.logprob + transformed.logprob)


@dataclass
class Sequential(Generic[Input, Output]):
    transforms: Reversible[Transform]

    def forward(self, input: Transformed[Input]) -> Transformed[Output]:
        return cast(
            Transformed[Output],
            reduce(
                lambda input, transform: transform.forward(input),
                self.transforms,
                input,
            ),
        )

    def inverse(self, input: Transformed[Output]) -> Transformed[Input]:
        return cast(
            Transformed[Input],
            reduce(
                lambda input, transform: transform.inverse(input),
                reversed(self.transforms),
                input,
            ),
        )
