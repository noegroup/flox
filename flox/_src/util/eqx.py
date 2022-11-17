from typing import Type, TypeVar, cast

import equinox as eqx
from typing_extensions import dataclass_transform

T = TypeVar("T")
S = TypeVar("S")


def to_equinox_module(cls: Type[T]) -> Type[T]:
    return cast(
        Type[T],
        type(
            cls.__name__,
            (
                *cls.__bases__,
                eqx.Module,
            ),
            {n: v for n, v in cls.__dict__.items() if n != "__dict__"},
        ),
    )


@dataclass_transform()
def equinox_module(cls: Type[T]) -> Type[T]:
    return to_equinox_module(cls)


__all__ = ["equinox_module"]
