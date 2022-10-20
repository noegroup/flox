from typing import Callable, TypeVar

from .geometry import Scalar

T = TypeVar("T")

Potential = Callable[[T], Scalar]
