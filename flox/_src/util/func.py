""" Functional utilities. """

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce, wraps
from typing import Any, Concatenate, Generic, ParamSpec, TypeVar, cast

P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

__all__ = ["compose2", "compose", "pipe2", "pipe", "Lens"]


def compose2(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def composed(x: A) -> C:
        return f(g(x))

    return composed


def compose(
    f: Callable[[Any], C], *fs: Callable[[Any], Any]
) -> Callable[[Any], C]:
    """composes functions"""
    return cast(Callable[[Any], C], reduce(compose2, fs, f))


def pipe2(
    f: Callable[Concatenate[A, P], B], g: Callable[[B], C]
) -> Callable[Concatenate[A, P], C]:
    def piped(x: A, *args: P.args, **kwargs: P.kwargs) -> C:
        return g(f(x, *args, **kwargs))

    return piped


def pipe(
    f: Callable[Concatenate[A, P], Any], *fs: Callable[[Any], Any]
) -> Callable[Concatenate[A, P], Any]:
    return cast(Callable[Concatenate[A, P], Any], reduce(pipe2, fs, f))


D = TypeVar("D")
X = TypeVar("X")
Y = TypeVar("Y")


@dataclass(frozen=True)
class Lens(Generic[A, B, C, D]):
    project: Callable[[A], C]
    inject: Callable[[A, D], B]

    def __call__(self, fn: Callable[[C], D]) -> Callable[[A], B]:
        def lifted(a: A) -> B:
            return self.inject(a, fn(self.project(a)))

        return lifted


T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")


def repeat(
    op: Callable[Concatenate[Callable[[T], S], P], Callable[[T], S]],
    nreps: int = 1,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[Callable[[T], S]], Callable[[T], S]]:
    @wraps(op)
    def wrapper(fn: Callable[[T], S]) -> Callable[[T], S]:
        for _ in range(nreps):
            fn = op(fn, *args, **kwargs)
        return fn

    return wrapper
