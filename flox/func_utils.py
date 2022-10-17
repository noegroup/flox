from collections.abc import Callable
from functools import reduce
from typing import Concatenate, TypeVar, Any, ParamSpec, cast


P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def compose2(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def composed(x: A) -> C:
        return f(g(x))

    return composed


def compose(f: Callable[[Any], C], *fs: Callable[[Any], Any]) -> Callable[[Any], C]:
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
