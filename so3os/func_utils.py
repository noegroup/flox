from collections.abc import Callable, Iterable
from functools import reduce
import inspect
from itertools import chain
from typing import Concatenate, TypeVar, Any, ParamSpec, Generic, cast

import equinox as eqx  # type: ignore

# UndefinedReturn = TypeVar("UndefinedReturn")
# UndefinedArgs = TypeVar("UndefinedArgs")


# R = TypeVar("R")
# A = TypeVar("A")
# B = TypeVar("B")


# class UnpackedArgs(eqx.Module, Generic[R]):

#     fn: Callable[..., R]

#     def __init__(self, fn: Callable[..., R]):
#         self.fn = fn

#     def __call__(self, arg: tuple) -> R:
#         return self.fn(*arg)


# def unpack_args(fn: Callable[..., R]) -> UnpackedArgs[R]:
#     """modifies call signture of `fn` from
#         fn: *args -> out
#     to
#         fn: args -> out"""
#     return UnpackedArgs(fn)


# class Composed(eqx.Module, Generic[A, B]):
#     f: Callable[[A], B]
#     g: Callable[..., A]

#     def __post_init__(self):
#         object.__setattr__(self, "__signature__", self._signature)

#     def __call__(self, *args, **kwargs) -> B:
#         return self.f(self.g(*args, **kwargs))

#     def __repr__(self) -> str:
#         return f"compose({', '.join(map(repr, self.flatten()))})"

#     @property
#     def _signature(self) -> inspect.Signature:
#         sig = inspect.signature(self)
#         fns = tuple(self.flatten())
#         try:
#             params = tuple(inspect.signature(fns[-1]).parameters.values())
#         except ValueError:
#             params = (
#                 inspect.Parameter(
#                     "unknown_args",
#                     kind=inspect._ParameterKind.VAR_POSITIONAL,
#                     annotation=UndefinedArgs,
#                 ),
#                 inspect.Parameter(
#                     "unknown_kwargs",
#                     kind=inspect._ParameterKind.VAR_KEYWORD,
#                     annotation=UndefinedArgs,
#                 ),
#             )
#         try:
#             return_annotation = inspect.signature(fns[0]).return_annotation
#         except ValueError:
#             return_annotation = UndefinedReturn
#         return sig.replace(parameters=params, return_annotation=return_annotation)

#     def flatten(self) -> Iterable[Callable]:
#         """flattens function graph into a iterator"""
#         fs = self.f.flatten() if isinstance(self.f, Composed) else [self.f]
#         gs = self.g.flatten() if isinstance(self.g, Composed) else [self.g]
#         return chain(fs, gs)

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
