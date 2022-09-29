from collections.abc import Callable, Iterable
from functools import reduce
import inspect
from itertools import chain
from typing import TypeVar, Any, ParamSpec, Generic

import equinox as eqx  # type: ignore

UndefinedReturn = TypeVar("UndefinedReturn")
UndefinedArgs = TypeVar("UndefinedArgs")


P = ParamSpec("P")
R = TypeVar("R")


class unpacked_args(eqx.Module, Generic[R]):

    fn: Callable

    def __init__(self, fn: Callable[P, R]):
        self.fn = fn

    def __call__(self, arg: tuple) -> R:
        return self.fn(*arg)


def unpack_args(fn: Callable[P, R]) -> Callable[[tuple], R]:
    """modifies call signture of `fn` from
        fn: *args -> out
    to
        fn: args -> out"""
    return unpacked_args(fn)


class composed(eqx.Module):
    f: Callable
    g: Callable

    def __post_init__(self):
        object.__setattr__(self, "__signature__", self._signature)

    def __call__(self, x: Any) -> Any:
        return self.f(self.g(x))

    def __repr__(self):
        return f"compose({', '.join(map(repr, self.flatten()))})"

    @property
    def _signature(self):
        sig = inspect.signature(self)
        fns = tuple(self.flatten())
        try:
            params = tuple(inspect.signature(fns[-1]).parameters.values())
        except ValueError:
            params = (
                inspect.Parameter(
                    "unknown_args",
                    kind=inspect._ParameterKind.VAR_POSITIONAL,
                    annotation=UndefinedArgs,
                ),
                inspect.Parameter(
                    "unknown_kwargs",
                    kind=inspect._ParameterKind.VAR_KEYWORD,
                    annotation=UndefinedArgs,
                ),
            )
        try:
            return_annotation = inspect.signature(fns[0]).return_annotation
        except ValueError:
            return_annotation = UndefinedReturn
        return sig.replace(parameters=params, return_annotation=return_annotation)

    def flatten(self) -> Iterable[Callable]:
        """flattens function graph into a iterator"""
        fs = self.f.flatten() if isinstance(self.f, composed) else [self.f]
        gs = self.g.flatten() if isinstance(self.g, composed) else [self.g]
        return chain(fs, gs)


def compose(*fs):
    """composes functions"""
    return reduce(composed, fs)
