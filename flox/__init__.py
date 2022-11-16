import logging
import os

use_strict_typing = os.getenv("FLOX_STRICT_TYPING", "False").lower() in (
    "true",
    "1",
    "t",
)

from jaxtyping import install_import_hook  # type: ignore

if use_strict_typing:
    logging.warning("using flox with strict typing")

    with install_import_hook(
        ["flox"],
        ("typeguard", "typechecked"),
    ):
        from . import bulk, flow, geom, nn, util
else:
    from . import bulk, flow, geom, nn, util

__all__ = ["bulk", "flow", "geom", "nn", "util"]
