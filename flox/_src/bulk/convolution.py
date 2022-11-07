""" equivariant convolution for particle densities """

import math
from functools import lru_cache
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from flox._src.bulk.lattice import neighbors  # type: ignore


@lru_cache()
def gradient_kernel(
    num_dims,
):
    buffer = np.zeros((num_dims,) + (3,) * num_dims, dtype=np.float32)
    il = (slice(0, 1),) + (slice(1, 2),) * (num_dims - 1)
    ir = (slice(2, 3),) + (slice(1, 2),) * (num_dims - 1)
    for i in range(num_dims):
        buffer[i][il] += 1
        buffer[i][ir] += -1
        il = (il[-1],) + il[:-1]
        ir = (ir[-1],) + ir[:-1]
    return buffer[None]


def discrete_gradient(
    signal: jnp.ndarray,
):
    """discrete lattice gradient.

    Parameters:
    -----------
    signal: [B, *D]
        signal for which we compute the gradient

    Returns:
    --------
    jnp.ndarray [B, *D, D]
        lattice gradient of signal
    """
    ndims = len(signal.shape) - 1
    signal = signal[..., None]
    kernel = -1.0 * gradient_kernel(ndims)
    dimstr = "".join(map(str, range(ndims)))
    dimension_numbers = jax.lax.conv_dimension_numbers(
        signal.shape,
        kernel.shape,
        (f"N{dimstr}C", f"IO{dimstr}", f"N{dimstr}C"),
    )
    padded = jnp.pad(
        signal,
        pad_width=(((0, 0),) + ((1, 1),) * ndims + ((0, 0),)),
        mode="wrap",
    )
    return jax.lax.conv_general_dilated(
        padded,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
    )


def grid(resolution: tuple[int, ...]):
    """returns grid for filter"""
    xi = (jnp.linspace(-1, 1, r) for r in resolution)
    grid = jnp.meshgrid(*xi, indexing="ij")
    return cast(Array, jnp.stack(grid, axis=-1))


def gauss_kernel(
    resolution: tuple[int, ...], gain: float, spectral_kernel: bool = True
) -> jnp.ndarray:
    """gaussian smoothing filter.

    Parameters:
    -----------
    resolution: tuple[int, ...]
        resolution of the signal / lattice
    gain: float
        filter gain

    Returns:
    --------
    jnp.ndarray [*D]
        Gaussian kernel
    """
    if not spectral_kernel:
        gain = math.prod(resolution) / (0.5 * gain)
    gauss = jnp.exp(
        -0.5 * gain * jnp.sum(jnp.square(grid(resolution)), axis=-1)
    )
    gauss = gauss.reshape(*resolution)
    if spectral_kernel:
        gauss = jnp.roll(
            gauss,
            shift=tuple(r // 2 for r in resolution),
            axis=range(len(resolution)),
        )
    gauss = gauss * jax.lax.rsqrt(jnp.square(gauss).sum() + 1e-12)
    return gauss


def gauss_conv(
    signal: jnp.ndarray,
    resolution: tuple[int, ...],
    gain: float,
    spectral_kernel: bool = True,
    conservative: bool = True,
) -> jnp.ndarray:
    """gaussian smoothing

    Parameters:
    -----------
    signal: jnp.ndarray [*D, *F]
        signal on a *D - lattice
    resolution: tuple[int, ...]
        resolution of the signal / lattice
    gain: float
        filter gain

    Return:
    -------
    jnp.ndarray [*D, *F]
        signal after smoothing
    """
    return spectral_convolve(
        signal,
        gauss_kernel(resolution, gain, spectral_kernel),
        spectral_kernel=spectral_kernel,
        conservative=conservative,
    )


def spectral_convolve(
    inp: jnp.ndarray,
    kernel: jnp.ndarray,
    spectral_kernel: bool = True,
    conservative: bool = True,
):
    """spectral convolution using the FFT.

    Parameters:
    -----------
    inp: jnp.ndarray [*D, *F]
        signal on a *D - lattice
    kernel: jnp.ndarray [*S, *F] / [*S, 1]
        filter with *S spectral coeffs

    Return:
    -------
    jnp.ndarray [*D, *F]
        signal after spectral convolution with filter
    """
    ndims = len(kernel.shape)
    axes = tuple(range(ndims))
    spectrum = jnp.fft.fftn(inp, s=kernel.shape, axes=axes)
    if not spectral_kernel:
        kernel = jnp.abs(jnp.fft.fftn(kernel, axes=axes))

    kslices = (slice(None),) * ndims + (None,) * (len(inp.shape) - ndims)
    out = jnp.fft.ifftn(
        spectrum * kernel[kslices],
        s=inp.shape[:ndims],
        axes=axes,
    )
    out = cast(Array, out).real

    if conservative:
        inp_norm = jnp.sqrt(1e-12 + jnp.square(inp).sum(axes, keepdims=True))
        out_norm = jax.lax.rsqrt(
            1e-12 + jnp.square(out).sum(axes, keepdims=True)
        )
        out = out * out_norm * inp_norm

    return out


def convolution_edges(dim):
    edges = neighbors(dim)
    edges = edges / jnp.sqrt(1e-12 + jnp.square(edges).sum(-1, keepdims=True))
    return edges


# def lattice_gauge_conv(
#     signal: jnp.ndarray,
#     idxs: jnp.ndarray,
# ):


def gauge_conv(
    focus: Float[Array, "F D"],
    other: Float[Array, "N F D"],
    edges: Float[Array, "N D"],
    kernel: Float[Array, "N F G"],
    order=True,
    handle_tie_break=True,
) -> Float[Array, "G D"]:
    """Gauge equivariant lattice convolution.

    F/G: features, N: neighbors, D: spatial dimensions

    Parameters:
    -----------
    focus: jnp.ndarray [F, D]
        focused cell
    other: jnp.ndarray [N, F, D]
        neighboring cells
    edges: jnp.ndarray [N, D]
        edges from focus to neighbors
    kernel: jnp.ndarray [N, F, G]
        kernel of convolution

    Return:
    -------
    jnp.ndarray [G, D]
        convolved focus
    """

    inner = jnp.einsum("fi, ni -> nf", focus, edges)
    if handle_tie_break:
        _, F, _ = other.shape
        inner_order = inner[None]
        coord_order = jnp.tile(jnp.transpose(edges)[:, :, None], (1, 1, F))
        keys = jnp.concatenate([coord_order, inner_order], axis=0)
        idxs = jnp.lexsort(keys, axis=0)
    else:
        idxs = jnp.argsort(inner, axis=0)
    if order:
        n, f, d = jnp.ogrid[map(slice, other.shape)]
        other = other[idxs[n, f], f, d]
        n, f = jnp.ogrid[map(slice, inner.shape)]
        inner = inner[idxs, f]
    inner = (inner + 1) / 9
    return jnp.einsum("nf, nfd, nfg -> gd", inner, other, kernel)
