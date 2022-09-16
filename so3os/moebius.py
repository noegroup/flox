import jax.numpy as jnp
from .geometry import UnitVector2, UnitVectorN, VectorN, Scalar, inner, norm, proj


def moebius_project(p: UnitVectorN, q: VectorN) -> UnitVectorN:
    """ projects p along q back onto the hypersphere
        this is a n-dimensional moebius transform
    """
    return p - 2 * proj(p, q - p)


def double_moebius_project(p: UnitVectorN, q: VectorN) -> UnitVectorN:
    """ the double moebius transform that satisfies anti-podal symmetry """
    r = moebius_project(p, q) + moebius_project(p, -q)
    r = r / norm(r)
    return r


def invert_on_great_circle(
    p: UnitVector2, r: Scalar, eps: float = 1e-12, threshold: float = 0.9
) -> UnitVector2:
    """ invert the double projection on the great circle """
    x, y = p
    r2 = r ** 2

    sign = jnp.array([-1.0, 1.0])
    numer = (sign * r2 - 1.0) * p
    denom = jnp.sqrt((r2 + sign) ** 2 - 4.0 * sign * (r * p) ** 2)

    x_, y_ = numer / (denom + eps)

    y_ = jnp.where(
        r > threshold, -jnp.sign(y) * jnp.sqrt(jnp.clip(1.0 - x_ ** 2, 0.0, 1.0)), y_
    )  # pim's numerical stability modification

    return jnp.stack([x_, y_])


def double_moebius_inverse(p: UnitVectorN, q: VectorN) -> UnitVectorN:
    """ inverts the double projection for general vectors """
    r = norm(q)

    dx = q / norm(q)  # fst basis: q
    ux = inner(p, dx)

    dy = p - ux * dx  # snd basis: p - proj(p, q)
    uy = norm(dy)
    dy = dy / uy

    x, y = invert_on_great_circle(
        jnp.stack([ux, uy]), r
    )  # solve problem on great circle

    return x * dx + y * dy  # project back on hyper-sphere


def double_moebius_volume_change(p, q):
    qq = inner(q, q)
    qp = inner(q, p)
    dim = len(p)
    numer = -(qq - 1.0) * (qq + 1.0) ** (dim - 1)
    denom = ((qq + 1.0) ** 2 - (2.0 * qp) ** 2) ** (dim / 2)
    return numer / denom


def moebius_volume_change(p, q):
    dim = len(p)
    numer = -(inner(q, q) - 1)
    denom = inner(q - p, q - p)
    return (numer / denom) ** (dim / 2)
