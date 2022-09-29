from jaxtyping import Float  # pyright: reportPrivateImportUsage=false

from jax import Array  # pyright: reportGeneralTypeIssues=false

Scalar = Float[Array, ""]

Vector2 = Float[Array, "2"]
Vector3 = Float[Array, "3"]
VectorN = Float[Array, "N"]
VectorM = Float[Array, "M"]

EulerAngles = Float[Array, "3"]
Quaternion = Float[Array, "4"]

Matrix2x2 = Float[Array, "2 2"]
Matrix3x3 = Float[Array, "3 3"]
Matrix4x4 = Float[Array, "4 4"]
MatrixNxN = Float[Array, "N N"]
MatrixNxM = Float[Array, "N M"]

UnitVector2 = Vector2
UnitVectorN = VectorN

TangentBasisN = Float[Array, "N-1 N"]
