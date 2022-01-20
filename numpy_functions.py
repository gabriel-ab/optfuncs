"""NumPy's implementation of many functions."""

import abc
import typing

import numpy as np

from optfuncs import core
from optfuncs import tensorflow_functions as tff


class NumpyFunction(core.Function):
  def __call__(self, x: np.ndarray) -> np.ndarray:
    return self._fn(x)

  def grads(self, x: np.ndarray):
    raise NotImplementedError("Gradients for NumPy functions are "
                              "not implemented yet.")
    pass

  def grads_at(self, x: np.ndarray):
    raise NotImplementedError("Gradients for NumPy functions are "
                              "not implemented yet.")
    pass

  @abc.abstractmethod
  def _call(self, x: np.ndarray) -> np.ndarray:
    pass


class Ackley(NumpyFunction):
  """Ackley function as defined in:
  https://www.sfu.ca/~ssurjano/ackley.html."""

  def __init__(self, domain=core.Domain(min=-32.768, max=32.768),
               a=20, b=0.2, c=2 * np.math.pi, dtype=np.float32):
    super().__init__(domain)
    self.a = a
    self.b = b
    self.c = c
    self.dtype = dtype

  def _call(self, x: np.ndarray):
    d = x.shape[-1]
    sum1 = np.sum(x * x, axis=-1)
    sum2 = np.sum(np.cos(self.c * x), axis=-1)
    term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
    term2 = np.exp(sum2 / d)
    result = term1 - term2 + self.a + np.math.e

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class Griewank(NumpyFunction):
  """Griewank function as defined in:
  https://www.sfu.ca/~ssurjano/griewank.html."""

  def __init__(self, domain=core.Domain(min=-600.0, max=600.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    x = np.atleast_2d(x)
    griewank_sum = np.sum(x ** 2, axis=-1) / 4000.0
    den = np.arange(1, x.shape[-1] + 1,
                    dtype=x.dtype)[None].repeat(x.shape[0], axis=0)
    prod = np.cos(x / np.sqrt(den))
    prod = np.prod(prod, axis=-1)
    result = griewank_sum - prod + 1

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return np.squeeze(result)


class Rastrigin(NumpyFunction):
  """Rastrigin function as defined in:
  https://www.sfu.ca/~ssurjano/rastr.html."""

  def __init__(self, domain=core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    d = x.shape[-1]
    result = 10 * d + np.sum(x ** 2 - 10 * np.cos(x * 2 * np.math.pi), axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class Levy(NumpyFunction):
  """Levy function as defined in:
  https://www.sfu.ca/~ssurjano/levy.html."""

  def __init__(self, domain=core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    x = np.atleast_2d(x)
    pi = np.math.pi
    d = x.shape[-1] - 1
    w = 1 + (x - 1) / 4

    term1 = np.sin(pi * w[:, 0]) ** 2
    wd = w[:, d]
    term3 = (wd - 1) ** 2 * (1 + np.sin(2 * pi * wd) ** 2)
    wi = w[:, 0:d]
    levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2),
                      axis=-1)
    result = term1 + levy_sum + term3

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return np.squeeze(result)


class Rosenbrock(NumpyFunction):
  """Rosenbrock function as defined in:
  https://www.sfu.ca/~ssurjano/rosen.html."""

  def __init__(self, domain=core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    x = np.atleast_2d(x)
    xi = x[:, :-1]
    xnext = x[:, 1:]
    result = np.sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return np.squeeze(result)


class Zakharov(NumpyFunction):
  """Zakharov function as defined in:
  https://www.sfu.ca/~ssurjano/zakharov.html."""

  def __init__(self, domain=core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    d = x.shape[-1]

    sum1 = np.sum(x * x, axis=-1)
    sum2 = np.sum(x * np.arange(start=1, stop=(d + 1), dtype=x.dtype) / 2,
                  axis=-1)
    result = sum1 + sum2 ** 2 + sum2 ** 4
    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class Bohachevsky(NumpyFunction):
  """Bohachevsky function (f1, 2 dims only) as defined in:
  https://www.sfu.ca/~ssurjano/boha.html."""

  def __init__(self, domain=core.Domain(min=-100.0, max=100.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    d = x.shape[-1]
    assert d == 2

    result = np.power(x[0], 2) + 2 * np.power(x[1], 2) - \
             0.3 * np.cos(3 * np.pi * x[0]) - \
             0.4 * np.cos(4 * np.pi * x[1]) + 0.7

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class SumSquares(NumpyFunction):
  """SumSquares function as defined in:
  https://www.sfu.ca/~ssurjano/sumsqu.html."""

  def __init__(self, domain=core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    d = x.shape[-1]
    mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
    result = np.sum((x ** 2) * mul, axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class Sphere(NumpyFunction):
  """Sphere function as defined in:
  https://www.sfu.ca/~ssurjano/spheref.html."""

  def __init__(self, domain=core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    result = np.sum(x * x, axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return result


class RotatedHyperEllipsoid(NumpyFunction):
  """Rotated Hyper-Ellipsoid function as defined in:
  https://www.sfu.ca/~ssurjano/rothyp.html."""

  def __init__(self, domain=core.Domain(min=-65.536, max=65.536)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    x = np.atleast_2d(x)
    mat = x[:, None].repeat(x.shape[-1], axis=1)
    matlow = np.tril(mat)
    inner = np.sum(matlow ** 2, axis=-1)
    result = np.sum(inner, axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return np.squeeze(result)


class DixonPrice(NumpyFunction):
  """Dixon-Price function as defined in:
  https://www.sfu.ca/~ssurjano/dixonpr.html."""

  def __init__(self, domain=core.Domain(-10, 10)):
    super().__init__(domain)

  def _call(self, x: np.ndarray):
    x = np.atleast_2d(x)
    x0 = x[:, 0]
    d = x.shape[-1]
    ii = np.arange(2.0, d + 1)
    xi = x[:, 1:]
    xold = x[:, :-1]
    dixon_sum = ii * (2 * xi ** 2 - xold) ** 2
    result = (x0 - 1) ** 2 + np.sum(dixon_sum, axis=-1)

    if result.dtype != x.dtype:
      result = result.astype(x.dtype)
    return np.squeeze(result)


@DeprecationWarning
def get_tf_function(function: NumpyFunction):
  domain = function.domain
  f = getattr(tff, function.name)
  return f(domain)


@DeprecationWarning
def list_all_functions() -> typing.List[NumpyFunction]:
  return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
          Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(),
          DixonPrice()]
