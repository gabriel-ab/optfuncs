"""TensorFlow's implementation of many functions.
References:
  [1] Momin Jamil and Xin-She Yang.
      A Literature Survey of Benchmark Functions For Global
        Optimization Problems, 2013. (10.1504/IJMMNO.2013.055204)
  [2] IEEE CEC 2021 C-2 (https://cec2021.mini.pw.edu.pl/en/program/competitions)
  [3] IEEE CEC 2021 C-3 (https://cec2021.mini.pw.edu.pl/en/program/competitions)
  [4] https://www.sfu.ca/~ssurjano/optimization.html
"""

import abc
import typing
from math import e, pi

import tensorflow as tf

from optfuncs import core
from optfuncs import numpy_functions as npf


class TensorflowFunction(core.Function):
  def __init__(self, domain: core.Domain):
    super(TensorflowFunction, self).__init__(domain)
    self._tf_function = False

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    return self._fn(x)

  @abc.abstractmethod
  def _call(self, x: tf.Tensor) -> tf.Tensor:
    pass

  def enable_tf_function(self):
    self._fn = tf.function(self._fn)
    self._tf_function = True

  def disable_tf_function(self):
    self._fn = self._call
    self._tf_function = False

  def executing_eagerly(self):
    return not self._tf_function

  def grads(self, x: tf.Tensor) -> tf.Tensor:
    grads, _ = self.grads_at(x)
    return grads

  def grads_at(self, x: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = self(x)

    return tape.gradient(y, x), y


class Ackley(TensorflowFunction):
  """Ackley function 1 defined in [1]."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-35.0, max=35.0),
               a=20,
               b=0.2, c=2 * pi):
    super(Ackley, self).__init__(domain)
    self.a = a
    self.b = b
    self.c = c

  def _call(self, x: tf.Tensor):
    d = tf.constant(x.shape[-1], x.dtype)
    sum1 = tf.reduce_sum(tf.math.pow(x, 2), axis=-1)
    sum2 = tf.reduce_sum(tf.cos(tf.math.multiply(x, self.c)), axis=-1)
    term1 = tf.math.multiply(tf.exp(
      tf.math.multiply(tf.sqrt(tf.math.divide(sum1, d)), -self.b)), -self.a)
    term2 = tf.exp(tf.math.divide(sum2, d))
    result = term1 - term2 + self.a + e
    return result


class Alpine2(TensorflowFunction):
  """Alpine function 2 defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=0.0, max=10.0)):
    super(Alpine2, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.reduce_prod(tf.multiply(tf.sqrt(x), tf.sin(x)), axis=-1)


class BentCigar(TensorflowFunction):
  """BentCigar function defined in [2].
  Implementation doesn't support batch yet.
  """

  def __init__(self, domain: core.Domain = core.Domain(min=0.0, max=10.0)):
    super(BentCigar, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.pow(x[0], 2) + tf.multiply(tf.reduce_sum(tf.pow(x[1:], 2),
                                                       axis=-1), 1e6)


class Bohachevsky(TensorflowFunction):
  """Bohachevsky function 1 defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super(Bohachevsky, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    tf.assert_equal(d, 2)

    return tf.pow(x[0], 2) + tf.math.multiply(tf.pow(x[1], 2), 2) - \
           tf.math.multiply(tf.cos(3 * pi * x[0]), 0.3) - \
           tf.math.multiply(tf.cos(4 * pi * x[1]), 0.4) + 0.7


class ChungReynolds(TensorflowFunction):
  """Chung Reynolds function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super(ChungReynolds, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), 2)


class Csendes(TensorflowFunction):
  """Csendes function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-1.0, max=1.0)):
    super(Csendes, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.cond(tf.equal(tf.reduce_prod(x), 0),
                   lambda: tf.constant(0, dtype=x.dtype),
                   lambda: tf.reduce_sum(tf.multiply(tf.pow(x, 6), 2 +
                                                     tf.sin(tf.divide(1, x))),
                                         axis=-1))


class Deb1(TensorflowFunction):
  """Deb function 1 defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-1.0, max=1.0)):
    super(Deb1, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return -tf.divide(tf.reduce_sum(tf.pow(tf.sin(tf.multiply(x, 5 * pi)), 6),
                                    axis=-1), d)


class Deb3(TensorflowFunction):
  """Deb function 3 defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=0.0, max=1.0)):
    super(Deb3, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return -tf.divide(
      tf.reduce_sum(
        tf.pow(
          tf.sin(
            tf.multiply(tf.pow(x, 3 / 4) - 0.05, 5 * pi)), 6),
        axis=-1), d)


class DixonPrice(TensorflowFunction):
  """Dixon-Price function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(DixonPrice, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    d = tf.shape(x)[-1]
    x0 = x[:, 0]
    ii = tf.range(2.0, d + 1, dtype=x.dtype)
    xi = x[:, 1:]
    xold = x[:, :-1]
    dixon_sum = ii * tf.pow(2 * tf.pow(xi, 2) - xold, 2)
    result = tf.pow(x0 - 1, 2) + tf.reduce_sum(dixon_sum, -1)
    return tf.squeeze(result)


class Griewank(TensorflowFunction):
  """Griewank function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super(Griewank, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    shape = tf.shape(x)
    griewank_sum = tf.divide(tf.reduce_sum(tf.math.pow(x, 2), axis=-1), 4000)
    den = tf.range(1, shape[-1] + 1, dtype=x.dtype)
    den = tf.repeat(tf.expand_dims(den, 0), shape[0], axis=0)
    prod = tf.cos(tf.math.divide(x, tf.sqrt(den)))
    prod = tf.reduce_prod(prod, axis=-1)
    return tf.squeeze(griewank_sum - prod + 1)


class Levy(TensorflowFunction):
  """Levy function defined in [4]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(Levy, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    d = tf.shape(x)[-1] - 1
    w = 1 + tf.math.divide(tf.math.subtract(x, 1), 4)

    term1 = tf.math.pow(tf.sin(pi * w[:, 0]), 2)
    wd = w[:, d]
    term3 = tf.math.pow(wd - 1, 2) * (1 + tf.math.pow(tf.sin(2 * pi * wd), 2))
    wi = w[:, 0:d]
    levy_sum = tf.reduce_sum(tf.math.pow((wi - 1), 2) *
                             (1 + 10 * tf.math.pow(tf.sin(pi * wi + 1), 2)),
                             axis=-1)
    return tf.squeeze(term1 + levy_sum + term3)


class PowellSum(TensorflowFunction):
  """Powell Sum function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-1.0, max=1.0)):
    super(PowellSum, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    indices = tf.range(start=1, limit=d + 1, dtype=x.dtype)
    return tf.reduce_sum(tf.pow(tf.math.abs(x), indices + 1))


class Qing(TensorflowFunction):
  """Qing function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-500.0, max=500.0)):
    super(Qing, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    indices = tf.range(start=1, limit=d + 1, dtype=x.dtype)
    return tf.reduce_sum(tf.pow(tf.pow(x, 2) - indices, 2), axis=-1)


class Rastrigin(TensorflowFunction):
  """Rastrigin function defined in [2]. Search range may vary."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super(Rastrigin, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return (10 * d) + tf.reduce_sum(tf.math.pow(x, 2) -
                                    (10 * tf.cos(tf.math.multiply(x, 2 * pi))),
                                    axis=-1)


class Rosenbrock(TensorflowFunction):
  """Rosenbrock function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-30.0, max=30.0)):
    super(Rosenbrock, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    xi = x[:, :-1]
    xnext = x[:, 1:]
    result = tf.reduce_sum(100 * tf.math.pow(xnext - tf.math.pow(xi, 2), 2) +
                           tf.math.pow(xi - 1, 2), axis=-1)
    return tf.squeeze(result)


class RotatedHyperEllipsoid(TensorflowFunction):
  """Rotated Hyper-Ellipsoid function defined in [4]."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-65.536, max=65.536)):
    super(RotatedHyperEllipsoid, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    d = tf.shape(x)[-1]
    mat = tf.repeat(tf.expand_dims(x, 1), d, 1)
    matlow = tf.linalg.band_part(mat, -1, 0)
    inner = tf.reduce_sum(matlow ** 2, -1)
    result = tf.reduce_sum(inner, -1)
    return tf.squeeze(result)


class SumSquares(TensorflowFunction):
  """Sum Squares function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(SumSquares, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    mul = tf.range(1, x.shape[-1] + 1, dtype=x.dtype)
    return tf.reduce_sum(tf.math.multiply(tf.math.pow(x, 2), mul), axis=-1)


class Schwefel(TensorflowFunction):
  """Schwefel function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0),
               a: float = pi):
    super(Schwefel, self).__init__(domain)
    self._a = a

  def _call(self, x: tf.Tensor):
    a = tf.cast(self._a, dtype=x.dtype)
    return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), a)


class Schwefel226(TensorflowFunction):
  """Schwefel 2.26 function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-500.0, max=500.0)):
    super(Schwefel226, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return -tf.divide(
      tf.reduce_sum(
        tf.multiply(x, tf.sin(tf.sqrt(tf.abs(x)))), axis=-1), d)


class SchumerSteiglitz(TensorflowFunction):
  """Schumer Steiglitz function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(SchumerSteiglitz, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(tf.pow(x, 4), axis=-1)


class Sphere(TensorflowFunction):
  """Sphere function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=0.0, max=10.0)):
    super(Sphere, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(tf.math.pow(x, 2), axis=-1)


class Weierstrass(TensorflowFunction):
  """Weierstrass function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-0.5, max=0.5),
               a: float = 0.5,
               b: float = 3,
               kmax: int = 20):
    super(Weierstrass, self).__init__(domain)
    self._a = a
    self._b = b
    self._kmax = kmax

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    a = tf.cast(self._a, dtype=x.dtype)
    b = tf.cast(self._b, dtype=x.dtype)

    kindices = tf.range(start=0, limit=self._kmax + 1, dtype=x.dtype)
    ak = tf.pow(a, kindices)
    bk = tf.pow(b, kindices)

    ak_bk_sum = d * tf.reduce_sum(tf.multiply(ak, tf.cos(tf.multiply(bk, pi))),
                                  axis=-1)

    def fn(acc, xi):
      s = tf.reduce_sum(
        tf.multiply(ak, tf.cos(tf.multiply(2 * pi * bk, xi + 0.5))),
        axis=-1)
      return acc + (s - ak_bk_sum)

    return tf.foldl(fn, x, initializer=tf.cast(0, dtype=x.dtype))


class WWavy(TensorflowFunction):
  """W / Wavy function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-pi, max=pi),
               k: float = 10):
    super(WWavy, self).__init__(domain)
    self._k = k

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return 1 - tf.divide(
      tf.reduce_sum(
        tf.multiply(tf.cos(tf.multiply(x, self._k)),
                    tf.exp(tf.divide(-tf.pow(x, 2), 2))),
        axis=-1), d)


class Zakharov(TensorflowFunction):
  """Zakharov function defined in [1]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super(Zakharov, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    sum1 = tf.reduce_sum(tf.math.pow(x, 2), axis=-1)
    sum2 = tf.reduce_sum(tf.math.divide(
      tf.math.multiply(x, tf.range(1, (d + 1), dtype=x.dtype)), 2), axis=-1)
    return sum1 + tf.math.pow(sum2, 2) + tf.math.pow(sum2, 4)


def atleast_2d(tensor: tf.Tensor) -> tf.Tensor:
  """Make sure a tensor is a matrix."""
  return tf.cond(tf.less(tf.size(tf.shape(tensor)), 2),
                 lambda: tf.expand_dims(tensor, 0),
                 lambda: tensor)


def list_all_functions() -> typing.List[core.Function]:
  """Deprecated. Manually collect all functions."""
  return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
          Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(),
          DixonPrice()]


def get_grads(fun: TensorflowFunction, pos: tf.Tensor):
  """Deprecated. fun.grads_at(pos) instead. """
  if pos.dtype != tf.float32:
    pos = tf.cast(pos, tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(pos)
    y = fun(pos)

  return tape.gradient(y, pos), y


def get_np_function(function: core.Function):
  """Deprecated. Manually convert functions."""
  domain = function.domain
  f = getattr(npf, function.name)
  return f(domain)
