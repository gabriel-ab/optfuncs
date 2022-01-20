"""TensorFlow implementation of many different functions."""

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
  """Ackley function as defined in:
  https://www.sfu.ca/~ssurjano/ackley.html."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-32.768, max=32.768),
               a=20,
               b=0.2, c=2 * pi):
    super(Ackley, self).__init__(domain)
    self.a = a
    self.b = b
    self.c = c

  def _call(self, x: tf.Tensor):
    d = tf.constant(x.shape[-1], x.dtype)
    sum1 = tf.reduce_sum(x * x, axis=-1)
    sum2 = tf.reduce_sum(tf.cos(tf.math.multiply(x, self.c)), axis=-1)
    term1 = -self.a * tf.exp(-self.b * tf.sqrt(sum1 / d))
    term2 = tf.exp(sum2 / d)
    result = term1 - term2 + self.a + e
    return result


class Griewank(TensorflowFunction):
  """Griewank function as defined in:
  https://www.sfu.ca/~ssurjano/griewank.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-600.0, max=600.0)):
    super(Griewank, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    shape = tf.shape(x)
    griewank_sum = tf.divide(tf.reduce_sum(tf.math.pow(x, 2), axis=-1), 4000)
    den = tf.range(1, shape[-1] + 1, dtype=x.dtype)
    den = tf.repeat(tf.expand_dims(den, 0), shape[0], axis=0)
    prod = tf.cos(x / tf.sqrt(den))
    prod = tf.reduce_prod(prod, axis=-1)
    return tf.squeeze(griewank_sum - prod + 1)


class Rastrigin(TensorflowFunction):
  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super(Rastrigin, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return 10 * d + tf.reduce_sum(x ** 2 - 10 *
                                  tf.cos(x * 2 * pi), axis=-1)


class Levy(TensorflowFunction):
  """Levy function as defined in:
  https://www.sfu.ca/~ssurjano/levy.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(Levy, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    d = tf.shape(x)[-1] - 1
    w = 1 + (x - 1) / 4

    term1 = tf.sin(pi * w[:, 0]) ** 2
    wd = w[:, d]
    term3 = (wd - 1) ** 2 * (1 + tf.sin(2 * pi * wd) ** 2)
    wi = w[:, 0:d]
    levy_sum = tf.reduce_sum((wi - 1) ** 2 *
                             (1 + 10 * tf.sin(pi * wi + 1) ** 2), axis=-1)
    return tf.squeeze(term1 + levy_sum + term3)


class Rosenbrock(TensorflowFunction):
  """Rosenbrock function as defined in:
  https://www.sfu.ca/~ssurjano/rosen.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super(Rosenbrock, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    xi = x[:, :-1]
    xnext = x[:, 1:]
    result = tf.reduce_sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2,
                           axis=-1)
    return tf.squeeze(result)


class Zakharov(TensorflowFunction):
  """Zakharov function as defined in:
  https://www.sfu.ca/~ssurjano/zakharov.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super(Zakharov, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    sum1 = tf.reduce_sum(x * x, axis=-1)
    sum2 = tf.reduce_sum(x * tf.range(1, (d + 1), dtype=x.dtype) / 2, axis=-1)
    return sum1 + sum2 ** 2 + sum2 ** 4


class Bohachevsky(TensorflowFunction):
  """Bohachevsky function (f1, 2 dims only) as defined in:
  https://www.sfu.ca/~ssurjano/boha.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-100.0, max=100.0)):
    super(Bohachevsky, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    tf.assert_equal(d, 2)

    return tf.pow(x[0], 2) + tf.math.multiply(2, tf.pow(x[1], 2)) - \
           tf.math.multiply(0.3, tf.cos(3 * pi * x[0])) - \
           tf.math.multiply(0.4, tf.cos(4 * pi * x[1])) + 0.7


class SumSquares(TensorflowFunction):
  """SumSquares function as defined in:
  https://www.sfu.ca/~ssurjano/sumsqu.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super(SumSquares, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    mul = tf.range(1, x.shape[-1] + 1, dtype=x.dtype)
    return tf.reduce_sum((x ** 2) * mul, axis=-1)


class Sphere(TensorflowFunction):
  """Sphere function as defined in:
  https://www.sfu.ca/~ssurjano/spheref.html."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super(Sphere, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(x * x, axis=-1)


class RotatedHyperEllipsoid(TensorflowFunction):
  """Rotated Hyper-Ellipsoid function as defined in:
  https://www.sfu.ca/~ssurjano/rothyp.html."""

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


class DixonPrice(TensorflowFunction):
  """Dixon-Price function as defined in:
  https://www.sfu.ca/~ssurjano/dixonpr.html."""

  def __init__(self, domain: core.Domain = core.Domain(-10, 10)):
    super(DixonPrice, self).__init__(domain)

  def _call(self, x: tf.Tensor):
    x = atleast_2d(x)
    d = tf.shape(x)[-1]
    x0 = x[:, 0]
    ii = tf.range(2.0, d + 1, dtype=x.dtype)
    xi = x[:, 1:]
    xold = x[:, :-1]
    dixon_sum = ii * (2 * xi ** 2 - xold) ** 2
    result = (x0 - 1) ** 2 + tf.reduce_sum(dixon_sum, -1)
    return tf.squeeze(result)


def atleast_2d(tensor: tf.Tensor) -> tf.Tensor:
  """Make sure a tensor is a matrix."""
  return tf.cond(tf.less(tf.size(tf.shape(tensor)), 2),
                 lambda: tf.expand_dims(tensor, 0),
                 lambda: tensor)


@DeprecationWarning
def list_all_functions() -> typing.List[core.Function]:
  return [Ackley(), Griewank(), Rastrigin(), Levy(), Rosenbrock(), Zakharov(),
          Bohachevsky(), SumSquares(), Sphere(), RotatedHyperEllipsoid(),
          DixonPrice()]


@DeprecationWarning
def get_grads(fun: TensorflowFunction, pos: tf.Tensor):
  """Deprecated. fun.grads_at(pos) instead. """
  if pos.dtype != tf.float32:
    pos = tf.cast(pos, tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(pos)
    y = fun(pos)

  return tape.gradient(y, pos), y


# Função utilitária para obter uma função equivalente de TensorFlow em Numpy.
@DeprecationWarning
def get_np_function(function: core.Function):
  domain = function.domain
  f = getattr(npf, function.name)
  return f(domain)
