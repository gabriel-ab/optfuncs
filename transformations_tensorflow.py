import abc

import tensorflow as tf

from optfuncs import tensorflow_functions as tff


class TensorflowFunctionWrapper(tff.TensorflowFunction):
  def __init__(self, function: tff.TensorflowFunction):
    super(TensorflowFunctionWrapper, self).__init__(function.domain)
    self._src_fn: tff.TensorflowFunction = function

  @property
  def name(self):
    return f"{str(self)}: {self._src_fn.name}"

  @abc.abstractmethod
  def _call(self, x: tf.Tensor) -> tf.Tensor:
    pass


class VerticalShift(TensorflowFunctionWrapper):
  def __init__(self,
               function: tff.TensorflowFunction,
               shift: float):
    super(VerticalShift, self).__init__(function)
    self._shift = tf.constant(shift)

  def _call(self, x: tf.Tensor) -> tf.Tensor:
    return tf.add(self._src_fn(x), self._shift)


class HorizontalShift(TensorflowFunctionWrapper):
  def __init__(self,
               function: tff.TensorflowFunction,
               shift: float):
    super(HorizontalShift, self).__init__(function)
    self._shift = tf.constant(shift)

  def _call(self, x: tf.Tensor) -> tf.Tensor:
    return self._src_fn(tf.add(x, self._shift))


class UniformScaling(TensorflowFunctionWrapper):
  def __init__(self,
               function: tff.TensorflowFunction,
               inner_scale: float,
               outer_scale: float):
    super(UniformScaling, self).__init__(function)
    self._inner = tf.constant(inner_scale)
    self._outer = tf.constant(outer_scale)

  def _call(self, x: tf.Tensor) -> tf.Tensor:
    return tf.multiply(self._src_fn(tf.multiply(x, self._inner)), self._outer)
