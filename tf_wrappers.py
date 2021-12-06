import tensorflow as tf
from optfuncs import wrappers

class GrandientWrapper(wrappers.BaseGradientWrapper):
  """Function Wrapper that calculates gradients for each function call"""
  
  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = super().__call__(x)
    self._grads = tape.gradient(y, x)
    return y
  
  def call_with_grads(self, x: tf.Tensor) -> tf.Tensor:
    return self(), self._grads

  @property
  def gradients(self):
    return self._grads