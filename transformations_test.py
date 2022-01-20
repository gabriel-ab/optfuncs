"""Transformations validation tests."""

import unittest

import numpy as np
import tensorflow as tf

from optfuncs import core
from optfuncs import numpy_functions as npf
from optfuncs import tensorflow_functions as tff
from optfuncs import transformations_numpy as t_npf
from optfuncs import transformations_tensorflow as t_tff


class DummyNumpyFunction(npf.NumpyFunction):
  def __init__(self, domain: core.Domain = core.Domain(-100.0, 100.0)):
    super().__init__(domain)

  def _call(self, x: np.ndarray) -> np.ndarray:
    return np.sum(x)


class TestNumpyTransformations(unittest.TestCase):
  batch_size = 10  # batch size of array in multiple input testing
  dtype = np.float32

  def test_vshift(self):
    fn = DummyNumpyFunction()
    tfn = t_npf.VerticalShift(fn, shift=1.0)
    arr = np.array([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 4.0)

  def test_hshift(self):
    fn = DummyNumpyFunction()
    tfn = t_npf.HorizontalShift(fn, shift=1.0)
    arr = np.array([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.0)

  def test_scaling(self):
    fn = DummyNumpyFunction()
    tfn = t_npf.UniformScaling(fn, inner_scale=2.0, outer_scale=0.5)
    arr = np.array([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 3.0)

  def test_composition(self):
    fn = DummyNumpyFunction()
    tfn = t_npf.UniformScaling(fn, inner_scale=2.0, outer_scale=1.0)
    tfn = t_npf.VerticalShift(tfn, shift=1.0)
    tfn = t_npf.HorizontalShift(tfn, shift=1.0)
    tfn = t_npf.UniformScaling(tfn, inner_scale=1.0, outer_scale=0.5)
    arr = np.array([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.5)

  def test_batched(self):
    # TODO
    pass


class DummyTensorflowFunction(tff.TensorflowFunction):
  def __init__(self, domain: core.Domain = core.Domain(-100.0, 100.0)):
    super().__init__(domain)

  def _call(self, x: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(x)


class TestTensorflowTransformations(unittest.TestCase):
  batch_size = 10  # batch size of array in multiple input testing
  dtype = tf.float32

  def test_vshift(self):
    fn = DummyTensorflowFunction()
    tfn = t_tff.VerticalShift(fn, shift=1.0)
    arr = tf.constant([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 4.0)

  def test_hshift(self):
    fn = DummyTensorflowFunction()
    tfn = t_tff.HorizontalShift(fn, shift=1.0)
    arr = tf.constant([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.0)

    fn.enable_tf_function()
    tfn.enable_tf_function()
    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.0)

  def test_scaling(self):
    fn = DummyTensorflowFunction()
    tfn = t_tff.UniformScaling(fn, inner_scale=2.0, outer_scale=0.5)
    arr = tf.constant([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 3.0)

    fn.enable_tf_function()
    tfn.enable_tf_function()
    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 3.0)

  def test_composition(self):
    fn = DummyTensorflowFunction()
    tfn = t_tff.UniformScaling(fn, inner_scale=2.0, outer_scale=1.0)
    tfn = t_tff.VerticalShift(tfn, shift=1.0)
    tfn = t_tff.HorizontalShift(tfn, shift=1.0)
    tfn = t_tff.UniformScaling(tfn, inner_scale=1.0, outer_scale=0.5)
    arr = tf.constant([1.0, 1.0, 1.0])

    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.5)

    fn.enable_tf_function()
    tfn.enable_tf_function()
    self.assertEqual(fn(arr), 3.0)
    self.assertEqual(tfn(arr), 6.5)

  def test_batched(self):
    # TODO
    pass


if __name__ == "__main__":
  unittest.main()
