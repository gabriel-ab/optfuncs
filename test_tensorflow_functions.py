"""Function validation tests."""

import unittest

import numpy
import tensorflow as tf

from optfuncs import tensorflow_functions as tff
from optfuncs.utils import FunctionEvaluation


class TestTensorflowFunctions(unittest.TestCase):
  batch_size = 10  # batch size of array in multiple input testing
  dtype = tf.float32

  @classmethod
  def setUpClass(cls) -> None:
    cls.array = tf.constant(FunctionEvaluation.default_x_4d, dtype=cls.dtype)
    cls.batch = tf.repeat(cls.array[None], cls.batch_size, 0)
    cls.zeros = tf.constant(FunctionEvaluation.zeros_x_4d, dtype=cls.dtype)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zeros
    del cls.batch

  # Get batch expected result from array expected result
  def batch_result(self, array_result: tf.Tensor) -> tf.Tensor:
    return tf.repeat(tf.expand_dims(array_result, 0), self.batch_size, 0)

  # Test a given function
  def full_test(self, f: tff.TensorflowFunction,
                relax_batch=False,
                tolerance: float = 10):
    # Eager mode.
    f.disable_tf_function()
    self.default_test(f, relax_batch=relax_batch, tolerance=tolerance)
    self.grads_test(f, tolerance=tolerance)

    # Graph mode.
    f.enable_tf_function()
    self.default_test(f, relax_batch, tolerance)
    self.grads_test(f, tolerance=tolerance)

  def default_test(self, f: tff.TensorflowFunction,
                   relax_batch=False,
                   tolerance: float = 10):
    array_result = tf.constant(FunctionEvaluation.default_fx_4d[f.name],
                               self.dtype)
    batch_result = self.batch_result(array_result)
    zero_result = tf.constant(FunctionEvaluation.zeros_fx_4d[f.name],
                              self.dtype)
    tol = tolerance * numpy.finfo(self.dtype.as_numpy_dtype).eps

    # Test default value [1,2,3,4]
    result = f(self.array)
    tf.debugging.assert_near(result, array_result, tol, tol)

    if not relax_batch:
      # Test batch of default value [[1,2,3,4],[1,2,3,4], ...]
      result = f(self.batch)
      tf.debugging.assert_near(result, batch_result)
      self.assertEqual(result.shape, batch_result.shape)

    result = f(self.zeros)
    tf.debugging.assert_near(result, zero_result, tol, tol)

    # Testing shape and dtype
    self.assertEqual(result.shape, array_result.shape)
    self.assertEqual(result.dtype, array_result.dtype)

  def grads_test(self,
                 f: tff.TensorflowFunction,
                 tolerance: float = 10):
    # TODO: Implement gradient testing.
    pass

  def test_ackley(self):
    self.full_test(tff.Ackley())

  def test_griewank(self):
    self.full_test(tff.Griewank())

  def test_rastrigin(self):
    self.full_test(tff.Rastrigin())

  def test_levy(self):
    self.full_test(tff.Levy())

  def test_rosenbrock(self):
    self.full_test(tff.Rosenbrock())

  def test_zakharov(self):
    self.full_test(tff.Zakharov())

  def test_sum_squares(self):
    self.full_test(tff.SumSquares())

  def test_sphere(self):
    self.full_test(tff.Sphere())

  def test_bent_cigar(self):
    self.full_test(tff.BentCigar(), True)

  def test_schumer_steiglitz(self):
    self.full_test(tff.SchumerSteiglitz(), True)

  def test_powell_sum(self):
    self.full_test(tff.PowellSum(), True)

  def test_alpine_2(self):
    self.full_test(tff.Alpine2(), True)

  def test_csendes(self):
    self.full_test(tff.Csendes(), True)

  def test_deb_1(self):
    self.full_test(tff.Deb1(), True)

  def test_deb_3(self):
    self.full_test(tff.Deb3(), True)

  def test_qing(self):
    self.full_test(tff.Qing(), True)

  def test_schwefel(self):
    self.full_test(tff.Schwefel(), True)

  def test_chung_reynolds(self):
    self.full_test(tff.ChungReynolds(), True)

  def test_schwefel_2_26(self):
    self.full_test(tff.Schwefel226(), True)

  def test_schwefel_2_22(self):
    self.full_test(tff.Schwefel222(), True)

  def test_schwefel_2_23(self):
    self.full_test(tff.Schwefel223(), True)

  def test_brown(self):
    self.full_test(tff.Brown(), True)

  def test_salomon(self):
    self.full_test(tff.Salomon(), True)

  def test_trigonometric_2(self):
    self.full_test(tff.Trigonometric2(), True)

  def test_weierstrass(self):
    self.full_test(tff.Weierstrass(), True, 1e3)

  def test_w_wavy(self):
    self.full_test(tff.WWavy(), True)

  def test_rotated_hyper_ellipsoid(self):
    self.full_test(tff.RotatedHyperEllipsoid())

  def test_dixon_price(self):
    self.full_test(tff.DixonPrice())


if __name__ == "__main__":
  unittest.main()
