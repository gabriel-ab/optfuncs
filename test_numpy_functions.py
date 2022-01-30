"""Function validation tests."""

import unittest

import numpy as np

from optfuncs import numpy_functions as npf
from optfuncs.utils import FunctionEvaluationExamples


class TestNumpyFunctions(unittest.TestCase):
  batch_size = 2  # batch size of array in multiple input testing
  dtype = np.float64
  dims = 4

  def default_test(self, f: npf.NumpyFunction):
    array, array_result = FunctionEvaluationExamples.get_default_eval(
      f, self.dims)
    array = np.array(array, self.dtype)
    array_result = np.array(array_result, self.dtype)

    # Test default value [1,2,3,4]
    result = f(array)
    self.assertEqual(result, array_result)

    batch = array[None].repeat(self.batch_size, axis=0)
    batch_result = np.array(array_result).repeat(self.batch_size)

    # Test batch of default value [[1,2,3,4],[1,2,3,4], ...]
    result = f(batch)
    self.assertTrue(np.array_equal(result, batch_result))
    self.assertEqual(result.shape, batch_result.shape)

    zero, zero_result = FunctionEvaluationExamples.get_eval_at_zeros(
      f, self.dims)
    zero = np.array(zero, dtype=self.dtype)
    zero_result = np.array(zero_result, dtype=self.dtype)

    result = f(zero)
    self.assertEqual(result, zero_result)

    # Testing shape and dtype
    self.assertEqual(result.shape, zero_result.shape)
    self.assertEqual(result.dtype, zero_result.dtype)

  def test_ackley(self):
    self.default_test(npf.Ackley())

  def test_griewank(self):
    self.default_test(npf.Griewank())

  def test_rastrigin(self):
    self.default_test(npf.Rastrigin())

  def test_levy(self):
    self.default_test(npf.Levy())

  def test_rosenbrock(self):
    self.default_test(npf.Rosenbrock())

  def test_zakharov(self):
    self.default_test(npf.Zakharov())

  def test_sum_squares(self):
    self.default_test(npf.SumSquares())

  def test_sphere(self):
    self.default_test(npf.Sphere())

  def test_rotated_hyper_ellipsoid(self):
    self.default_test(npf.RotatedHyperEllipsoid())

  def test_dixon_price(self):
    self.default_test(npf.DixonPrice())


if __name__ == "__main__":
  unittest.main()
