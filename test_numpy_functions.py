"""Function validation tests."""

import unittest

import numpy as np

from optfuncs import numpy_functions as npf
from optfuncs.utils import FunctionEvaluation


class TestNumpyFunctions(unittest.TestCase):
  batch_size = 2  # batch size of array in multiple input testing
  dtype = np.float64

  @classmethod
  def setUpClass(cls) -> None:
    cls.array = np.array(FunctionEvaluation.default_x_4d, cls.dtype)
    cls.batch = cls.array[None].repeat(cls.batch_size, axis=0)
    cls.zero = np.array(FunctionEvaluation.zeros_x_4d, cls.dtype)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zero
    del cls.batch

  def default_test(self, f: npf.NumpyFunction):
    array_result = np.array(FunctionEvaluation.default_fx_4d[f.name],
                            self.dtype)
    batch_result = np.array(array_result).repeat(self.batch_size)
    zero_result = np.array(FunctionEvaluation.zeros_fx_4d[f.name],
                           self.dtype)

    # Test default value [1,2,3,4]
    result = f(self.array)
    self.assertEqual(result, array_result)

    # Test batch of default value [[1,2,3,4],[1,2,3,4], ...]
    result = f(self.batch)
    self.assertTrue(np.array_equal(result, batch_result))
    self.assertEqual(result.shape, batch_result.shape)

    result = f(self.zero)
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
