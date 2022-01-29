"""Function validation tests."""

import unittest

import numpy
import numpy as np
import tensorflow as tf

from optfuncs import numpy_functions as npf
from optfuncs import tensorflow_functions as tff

# TODO: Check how to run R implementation in this test
# TODO: Add gradient test

array = [1, 2, 3, 4]
array_lookup = {
  "Ackley": 8.43469444443746497,
  "Griewank": 1.00187037800320189,
  "Rastrigin": 30.0,
  "Levy": 2.76397190019909811,
  "Rosenbrock": 2705.0,
  "Zakharov": 50880.0,
  "SumSquares": 100.0,
  "Sphere": 30.0,
  "BentCigar": 29000001.0,
  "PowellSum": 1114.0,
  "Alpine2": -0.40033344730936005,
  "Csendes": 11063.416256526398,
  "Deb1": -6.182844847431069e-87,
  "Deb3": -0.036599504738713866,
  "Qing": 184.0,
  "Schwefel": 43703.20448793846,
  "ChungReynolds": 900.0,
  "Schwefel222": 34.0,
  "Schwefel223": 1108650.0,
  "Schwefel226": -2.353818129766789,
  "Weierstrass": 23.999988555908203,
  "WWavy": 1.1130512151573806,
  "Brown": 1.6678281e+16,
  "SchumerSteiglitz": 354.0,
  "RotatedHyperEllipsoid": 50.0,
  "DixonPrice": 4230.0,
}

zeros = [0, 0, 0, 0]
zero_lookup = {
  "Ackley": 4.44089209850062616e-16,
  "Griewank": 0.0,
  "Rastrigin": 0.0,
  "Levy": 0.897533662350923467,
  "Rosenbrock": 3.0,
  "Zakharov": 0.0,
  "SumSquares": 0.0,
  "Sphere": 0.0,
  "BentCigar": 0.0,
  "PowellSum": 0.0,
  "Alpine2": 0.0,
  "Csendes": 0.0,
  "Deb1": 0.0,
  "Deb3": -0.1249999850988388,
  "Qing": 30.0,
  "Schwefel": 0.0,
  "ChungReynolds": 0.0,
  "Schwefel222": 0.0,
  "Schwefel223": 0.0,
  "Schwefel226": 0.0,
  "Weierstrass": 23.999988555908203,
  "WWavy": 0.0,
  "Brown": 0.0,
  "SchumerSteiglitz": 0.0,
  "RotatedHyperEllipsoid": 0.0,
  "DixonPrice": 1.0,
}


class TestNumpyFunctions(unittest.TestCase):
  batch_size = 2  # batch size of array in multiple input testing
  dtype = np.float64

  @classmethod
  def setUpClass(cls) -> None:
    cls.array = np.array(array, cls.dtype)
    cls.batch = cls.array[None].repeat(cls.batch_size, axis=0)
    cls.zero = np.array(zeros, cls.dtype)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zero
    del cls.batch

  def default_test(self, f: npf.NumpyFunction):
    array_result = np.array(array_lookup[f.name], self.dtype)
    batch_result = np.array(array_result).repeat(self.batch_size)
    zero_result = np.array(zero_lookup[f.name], self.dtype)

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


class TestTensorflowFunctions(unittest.TestCase):
  batch_size = 10  # batch size of array in multiple input testing
  dtype = tf.float32

  @classmethod
  def setUpClass(cls) -> None:
    cls.array = tf.constant(array, dtype=cls.dtype)
    cls.batch = tf.repeat(cls.array[None], cls.batch_size, 0)
    cls.zeros = tf.constant(zeros, dtype=cls.dtype)

  @classmethod
  def tearDownClass(cls) -> None:
    del cls.array
    del cls.zeros
    del cls.batch

  # Get batch expected result from array expected result
  def batch_result(self, array_result: tf.Tensor) -> tf.Tensor:
    return tf.repeat(tf.expand_dims(array_result, 0), self.batch_size, 0)

  # Test a given function
  def default_test(self, f: tff.TensorflowFunction,
                   relax_batch=False,
                   tolerance: float = 10):
    array_result = tf.constant(array_lookup[f.name], self.dtype)
    batch_result = self.batch_result(array_result)
    zero_result = tf.constant(zero_lookup[f.name], self.dtype)
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

  def tf_function_test(self, f: tff.TensorflowFunction,
                       relax_batch=False,
                       tolerance: float = 10):
    f.enable_tf_function()
    self.default_test(f, relax_batch, tolerance=tolerance)
    f.disable_tf_function()

  def test_ackley(self):
    self.default_test(tff.Ackley())
    self.tf_function_test(tff.Ackley())

  def test_griewank(self):
    self.default_test(tff.Griewank())
    self.tf_function_test(tff.Griewank())

  def test_rastrigin(self):
    self.default_test(tff.Rastrigin())
    self.tf_function_test(tff.Rastrigin())

  def test_levy(self):
    self.default_test(tff.Levy())
    self.tf_function_test(tff.Levy())

  def test_rosenbrock(self):
    self.default_test(tff.Rosenbrock())
    self.tf_function_test(tff.Rosenbrock())

  def test_zakharov(self):
    self.default_test(tff.Zakharov())
    self.tf_function_test(tff.Zakharov())

  def test_sum_squares(self):
    self.default_test(tff.SumSquares())
    self.tf_function_test(tff.SumSquares())

  def test_sphere(self):
    self.default_test(tff.Sphere())
    self.tf_function_test(tff.Sphere())

  def test_bent_cigar(self):
    self.default_test(tff.BentCigar(), True)
    self.tf_function_test(tff.BentCigar(), True)

  def test_schumer_steiglitz(self):
    self.default_test(tff.SchumerSteiglitz(), True)
    self.tf_function_test(tff.SchumerSteiglitz(), True)

  def test_powell_sum(self):
    self.default_test(tff.PowellSum(), True)
    self.tf_function_test(tff.PowellSum(), True)

  def test_alpine_2(self):
    self.default_test(tff.Alpine2(), True)
    self.tf_function_test(tff.Alpine2(), True)

  def test_csendes(self):
    self.default_test(tff.Csendes(), True)
    self.tf_function_test(tff.Csendes(), True)

  def test_deb_1(self):
    self.default_test(tff.Deb1(), True)
    self.tf_function_test(tff.Deb1(), True)

  def test_deb_3(self):
    self.default_test(tff.Deb3(), True)
    self.tf_function_test(tff.Deb3(), True)

  def test_qing(self):
    self.default_test(tff.Qing(), True)
    self.tf_function_test(tff.Qing(), True)

  def test_schwefel(self):
    self.default_test(tff.Schwefel(), True)
    self.tf_function_test(tff.Schwefel(), True)

  def test_chung_reynolds(self):
    self.default_test(tff.ChungReynolds(), True)
    self.tf_function_test(tff.ChungReynolds(), True)

  def test_schwefel_2_26(self):
    self.default_test(tff.Schwefel226(), True)
    self.tf_function_test(tff.Schwefel226(), True)

  def test_schwefel_2_22(self):
    self.default_test(tff.Schwefel222(), True)
    self.tf_function_test(tff.Schwefel222(), True)

  def test_schwefel_2_23(self):
    self.default_test(tff.Schwefel223(), True)
    self.tf_function_test(tff.Schwefel223(), True)

  def test_brown(self):
    self.default_test(tff.Brown(), True)
    self.tf_function_test(tff.Brown(), True)

  def test_weierstrass(self):
    self.default_test(tff.Weierstrass(), True, 1e3)
    self.tf_function_test(tff.Weierstrass(), True, 1e3)

  def test_w_wavy(self):
    self.default_test(tff.WWavy(), True)
    self.tf_function_test(tff.WWavy(), True)

  def test_rotated_hyper_ellipsoid(self):
    self.default_test(tff.RotatedHyperEllipsoid())
    self.tf_function_test(tff.RotatedHyperEllipsoid())

  def test_dixon_price(self):
    self.default_test(tff.DixonPrice())
    self.tf_function_test(tff.DixonPrice())


if __name__ == "__main__":
  unittest.main()
