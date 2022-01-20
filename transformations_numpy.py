import abc

import numpy as np

from optfuncs import numpy_functions as npf


class NumpyFunctionWrapper(npf.NumpyFunction):
  def __init__(self, function: npf.NumpyFunction):
    super(NumpyFunctionWrapper, self).__init__(function.domain)
    self._src_fn: npf.NumpyFunction = function

  @property
  def name(self):
    return f"{str(self)}: {self._src_fn.name}"

  @abc.abstractmethod
  def _call(self, x: np.ndarray) -> np.ndarray:
    pass


class VerticalShift(NumpyFunctionWrapper):
  def __init__(self,
               function: npf.NumpyFunction,
               shift: float):
    super(VerticalShift, self).__init__(function)
    self._shift = shift

  def _call(self, x: np.ndarray) -> np.ndarray:
    return self._src_fn(x) + self._shift


class HorizontalShift(NumpyFunctionWrapper):
  def __init__(self,
               function: npf.NumpyFunction,
               shift: float):
    super(HorizontalShift, self).__init__(function)
    self._shift = shift

  def _call(self, x: np.ndarray) -> np.ndarray:
    return self._src_fn(x + self._shift)


class UniformScaling(NumpyFunctionWrapper):
  def __init__(self,
               function: npf.NumpyFunction,
               inner_scale: float,
               outer_scale: float):
    super(UniformScaling, self).__init__(function)
    self._inner = inner_scale
    self._outer = outer_scale

  def _call(self, x: np.ndarray) -> np.ndarray:
    return self._outer * self._src_fn(self._inner * x)
