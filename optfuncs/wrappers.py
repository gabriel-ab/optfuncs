"""Wrappers for Optimization Functions"""

import abc
from optfuncs import core


class BaseWrapper(abc.ABC):
  """Function Wrapper that extend some function functionality"""
  def __init__(self, function: core.Function) -> None:
    self.func = function

  @abc.abstractmethod
  def __call__(self, x):
    return self.func(x)
  
  @property
  def domain(self):
    return self.func.domain
  
  @property
  def name(self):
    return self.func.name


class BaseGradientWrapper(BaseWrapper, abc.ABC):
  """Base class to store gradients for each function call"""
  
  @property
  @abc.abstractmethod
  def gradients(self):
    pass


class TransformationWrapper(BaseWrapper):
  """Add Transformation to a function"""
  def __init__(self,
               function: core.Function,
               scale_before=1.0,
               translate_before=0.0,
               scale_after=1.0,
               translate_after=0.0):
    
    super().__init__(function)
    self.sb = scale_before
    self.tb = translate_before
    self.sa = scale_after
    self.ta = translate_after

  def __call__(self, x):
    x *= self.sb
    x += self.tb
    x = self.func(x)
    x *= self.sa
    x += self.ta
    return x
  
  @property
  def name(self) -> str:
    return f'Transformed{self.func.name}'

  def __repr__(self) -> str:
    return super().__repr__() + \
      ' - Transformation(*{} -> +{} -> {} ->  *{} -> +{})'.format(
      self.sb, self.tb, self.func.name, self.sa, self.ta)
