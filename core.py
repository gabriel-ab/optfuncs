"""Classes base para todas funções."""

import abc
import typing


class Domain(typing.NamedTuple):
  min: float
  max: float


class Function:
  """Base Class for all mathematical functions"""

  def __init__(self, domain: Domain):
    assert domain is not None
    self._domain = domain
    self._fn = self.call
    self.transformations = None

  def __call__(self, x):
    return self._fn(x)

  def __str__(self) -> str:
    return self.name

  def __repr__(self) -> str:
    default = f'{self.name}({self.domain}) at {id(self)}'
    if self.transformations is None:
      return default
    else:
      sb, tb, sa, ta = self.transformations
      return default + ' - Transformation(*{} -> +{} -> {} ->  *{} -> +{})'.format(
          sb, tb, self.name, sa, ta)

  @property
  def domain(self) -> Domain:
    return self._domain

  @domain.setter
  def domain(self, new_domain: Domain):
    self._domain = new_domain

  @property
  def name(self):
    return type(self).__name__

  @property
  def gradients(self):
    raise NotImplementedError(f'{self.name} has no gradients implemented.')

  def transform(self, scale_before=1.0, translate_before=0.0, scale_after=1.0, translate_after=0.0):
    """Transform current function"""
    self.transformations = scale_before, translate_before, scale_after, translate_after

    def transformed(x):
      x *= scale_before
      x += translate_before
      x = self.call(x)
      x *= scale_after
      x += translate_after
      return x
    self._fn = transformed
    return self

  @abc.abstractmethod
  def call(self, x):
    """Default call interface
    This method must have the default implementation of the function.
    """
    pass
