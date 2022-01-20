"""Core classes for implementing new functions."""

import abc
import typing


class Domain(typing.NamedTuple):
  min: float
  max: float


class Function:
  """Function's base class."""

  def __init__(self, domain: Domain):
    assert domain is not None
    self._domain = domain
    self._fn = self._call

  def __call__(self, x):
    return self._fn(x)

  @property
  def domain(self) -> Domain:
    return self._domain

  @domain.setter
  def domain(self, new_domain: Domain):
    self._domain = new_domain

  @property
  def name(self):
    return str(self)

  def __str__(self):
    return self.__class__.__name__

  @abc.abstractmethod
  def grads(self, x):
    """Returns the gradients of the function at x.
    """
    pass

  @abc.abstractmethod
  def grads_at(self, x):
    """Returns a tuple containing f(x) and grad(x)."""
    pass

  @abc.abstractmethod
  def _call(self, x):
    """Default call interface
    This method must have the default implementation of the function.
    """
    pass
