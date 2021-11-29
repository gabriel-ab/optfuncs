"""Classes base para todas funções."""

import abc
import typing


class Domain(typing.NamedTuple):
  min: float
  max: float


class Function:
  """Classe base para todas funções."""

  def __init__(self, domain: Domain):
    assert domain is not None
    self._domain = domain
    self._fn = self.call

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
  def call(self, x):
    """Default call interface
    This method must have the default implementation of the function.
    """
    pass
