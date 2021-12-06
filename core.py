"""Classes base para todas funções."""

import abc
import typing
from typing import Union, Sequence


class Domain(typing.NamedTuple):
  min: Union[float, Sequence[float]]
  max: Union[float, Sequence[float]]

class Function(abc.ABC):
  """Base Class for all optimization functions"""

  def __init__(self, domain: Domain):
    assert domain is not None
    self._domain = domain

  @abc.abstractmethod
  def __call__(self, x):
    """Default function implementation
    usage cases:
      1: receive a vector and return a floating point value
      2: receive batch of vectors and return a vector of results
    """
    raise NotImplementedError('method must be implemented')

  def __str__(self) -> str:
    return self.name

  @property
  def name(self):
    return type(self).__name__
  
  @property
  def domain(self):
    return self._domain