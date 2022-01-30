"""Utilities for optfuncs."""

import time
import inspect
import typing

import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from optfuncs import core
from optfuncs import numpy_functions as npf
from optfuncs import tensorflow_functions as tff

""" Plotting utils. """


class FunctionDrawer:
  """ Draw 2-Dimensional functions in the given domain.
  domain:
      hypercube tuple with minimum and maximum values for the domain.
      ex: (-10, 10)
          (-5.12, 5.12)
  resolution:
      integer representing quality of render.
  """

  def __init__(self, function: core.Function, resolution=80):
    self._set_mesh(function, resolution)

  def _set_mesh(self, function: core.Function, resolution=80):
    self._fn = function
    self._fig: plt.Figure = plt.figure()
    self._ax = self._fig.add_subplot(projection='3d')
    self._resolution = resolution

    # creating mesh
    linspace = np.linspace(self._fn.domain.min,
                           self._fn.domain.max,
                           self._resolution)
    X, Y = np.lib.meshgrid(linspace, linspace)

    if isinstance(self._fn, npf.NumpyFunction):
      zs = [np.array([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]
      Z = np.array([self._fn(v) for v in zs]).reshape(X.shape)
    else:
      self._fn: tff.TensorflowFunction
      self._fn.enable_tf_function()
      zs = [tf.constant([x, y]) for x, y in zip(np.ravel(X).astype(np.float32),
                                                np.ravel(Y).astype(np.float32))]
      Z = np.array([self._fn(v).numpy() for v in zs]).reshape(X.shape)

    self._cmap = cm.get_cmap("jet")
    self._mesh = (X, Y, Z)

  def clear(self):
    self._ax.clear()

  def draw_mesh(self,
                show: bool = True,
                save: bool = False, **kwargs):
    self.clear()
    self._ax.set_xlabel(r'$x_1$', fontsize=8)
    self._ax.set_ylabel(r'$x_2$', fontsize=8)
    self._ax.set_zlabel(r'$f(x_1, x_2)$', fontsize=8)
    self._ax.plot_surface(self._mesh[0],
                          self._mesh[1],
                          self._mesh[2],
                          rstride=1,
                          cstride=1,
                          cmap=self._cmap,
                          linewidth=0.0,
                          shade=True,
                          **kwargs)
    plt.contour(self._mesh[0],
                self._mesh[1],
                self._mesh[2],
                zdir='z',
                offset=self._ax.get_zlim()[0],
                alpha=0.3)
    if save:
      self._fig.savefig(f"plot-2d-{self._fn.name}")

    if show:
      plt.show()

  def close_fig(self):
    plt.close(self._fig)


def plot_all_functions_in(module, interactive=False, save=True):
  is_tf = module == tff
  is_np = module == npf
  assert is_tf or is_np

  base_cls = tff.TensorflowFunction if is_tf else npf.NumpyFunction

  start_time = time.time()
  for fn in [f() for n, f in
             inspect.getmembers(tff, lambda x:
             inspect.isclass(x) and
             issubclass(x, base_cls) and
             x.__name__ != str(base_cls.__name__))]:
    print(f'--- {fn.name} ---')
    drawer = FunctionDrawer(fn)
    drawer.draw_mesh(show=interactive, save=save)
    drawer.close_fig()
    print('--------------')

  end_time = time.time()
  print(f"Execution time: {end_time - start_time}s")


""" Test utils. """


class FunctionEvaluationExamples:
  default_str = "default"

  default_x_4d: typing.Dict[str, typing.List[float]] = {
    "Mishra2": [0., 0.25, 0.5, 0.75],
    default_str: [1., 2., 3., 4.],
  }
  default_fx_4d: typing.Dict[str, float] = {
    "Ackley": 8.43469444443746497,
    "Alpine2": -0.40033344730936005,
    "BentCigar": 29000001.0,
    "Brown": 1.6678281e+16,
    "ChungReynolds": 900.0,
    "Csendes": 11063.416256526398,
    "Deb1": -6.182844847431069e-87,
    "Deb3": -0.036599504738713866,
    "DixonPrice": 4230.0,
    "Griewank": 1.00187037800320189,
    "Levy": 2.76397190019909811,
    "Mishra2": 49.12257870688604,
    "PowellSum": 1114.0,
    "Qing": 184.0,
    "Rastrigin": 30.0,
    "Rosenbrock": 2705.0,
    "RotatedHyperEllipsoid": 50.0,
    "Salomon": 2.5375017928784365,
    "SchumerSteiglitz": 354.0,
    "Schwefel222": 34.0,
    "Schwefel223": 1108650.0,
    "Schwefel226": -2.353818129766789,
    "Schwefel": 43703.20448793846,
    "Sphere": 30.0,
    "SumSquares": 100.0,
    "Trigonometric2": 42.98949432373047,
    "WWavy": 1.1130512151573806,
    "Weierstrass": 23.999988555908203,
    "Zakharov": 50880.0,
  }

  defaults_x: typing.Dict[int, typing.Dict[str, typing.List[float]]] = {
    4: default_x_4d,
  }
  defaults_fx: typing.Dict[int, typing.Dict[str, float]] = {
    4: default_fx_4d,
  }

  zeros_x_4d: typing.List[float] = [0., 0., 0., 0.]
  zeros_fx_4d: typing.Dict[str, float] = {
    "Ackley": 4.44089209850062616e-16,
    "Alpine2": 0.0,
    "BentCigar": 0.0,
    "Brown": 0.0,
    "ChungReynolds": 0.0,
    "Csendes": 0.0,
    "Deb1": 0.0,
    "Deb3": -0.1249999850988388,
    "DixonPrice": 1.0,
    "Griewank": 0.0,
    "Levy": 0.897533662350923467,
    "Mishra2": 625.0,
    "PowellSum": 0.0,
    "Qing": 30.0,
    "Rastrigin": 0.0,
    "Rosenbrock": 3.0,
    "RotatedHyperEllipsoid": 0.0,
    "Salomon": 0.0,
    "SchumerSteiglitz": 0.0,
    "Schwefel222": 0.0,
    "Schwefel223": 0.0,
    "Schwefel226": 0.0,
    "Schwefel": 0.0,
    "Sphere": 0.0,
    "SumSquares": 0.0,
    "Trigonometric2": 36.10124588012695,
    "WWavy": 0.0,
    "Weierstrass": 23.999988555908203,
    "Zakharov": 0.0,
  }

  zeros_x: typing.Dict[int, typing.List[float]] = {
    4: zeros_x_4d,
  }
  zeros_fx: typing.Dict[int, typing.Dict[str, float]] = {
    4: zeros_fx_4d,
  }

  @classmethod
  def get_default_eval(cls,
                       function: core.Function,
                       dims: int) -> typing.Tuple[typing.List, float]:
    assert dims in cls.defaults_x and dims in cls.defaults_fx
    fn_str = function.name
    x_dict = cls.defaults_x[dims]
    fx_dict = cls.defaults_fx[dims]

    query_arr = x_dict[fn_str] if fn_str in x_dict else x_dict[cls.default_str]
    result = fx_dict[fn_str]

    return query_arr, result

  @classmethod
  def get_eval_at_zeros(cls,
                        function: core.Function,
                        dims: int) -> typing.Tuple[typing.List, float]:
    assert dims in cls.zeros_x and dims in cls.zeros_fx
    fx_dict = cls.zeros_fx[dims]

    query_arr = cls.zeros_x[dims]
    result = fx_dict[function.name]

    return query_arr, result
