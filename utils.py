"""Utilities for optfuncs."""

import time
import inspect

import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from optfuncs import core
from optfuncs import numpy_functions as npf
from optfuncs import tensorflow_functions as tff


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
