# Copyright (c) 2018 NVIDIA Corporation
#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import inspect
import sys


class BenchmarkGenerator:
  def __init__(self, random_seed=None, dim=1):
    if random_seed is not None:
      np.random.seed(random_seed)
    self._dim = dim

  def _gen_xopt_fopt(self):
    x_opt = np.random.uniform(-5.0, 5.0, size=self._dim)
    # TODO: should this be Cauchy as in the BBOB paper?
    f_opt = np.random.uniform(-1000, 1000)
    return x_opt, f_opt

  def _tosz(self, x):
    x_hat = np.zeros_like(x)
    x_hat[x != 0] = np.log(np.abs(x[x != 0]))
    c1 = np.ones_like(x) * 5.5
    c1[x > 0] = 10
    c2 = np.ones_like(x) * 3.1
    c2[x > 0] = 7.9
    return np.sign(x) * \
           np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))

  def _tasy(self, x, beta):
    if self._dim == 1:
      raise ValueError("Tasy can't be applied with D=1, use dim > 1")
    assert len(x.shape) == 1
    assert x.shape[0] == self._dim
    x_tr = x.copy()
    x_tr_gr0 = x_tr[x_tr > 0]
    tmp_pr = (np.arange(self._dim) / (self._dim - 1.0))[x_tr > 0]
    x_tr[x_tr > 0] = x_tr_gr0 ** (1.0 + beta * tmp_pr * np.sqrt(x_tr_gr0))
    return x_tr

  def _lambda(self, alpha):
    if self._dim == 1:
      raise ValueError("Can't build lambda with dim < 2")
    return np.diag(alpha ** (0.5 * np.arange(self._dim) / (self._dim - 1.0)))

  def get_function_by_name(self, name):
    for m_name, m_func in inspect.getmembers(self, predicate=inspect.ismethod):
      if m_name == "get_{}".format(name):
        return m_func()
    raise ValueError('Function "{}" is not supported'.format(name))

  def get_sphere(self):
    x_opt, f_opt = self._gen_xopt_fopt()

    def func(x):
      z = x - x_opt
      return np.sum(z ** 2) + f_opt

    return func, x_opt, f_opt

  def get_elipsoidal(self):
    if self._dim == 1:
      raise ValueError("Can't build 1D elipsoid, dim should be > 1")
    x_opt, f_opt = self._gen_xopt_fopt()

    def func(x):
      z = self._tosz(x - x_opt)
      return np.sum(1e6 ** (np.arange(self._dim) / (self._dim - 1.0)) *
                    z ** 2) + f_opt

    return func, x_opt, f_opt

  def get_rastrigin(self):
    x_opt, f_opt = self._gen_xopt_fopt()

    def func(x):
      z = self._lambda(10.0).dot(self._tasy(self._tosz(x - x_opt), 0.2))
      return 10.0 * (self._dim - np.sum(np.cos(2.0 * np.pi * z))) + \
             np.sum(z ** 2) + f_opt

    return func, x_opt, f_opt

  def get_rosenbrock(self):
    x_opt, f_opt = self._gen_xopt_fopt()

    def func(x):
      z = np.maximum(1.0, np.sqrt(1.0 * self._dim) / 8.0) * (x - x_opt) + 1.0
      return np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2) \
             + f_opt

    return func, x_opt, f_opt

  def visualize_function(self, func, show_3d=True, show_3d_inv=True,
                         show_contour=True, num_levels=15, rng_x=(-5, 5),
                         rng_y=(-5, 5)):
    import matplotlib.pyplot as plt
    if self._dim == 1:
      xs = np.linspace(rng_x[0], rng_x[1], 100)
      plt.plot(xs, np.apply_along_axis(func, 0, xs[np.newaxis, :]))
    elif self._dim == 2:
      freq = 50
      x = np.linspace(rng_x[0], rng_x[1], freq)
      y = np.linspace(rng_y[0], rng_y[1], freq)
      Xs, Ys = np.meshgrid(x, y)
      xs = np.reshape(Xs, -1)
      ys = np.reshape(Ys, -1)
      zs = np.apply_along_axis(func, 0, np.vstack((xs, ys)))

      if show_3d:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True)
        plt.show()
      if show_3d_inv:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.invert_zaxis()
        ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True)
        plt.show()
      if show_contour:
        fig = plt.figure(figsize=(10, 10))
        cs = plt.contour(Xs, Ys, zs.reshape(freq, freq), num_levels)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.show()
    else:
      raise ValueError("Only dim=1 or dim=2 are supported")


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Not enough arguments provided. Should be func_name= x1= x2= ...")
    sys.exit(1)
  func_name = None
  x = np.empty(len(sys.argv) - 2)
  for arg in sys.argv[1:]:
    name, value = arg.split('=')
    if name == 'func_name':
      func_name = value
    else:
      pos = int(name[1:])
      x[pos] = value
  if func_name is None:
    raise ValueError("func_name is not defined")
  benchmarks = BenchmarkGenerator(random_seed=0, dim=x.shape[0])
  func, x_opt, f_opt = benchmarks.get_function_by_name(func_name)
  value = func(x)
  print("Result: {}".format(np.abs(f_opt - value)))
