# ##
# # Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
# #
# # This code is written for research and educational purposes only to
# # supplement the paper entitled "Practical Bayesian Optimization of
# # Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# # in Neural Information Processing Systems, 2012
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful, but
# # WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# # General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see
# # <http://www.gnu.org/licenses/>.

# This code was modified to be compatible with NVAML project

import numpy        as np
import numpy.random as npr

from .sobol_lib import i4_sobol_generate


def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000,
                 compwise=False, verbose=False):
  def direction_slice(direction, init_x):
    def dir_logprob(z):
      return logprob(direction * z + init_x)

    upper = sigma * npr.rand()
    lower = upper - sigma
    llh_s = np.log(npr.rand()) + dir_logprob(0.0)

    l_steps_out = 0
    u_steps_out = 0
    if step_out:
      while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
        l_steps_out += 1
        lower -= sigma
      while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
        u_steps_out += 1
        upper += sigma

    steps_in = 0
    while True:
      steps_in += 1
      new_z = (upper - lower) * npr.rand() + lower
      new_llh = dir_logprob(new_z)
      if np.isnan(new_llh):
        print(new_z, direction * new_z + init_x, new_llh, llh_s, init_x,
              logprob(init_x))
        raise Exception("Slice sampler got a NaN")
      if new_llh > llh_s:
        break
      elif new_z < 0:
        lower = new_z
      elif new_z > 0:
        upper = new_z
      else:
        raise Exception("Slice sampler shrank to zero!")

    if verbose:
      print("Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in)

    return new_z * direction + init_x

  if not init_x.shape:
    init_x = np.array([init_x])

  dims = init_x.shape[0]
  if compwise:
    ordering = np.arange(dims)
    npr.shuffle(ordering)
    cur_x = init_x.copy()
    for d in ordering:
      direction = np.zeros((dims))
      direction[d] = 1.0
      cur_x = direction_slice(direction, cur_x)
    return cur_x

  else:
    direction = npr.randn(dims)
    direction = direction / np.sqrt(np.sum(direction ** 2))
    return direction_slice(direction, init_x)


class Parameter:
  def __init__(self):
    self.type = []
    self.name = []
    self.type = []
    self.min = []
    self.max = []
    self.options = []
    self.int_val = []
    self.dbl_val = []
    self.str_val = []


class GridMap:

  def __init__(self, variables, grid_size):
    self.variables = variables
    self.cardinality = 0

    # Count the total number of dimensions and roll into new format.
    for variable in variables:
      self.cardinality += variable['size']

  # Get a list of candidate experiments generated from a sobol sequence
  def hypercube_grid(self, size, seed):
    # Generate from a sobol sequence
    sobol_grid = np.transpose(i4_sobol_generate(self.cardinality, size, seed))

    return sobol_grid

  # Convert a variable to the unit hypercube
  # Takes a single variable encoded as a list, assuming the ordering is
  # the same as specified in the configuration file
  def to_unit(self, v):
    unit = np.zeros(self.cardinality)
    index = 0

    for variable in self.variables:
      # param.name = variable['name']
      if variable['type'] == 'int':
        for dd in range(variable['size']):
          unit[index] = self._index_unmap(float(v.pop(0)) - variable['min'], (
                  variable['max'] - variable['min']) + 1)
          index += 1

      elif variable['type'] == 'float':
        for dd in range(variable['size']):
          unit[index] = (float(v.pop(0)) - variable['min']) / (
                  variable['max'] - variable['min'])
          index += 1

      elif variable['type'] == 'enum':
        for dd in range(variable['size']):
          unit[index] = variable['options'].index(v.pop(0))
          index += 1
      # TODO: add log_float if this function is going to be used
      else:
        raise Exception("Unknown parameter type.")

    if len(v) > 0:
      raise Exception("Too many variables passed to parser")
    return unit

  def unit_to_list(self, u):
    params = self.get_params(u)
    paramlist = []
    for p in params:
      if p.type == 'int':
        for v in p.int_val:
          paramlist.append(v)
      if p.type == 'float':
        for v in p.dbl_val:
          paramlist.append(v)
      if p.type == 'enum':
        for v in p.str_val:
          paramlist.append(v)
    return paramlist

  def get_params(self, u):
    if u.shape[0] != self.cardinality:
      raise Exception("Hypercube dimensionality is incorrect.")

    params = []
    index = 0
    for variable in self.variables:
      param = Parameter()

      param.name = variable['name']
      if variable['type'] == 'int':
        param.type = 'int'
        for dd in range(variable['size']):
          param.int_val.append(
            variable['min'] + self._index_map(u[index], variable['max'] -
                                              variable['min'] + 1)
          )
          index += 1

      elif variable['type'] == 'float':
        param.type = 'float'
        for dd in range(variable['size']):
          val = variable['min'] + u[index] * (variable['max'] - variable['min'])
          val = variable['min'] if val < variable['min'] else val
          val = variable['max'] if val > variable['max'] else val
          param.dbl_val.append(val)
          index += 1

      elif variable['type'] == 'log_float':
        param.type = 'float'
        for dd in range(variable['size']):
          val = np.log(variable['min']) + u[index] * (np.log(variable['max']) - np.log(variable['min']))
          val = np.log(variable['min']) if val < np.log(variable['min']) else val
          val = np.log(variable['max']) if val > np.log(variable['max']) else val
          param.dbl_val.append(np.exp(val))
          index += 1

      elif variable['type'] == 'enum':
        param.type = 'enum'
        for dd in range(variable['size']):
          ii = self._index_map(u[index], len(variable['options']))
          index += 1
          param.str_val.append(variable['options'][ii])

      else:
        raise Exception("Unknown parameter type.")

      params.append(param)

    return params

  def card(self):
    return self.cardinality

  def _index_map(self, u, items):
    return int(np.floor((1 - np.finfo(float).eps) * u * float(items)))

  def _index_unmap(self, u, items):
    return float(float(u) / float(items))