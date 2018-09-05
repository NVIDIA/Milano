# Copyright (c) 2017 NVIDIA Corporation
# This is wrapper around spearmint library. Please note that it imports code licensed under GPL v3.
import numpy as np
import collections

from typing import Iterable, Mapping, Optional

from milano.search_algorithms.base import SearchAlgorithm
from milano.search_algorithms.gp.spearmint.gpei_chooser import GPEIChooser
from milano.search_algorithms.gp.spearmint.utils import GridMap


def hash_dict(dct):
  return " ".join("{}={}".format(key, val) for key, val in sorted(dct.items()))


class GPSearch(SearchAlgorithm):
  CANDIDATE_STATUS = 0
  PENDING_STATUS = 1
  COMPLETE_STATUS = 2

  def __init__(self,
               params_to_tune: Mapping,
               params_to_try_first: Mapping,
               objective: str,
               num_evals: int,
               random_seed: int = None,
               chooser=None,
               chooser_params: Mapping = None,
               num_init_jobs=1,
               num_jobs_to_launch_each_time=1,
               grid_size=1000,
               smooth_inf_to=1e7) -> None:
    super().__init__(params_to_tune, params_to_try_first,
                     objective, num_evals, random_seed)
    self._num_init_jobs = num_init_jobs
    self._num_jobs_to_launch_each_time = num_jobs_to_launch_each_time
    self._fixed_params = {}
    self._smooth_inf_to = smooth_inf_to

    params = []
    self._pm_names = []
    for pm_name, pm_dict in self._params_to_tune.items():
      if pm_dict["type"] == "range":
        self._pm_names.append(pm_name)
        params.append(collections.OrderedDict([
          ('name', pm_name),
          ('type', 'float'),
          ('min', pm_dict["min"]),
          ('max', pm_dict["max"]),
          ('size', 1)
        ]))
      elif pm_dict["type"] == "log_range":
        self._pm_names.append(pm_name)
        params.append(collections.OrderedDict([
          ('name', pm_name),
          ('type', 'log_float'),
          ('min', pm_dict["min"]),
          ('max', pm_dict["max"]),
          ('size', 1)
        ]))
      elif pm_dict["type"] == "values":
        if len(pm_dict["values"]) > 1:
          self._pm_names.append(pm_name)
          params.append(collections.OrderedDict([
            ('name', pm_name),
            ('type', 'enum'),
            ('options', pm_dict["values"]),
            ('size', 1)
          ]))
        else:
          self._fixed_params[pm_name] = pm_dict["values"][0]

    if chooser is None:
      self._chooser = GPEIChooser(noiseless=True)
    else:
      if chooser_params is None:
        chooser_params = {}
      self._chooser = chooser(**chooser_params)

    self._gmap = GridMap(params, grid_size)

    # has to be explicitly set to number fo Sobol sequence
    if random_seed is None:
      random_seed = np.random.randint(100000)
    self._grid = self._gmap.hypercube_grid(grid_size, random_seed)

    self._values = np.zeros(grid_size) + np.inf
    self._durations = np.zeros(grid_size) + np.inf
    self._status = np.zeros(grid_size) + GPSearch.CANDIDATE_STATUS

    self._params_to_id = {}
    self._evals_count = 0

  def _add_to_grid(self, candidate):
    # Checks to prevent numerical over/underflow from corrupting the grid
    candidate[candidate > 1.0] = 1.0
    candidate[candidate < 0.0] = 0.0

    # Set up the grid
    self._grid = np.vstack((self._grid, candidate))
    self._status = np.append(
      self._status,
      np.zeros(1, dtype=int) + GPSearch.CANDIDATE_STATUS,
    )

    self._values = np.append(self._values, np.zeros(1) + np.inf)
    self._durations = np.append(self._durations, np.zeros(1) + np.inf)

    return self._grid.shape[0] - 1

  def _get_new_point(self) -> Mapping:
    job_id = self._chooser.next(
      self._grid, self._values, self._durations,
      np.nonzero(self._status == GPSearch.CANDIDATE_STATUS)[0],
      np.nonzero(self._status == GPSearch.PENDING_STATUS)[0],
      np.nonzero(self._status == GPSearch.COMPLETE_STATUS)[0],
    )

    # spearmint can return tuple when it decides to add new points to the grid
    if isinstance(job_id, tuple):
      (job_id, candidate) = job_id
      job_id = self._add_to_grid(candidate)

    candidate = self._grid[job_id]
    self._status[job_id] = GPSearch.PENDING_STATUS

    cur_params = dict(zip(self._pm_names, self._gmap.unit_to_list(candidate)))
    cur_params.update(self._fixed_params)

    # if we ever generate same parameters again, want to remember all of them
    # and then take arbitrary grid id, since they all will point to the same
    # point in our search space
    pm_hash = hash_dict(cur_params)
    if pm_hash not in self._params_to_id:
      self._params_to_id[pm_hash] = []
    self._params_to_id[pm_hash].append(job_id)
    self._evals_count += 1
    return cur_params

  def gen_initial_params(self) -> Iterable[Mapping]:
    init_params = super().gen_initial_params()

    params = []
    for _ in range(min(self._num_evals, self._num_init_jobs)):
      params.append(self._get_new_point())

    if init_params is not None:
      return init_params + params
    else:
      return params

  def gen_new_params(self,
                     result: float,
                     params: Mapping,
                     evaluation_succeeded: bool) -> Iterable[Optional[Mapping]]:
    if self._evals_count == self._num_evals:
      return [None]
    idx = self._params_to_id[hash_dict(params)].pop()
    if evaluation_succeeded:
      self._status[idx] = GPSearch.COMPLETE_STATUS
      if self._objective == "maximize":
        result = -result
      # smoothing out infinities that can arise from constraints failure
      if np.isinf(result):
        result = self._smooth_inf_to

      self._values[idx] = result
    else:
      # if not succeeded, marking point as a potential candidate again
      self._status[idx] = GPSearch.CANDIDATE_STATUS

    params = []
    for _ in range(self._num_jobs_to_launch_each_time):
      params.append(self._get_new_point())

    return params
