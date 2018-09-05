# Copyright (c) 2017 NVIDIA Corporation
import numpy as np
from typing import Iterable, Mapping, Optional
from .base import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
  def _sample_params(self) -> Mapping:
    """Generates new parameters by random sampling."""
    sampled_params = {}
    for pm_name, pm_dict in self._params_to_tune.items():
      if pm_dict["type"] == "range":
        sampled_params[pm_name] = np.random.uniform(
          pm_dict["min"], pm_dict["max"],
        )
      if pm_dict["type"] == "log_range":
        sampled_params[pm_name] = np.exp(np.random.uniform(
          np.log(pm_dict["min"]), np.log(pm_dict["max"]),
        ))
      if pm_dict["type"] == "values":
        sampled_params[pm_name] = np.random.choice(pm_dict["values"])

    return sampled_params

  def gen_initial_params(self) -> Iterable[Mapping]:
    """Generate all parameters here as all evaluations are independent
    from each other.
    """
    init_params = super().gen_initial_params()

    params = []
    for _ in range(self._num_evals):
      params.append(self._sample_params())

    if init_params is not None:
      return init_params + params
    else:
      return params

  def gen_new_params(self,
                     result: float,
                     params: Mapping,
                     evaluation_succeeded: bool) -> Iterable[Optional[Mapping]]:
    """Returning None to signal stop."""
    return [None]
