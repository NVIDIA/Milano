# Copyright (c) 2017 NVIDIA Corporation
import abc
import six
import numpy as np

from typing import Iterable, Mapping, Optional


@six.add_metaclass(abc.ABCMeta)
class SearchAlgorithm:
  """All search algorithms in MLQuest must inherit from this."""
  def __init__(self,
               params_to_tune: Mapping,
               params_to_try_first: Mapping,
               objective: str,
               num_evals: int,
               random_seed: int = None) -> None:
    """Base SearchAlgorithm constructor.

    Args:
      params_to_tune (dict): dictionary with parameters that need to be tuned.
          Example::
            {
              "x1": {"type": "range", "min": 0.0, "max": 100.0},
              "x2": {"type": "log_range", "min": 1e-7, "max": 1.0},
              "color": {"type": "values", "values": ["red", "green", "blue"]},
            }
          Supported types are:
            * "range": this parameter should be sampled uniformly from
              min to max values.
            * "log_range": this parameter should be sampled "logarithmically"
              from min to max values. This means that a uniform value will be
              sampled between [log(min), log(max)] and then it will be
              exponentiated.
            * "values": this parameter can be one of the supplied values.
      params_to_try_first (dict): dictionary with configurations to try first
      objective (string): "minimize" or "maximize", case insensitive.
      num_evals (int): maximum number of evaluations that the algorithm can do.
      random_seed (int): random seed to use.
    """
    for pm_name, pm_dict in params_to_tune.items():
      if "type" not in pm_dict:
        raise ValueError('"type" has to be specified for each '
                         'parameter, not found for "{}"'.format(pm_name))
      if pm_dict["type"] not in ["range", "log_range", "values"]:
        raise ValueError(
          'Unsupported type: "{}" for "{}". '.format(pm_dict["type"], pm_name) +
          'type has to be "range", "log_range" or "values"'
        )
      if pm_dict["type"] == "range" or pm_dict["type"] == "log_range":
        if "min" not in pm_dict:
          raise ValueError(
            '"min" value has to be specified for parameter {}'.format(pm_name)
          )
        if "max" not in pm_dict:
          raise ValueError(
            '"max" value has to be specified for parameter {}'.format(pm_name)
          )
      if pm_dict["type"] == "log_range":
        if pm_dict["min"] <= 0:
          raise ValueError('"min" value has to be positive '
                           'when type is "log_range"')
        if pm_dict["max"] <= 0:
          raise ValueError('"ma" value has to be positive '
                           'when type is "log_range"')
      if pm_dict["type"] == "values":
        if "values" not in pm_dict:
          raise ValueError(
            '"values" has to be specified for parameter {}'.format(pm_name)
          )
        if len(pm_dict["values"]) == 0:
          raise ValueError("No values specified for {}".format(pm_name))

    self._params_to_tune = params_to_tune
    self._params_to_try_first = params_to_try_first
    if self._params_to_try_first is not None:
      # TODO check for format correctness
      self._pre_configs_counter = len(next(iter(self._params_to_try_first.values())))
    else:
      self._pre_configs_counter = -1
    self._num_evals = num_evals
    if objective.lower() not in ["minimize", "maximize"]:
      raise ValueError(
        'Objective has to be "minimize" or "maximize", '
        'but "{}" was provided'.format(objective)
      )
    self._objective = objective.lower()
    self._random_seed = random_seed
    np.random.seed(self._random_seed)

  #@abc.abstractmethod
  def gen_initial_params(self) -> Iterable[Mapping]:
    """This method should return all initial parameters to start the tuning.

    Returns:
      list of dicts: [{param_name: param_value, ...}, ...]
    """
    if self._params_to_try_first is not None:
      user_pre_specified_experiments = []
      count = len(
        next(iter(self._params_to_try_first.values())))
      for ind in range(0, count):
        one_experiment = {}
        for key, value in self._params_to_try_first.items():
          one_experiment[key] = value[ind]
        user_pre_specified_experiments.append(one_experiment)
      return user_pre_specified_experiments
    else:
      return None


  @abc.abstractmethod
  def gen_new_params(self,
                     result: float,
                     params: Mapping,
                     evaluation_succeeded: bool) -> Iterable[Optional[Mapping]]:
    """This method should return new parameters to evaluate
    based on the last retrieved result.

    To indicate that the search is over (which usually happens when
    `self._num_evals` values have been tried), the method should return
    `None` instead of dictionary with function parameters.

    Args:
      result (float): the value of the function being optimized.
      params (dict): parameters, describing the point at which function was
          evaluated. This is the same dictionary as was returned from
          `self.gen_initial_params` or `self.gen_new_params`.
      evaluation_succeeded (bool): whether the evaluation was successful.
          In big experiments some jobs evaluating the function might fail for
          various reasons. In this case it is up for the algorithm to decide
          if the failed point need to be re-evaluated or if there are more
          promising points to focus on.

    Returns:
      list of dicts: [{param_name: param_value, ...}, ...]
    """
    pass
