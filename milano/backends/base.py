# Copyright (c) 2018 NVIDIA Corporation
"""
This module contains base Backend class and a number of exceptions that
backends can raise.

When defining new backends you must inherit from Backend class. Since it is
important for backend to be failure-safe, we define a number of "safe"
exceptions here which all inherit from BackendError and should be raise by
corresponding Backend functions when "safe" error occurs. By that we mean
errors that should not stop execution of ExecutionManager, but rather be
treated as normal failures and require retrying to run the command that raised
the exception. All other exceptions will not be handled and will generally
break the normal flow of ExecutionManager.
"""

import abc
import six
from enum import Enum


class BackendError(Exception):
  """Base class for exceptions in this module."""
  pass


class RetrievingJobLogsError(BackendError):
  """Exception raised for errors occurring while retrieving job logs."""

  def __init__(self, message):
    self.message = message


class GettingJobStatusError(BackendError):
  """Exception raised for errors occurring while getting job status."""

  def __init__(self, message):
    self.message = message


class KillingJobError(BackendError):
  """Exception raised for errors occurring while trying to kill job."""

  def __init__(self, message):
    self.message = message


class LaunchingJobError(BackendError):
  """Exception raised for errors occurring while trying to launch a job."""

  def __init__(self, message):
    self.message = message


class IsWorkerAvailableError(BackendError):
  """Exception raised for errors occurring while trying to check
  if worker is available.
  """

  def __init__(self, message):
    self.message = message


class JobStatus(Enum):
  RUNNING = 0
  SUCCEEDED = 1
  FAILED = 2
  PENDING = 3
  KILLED = 4
  NOTFOUND = 5
  UNKNOWN = 6


@six.add_metaclass(abc.ABCMeta)
class Backend:
  def __init__(self, script_to_run: str, workers_config: object) -> None:
    self._script_to_run = script_to_run
    self._workers_config = workers_config

  @abc.abstractmethod
  def get_job_status(self, job_info: object) -> JobStatus:
    """This method should take the ``job_info`` as returned from
    ``self.launch_job`` and return correct JobStatus for that job.
    """
    pass

  @abc.abstractmethod
  def get_logs_for_job(self, job_info: object) -> str:
    """This method should take the ``job_info`` as returned from
    ``self.launch_job`` and return job logs or raise ``RetrievingLogsError``
    exception if something goes wrong. If exception is raised, ExecutionManager
    will retry getting logs a few times and indicate that the job failed if
    still unsuccessful.
    """
    pass

  @abc.abstractmethod
  def kill_job(self, job_info: object) -> None:
    """This method should kill the job, identified with ``job_info``.
    ``job_info`` is returned from ``self._launch_job``.
    """
    pass

  @abc.abstractmethod
  def is_worker_available(self, worker_id: int) -> bool:
    """This method takes worker id and should return whether there are any jobs
    running on that worker. ``self.launch_job`` will only be executed on a
    worker that returned True from this method.
    """
    pass

  @abc.abstractmethod
  def launch_job(self, worker_id: int, params: str) -> object:
    """This method should start a new job on a worker <worker_id> with
    parameters specified with ``params`` string. This method does not need to
    check if the worker is available since this method is only called after
    getting True from ``self.is_worker_available`` function.
    """
    pass

  @property
  @abc.abstractmethod
  def num_workers(self) -> int:
    """Total number of workers available."""
    pass
