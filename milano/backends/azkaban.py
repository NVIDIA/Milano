# Copyright (c) 2018 NVIDIA Corporation
from .base import Backend, JobStatus, RetrievingJobLogsError, KillingJobError, \
                  IsWorkerAvailableError, GettingJobStatusError, \
                  LaunchingJobError
from .azkaban_utils import AzkabanManager, commands_to_job, \
                           strings_to_zipped_file, AzkabanConnectionError
from typing import Iterable


class AzkabanBackend(Backend):
  def __init__(self, script_to_run: str, workers_config: Iterable,
               url="http://127.0.0.1", port="8081",
               username="azkaban", password="azkaban") -> None:
    """Constructor of Azkaban backend class.
    In this method, project named "Milano" will be created with number of flows
    equal to the number of workers. Each flow will have just one job named
    "worker-<worker_id>.job" which will set environment variables as described
    in workers_config and launch script_to_run appending current parameters
    with ${params} argument.

    Note that it's user's responsibility to ensure that script_to_run will
    correctly execute with specified parameters in the environment where
    Azkaban is launched.
    """
    super().__init__(script_to_run, workers_config)
    self._azkaban_manager = AzkabanManager()
    self._azkaban_manager.connect(url, port, username, password)
    self._project_name = "Milano"
    # TODO: delete project if exists?
    self._azkaban_manager.create_project(
      self._project_name, "Milano tuning for {}".format(script_to_run)
    )

    workers_envs = []
    for worker_dict in workers_config:
      workers_envs.extend(
        [worker_dict["env_vars"]] * worker_dict["num_workers"]
      )

    with open(script_to_run, "r") as fin:
      script_code = fin.read()

    strings_dict = {script_to_run: script_code}

    for worker_id, worker_envs in enumerate(workers_envs):
      job_name = 'worker-{}.job'.format(worker_id)
      strings_dict[job_name] = commands_to_job(
        [
          "chmod +x {}".format(script_to_run),
          "./{} ".format(script_to_run) + "${params}",
        ],
        job_name=job_name,
        envs=worker_envs,
      )
    self._azkaban_manager.upload_zip(
      self._project_name,
      strings_to_zipped_file(strings_dict),
    )
    self._num_workers = len(workers_envs)

  def get_job_status(self, job_info: dict) -> JobStatus:
    try:
      status = self._azkaban_manager.get_run_status(job_info)
    except AzkabanConnectionError as e:
      raise GettingJobStatusError(e.message)
    # TODO: check other statuses?
    if status == "SUCCEEDED":
      return JobStatus.SUCCEEDED
    elif status == "FAILED":
      return JobStatus.FAILED
    else:
      return JobStatus.RUNNING

  def kill_job(self, job_info: dict) -> None:
    try:
      self._azkaban_manager.kill_flow_execution(job_info)
    except AzkabanConnectionError as e:
      raise KillingJobError(e.message)

  def get_logs_for_job(self, job_info: dict) -> str:
    try:
      return self._azkaban_manager.get_logs_for_job(job_info)
    except AzkabanConnectionError as e:
      raise RetrievingJobLogsError(e.message)

  def is_worker_available(self, worker_id: int) -> bool:
    try:
      flow_running = self._azkaban_manager.is_flow_running(
        self._project_name, "worker-{}".format(worker_id),
      )
    except AzkabanConnectionError as e:
      raise IsWorkerAvailableError(e.message)
    return not flow_running

  def launch_job(self, worker_id: int, job_params: str) -> dict:
    try:
      return self._azkaban_manager.run_flow(
        project_name=self._project_name,
        flow_id="worker-{}".format(worker_id),
        properties=[("params", job_params)],
      )
    except AzkabanConnectionError as e:
      raise LaunchingJobError(e.message)

  @property
  def num_workers(self):
    return self._num_workers
