# Copyright (c) 2018 NVIDIA Corporation
import subprocess
import json
import os
import shutil
import copy
import time
import re
from .utils import SSHClient

from .base import Backend, JobStatus, RetrievingJobLogsError, \
                  GettingJobStatusError, KillingJobError, \
                  IsWorkerAvailableError, LaunchingJobError

class SLURMBackend(Backend):
  """Class for working with SLURM backend."""
  def __init__(self, script_to_run: str, workers_config: dict) -> None:
    super().__init__(script_to_run, workers_config)

    self._workers_config = workers_config
    self._num_workers = self._workers_config['num_workers']
    self._partition = self._workers_config['partition']
    self._key_path = workers_config['key_path']
    self._entrypoint = workers_config['entrypoint']
    self._username = workers_config['username']

    with open(script_to_run) as fin:
      self._script_code = fin.read()

    self._ssh_client = SSHClient(private_key_path=self._key_path)
    try:
      self._ssh_client.connect(address=self._entrypoint,
                               username=self._username)
      self._ssh_client.exec_command_blocking("cat {} > milano_script.sh")
      self._workers_job = [-1] * self._num_workers
    except:
      raise Exception("Couldn't connect to the backend. Check your credentials")

  def get_job_status(self, job_info: object) -> JobStatus:
    job_id = int(job_info)
    try:
      ec, stdout, stderr = self._ssh_client.exec_command_blocking(
        "scontrol show job {}".format(job_id))
    except:
      raise GettingJobStatusError(stderr)

    match = re.search('JobState=(\S*)', stdout, re.IGNORECASE)
    if match is not None:
      result_string = match.group(1)
    else:
      return JobStatus.NOTFOUND

    if result_string == "COMPLETED":
      return JobStatus.SUCCEEDED
    elif result_string == "FAILED" or result_string == "NODE_FAIL" or \
        result_string == "REVOKED" or result_string == "TIMEOUT":
      return JobStatus.FAILED
    elif result_string == "PENDING":
      return JobStatus.PENDING
    elif result_string == "RUNNING" or result_string == "CONFIGURING" or \
        result_string == "COMPLETING":
      return JobStatus.RUNNING
    elif result_string == "CANCELLED" or result_string == "STOPPED" or \
        result_string == "SUSPENDED":
      return JobStatus.KILLED
    else:
      print("~~~~~~~~~~~~~~~~~Got the following status: {}".format(result_string))
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      print(stdout)
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      return JobStatus.UNKNOWN

  def get_logs_for_job(self, job_info: object) -> str:
    try:
      job_id = int(job_info)
      ec, stdout, stderr = self._ssh_client.exec_command_blocking(
        "scontrol show job {}".format(job_id))
      match = re.search('StdOut=(\S*)', stdout, re.IGNORECASE)
      path = match.group(1)
      ec, stdout, stderr = self._ssh_client.exec_command_blocking(
        "cat {}".format(path))
      return stdout
    except:
      raise RetrievingJobLogsError(stderr)

  def kill_job(self, job_info: object) -> None:
    try:
      job_id = job_info
      ec, stdout, stderr = self._ssh_client.exec_command_blocking(
        "scancel {}".format(job_id))
    except:
      raise KillingJobError(stderr)

  def is_worker_available(self, worker_id: int) -> bool:
    if self._workers_job[worker_id] == -1:
      return True
    else:
      try:
        status = self.get_job_status(self._workers_job[worker_id])
      except GettingJobStatusError as e:
        raise IsWorkerAvailableError(e.message)
      if status == JobStatus.RUNNING or status == JobStatus.PENDING \
          or status == JobStatus.KILLED:
        return False
      else:
        self._workers_job[worker_id] = -1
        return True

  def launch_job(self, worker_id: int, params: str) -> object:
    # first, copy script to entrypoint
    try:
      script_code = self._script_code.replace('"$@"', params)
      script_name = "script-{}-{}.sh".format(time.time(), worker_id)
      # move script to entrypoint
      ec, stdout, stderr = self._ssh_client.exec_command_blocking('echo "{}" > {}'.format(script_code,
                                                                   script_name))
      # execute script on entrypoint
      ec, stdout, stderr = self._ssh_client\
        .exec_command_blocking('sbatch -p {} {}'
                               .format(self._partition, script_name))
      match = re.search('Submitted batch job (\S*)', stdout, re.IGNORECASE)
      job_id = int(match.group(1))
      self._workers_job[worker_id] = job_id
      return job_id
    except:
      LaunchingJobError(stderr)


  @property
  def num_workers(self) -> int:
    return self._num_workers