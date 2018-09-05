# Copyright (c) 2018 NVIDIA Corporation
from .aws_utils import EC2InstanceManager

from .base import Backend, JobStatus, RetrievingJobLogsError, \
                  GettingJobStatusError, KillingJobError, \
                  IsWorkerAvailableError, LaunchingJobError


class AWSJob:
  def __init__(self, worker, container_id):
    self._container_id = container_id
    self._worker = worker
    self._archived = False
    self._archived_logs = ""
    self._archived_exit_code = ""

  def is_running(self):
    if self._archived:
      return False
    try:
      return self._exec("sudo docker inspect -f {{.State.Running}} " +
                        self._container_id).strip() == 'true'
    except Exception as e:
      # If something went wrong, assume it's not running anymore.
      # Should probably retry here.
      return False

  def exit_code(self):
    if self._archived:
      return self._archived_exit_code
    try:
      return int(
          self._exec("sudo docker inspect -f {{.State.ExitCode}} " +
                     self._container_id).strip())
    except Exception as e:
      # If something went wrong, assume it's not running anymore.
      # Should probably retry here.
      return -1

  def logs(self):
    if self._archived:
      return self._archived_logs
    return self._exec("sudo docker logs " + self._container_id)

  def kill(self):
    # It would be nice if we could just rely on docker to keep these, but we
    # have tp prune old containers as we launch new ones, or the instance's EBS
    # volume fills up really quickly.
    try:
      self._archived_exit_code = self.exit_code()
      self._archived_logs = self.logs()
    except:
      self._archived_exit_code = -1
      pass
    self._archived = True
    try:
      self._exec("sudo docker kill " + self._container_id)
    except Exception as e:
      # It's fine if we fail to find the container to kill.
      pass
    self._worker = None

  def _exec(self, command):
    exit_code, stdout, stderr = self._worker.exec_command_blocking(command)
    if exit_code != 0:
      raise Exception(
          "remote command failed with exit code {code}: {log}".format(
              code=exit_code, log=stderr))
    return stdout


class AWSBackend(Backend):

  def __init__(self, script_to_run: str, config: dict) -> None:
    super().__init__(script_to_run, config)

    self._config = config
    with open(script_to_run) as fin:
      self._script_code = fin.read()

    self._datasets = self._config.get('datasets', [])

    self._instance_manager = EC2InstanceManager(
        count=self.num_workers,
        region_name=self._config.get('region_name', "us-west-2"),
        key_name=self._config['key_name'],
        private_key_path=self._config['private_key_path'],
        spot_instances=self._config.get('spot_instances', False),
        datasets=self._datasets,
        iam_role=self._config.get('iam_role', None),
        user_params=self._config.get('instance_params', {}),
    )

    self._instance_workers = {}
    self._worker_instances = [None] * self.num_workers

    self._worker_jobs = [-1] * self.num_workers
    self._job_workers = {}

    self._jobs = {}
    self._job_index = 0

  def _get_job(self, job_info: int) -> AWSJob:
    job = self._jobs[job_info]
    if job is None:
      raise Exception("no active job for id {}".format(job_info))
    return job

  def get_job_status(self, job_info: int) -> JobStatus:
    try:
      job = self._get_job(job_info)
      if job.is_running():
        return JobStatus.RUNNING
      elif job.exit_code() == 0:
        return JobStatus.SUCCEEDED
      else:
        return JobStatus.FAILED
    except Exception as e:
      return GettingJobStatusError(
          "failed to retrieve job status: {}".format(e))

  def get_logs_for_job(self, job_info: int) -> str:
    try:
      job = self._get_job(job_info)
      return job.logs()
    except Exception as e:
      print("error retrieving logs", e)
      raise RetrievingJobLogsError("failed to retrieve logs: {}".format(e))

  def kill_job(self, job_info: int) -> None:
    job = self._get_job(job_info)
    job.kill()
    self._worker_jobs[self._job_workers[job_info]] = -1
    del self._job_workers[job_info]

  def _update_worker_instances(self):
    # Since the backend API relies on these fixed worker ids, we need a
    # dynamic mapping from those to the actual instances, which can be swapped
    # in and out.
    active_ids = self._instance_manager.active_instance_ids()
    active_id_set = set(active_ids)
    for i in range(self.num_workers):
      if self._worker_instances[i] is not None:
        if self._worker_instances[i] not in active_ids and self._worker_jobs[i] == -1:
          # This worker is assigned to an inactive instance, and it has no job
          # currently running. Free this slot.
          del self._instance_workers[self._worker_instances[i]]
          self._worker_instances[i] = None
    for active_id in active_ids:
      if active_id not in self._instance_workers:
        # Try to assign a worker slot for this instance, since it's unassigned.
        for i in range(self.num_workers):
          if self._worker_instances[i] is None:
            self._worker_instances[i] = active_id
            self._instance_workers[active_id] = i
            break
    return active_id_set

  def is_worker_available(self, worker_id: int) -> bool:
    active_instances = self._update_worker_instances()
    if self._worker_instances[worker_id] is None:
      # This worker slot isn't assigned to an instance, unavailable.
      return False
    job_id = self._worker_jobs[worker_id]
    if job_id == -1:
      return self._worker_instances[worker_id] in active_instances
    else:
      try:
        status = self.get_job_status(job_id)
      except GettingJobStatusError as e:
        raise IsWorkerAvailableError(e.message)
      if status == JobStatus.RUNNING:
        return False
      else:
        self.kill_job(job_id)
        return self._worker_instances[worker_id] in active_instances

  def _worker_exec(self, worker_id, command):
    instance = self._instance_manager.get_instance(
        self._worker_instances[worker_id])
    exit_code, stdout, stderr = instance.exec_command_blocking(command)
    if exit_code != 0:
      raise Exception(
          "remote command failed with exit code {code}: {log}".format(
              code=exit_code, log=stderr))
    return stdout

  def launch_job(self, worker_id: int, params: str) -> int:
    if not self.is_worker_available(worker_id):
      raise LaunchingJobError("worker busy")

    # The command to run inside the docker container.
    command = "echo $'{script}' > start_exp.sh && chmod +x start_exp.sh && ./start_exp.sh {params}".format(
        script=self._script_code.replace("'", "\\'").replace("\n", "\\n"),
        params=params)

    # Bind datasets.
    volumes = ""
    for i in range(len(self._datasets)):
      volumes += "-v {src}:{dst} ".format(
        src="/home/ubuntu/data/" + str(i),
        dst=self._datasets[i]['mount'],
      )
  
    # The command for running the docker container.
    docker_command = "sudo nvidia-docker run -d {volumes} {docker_image} /bin/bash -c $'{command}'".format(
        volumes=volumes,
        docker_image=self._config['docker_image_name'],
        command=command.replace("'", "\\'"))

    try:
      # Currently we allow only one job per worker, so it's safe to kill any
      # stragglers and purge.
      self._worker_exec(worker_id,
                        "sudo docker kill $(sudo docker ps -q) || true")
      self._worker_exec(worker_id, "sudo docker container prune -f")
      job_id = self._job_index
      self._job_index = self._job_index + 1
      print("launching job", job_id, "on worker", worker_id)
      # docker run -d returns the container id via stdout
      container_id = self._worker_exec(worker_id, docker_command).strip()
      instance = self._instance_manager.get_instance(
          self._worker_instances[worker_id])
      self._jobs[job_id] = AWSJob(instance, container_id)
    except Exception as e:
      raise LaunchingJobError("failed to launch job on worker: {}".format(
          worker_id, e))

    self._worker_jobs[worker_id] = job_id
    self._job_workers[job_id] = worker_id

    return job_id

  @property
  def num_workers(self) -> int:
    return self._config['num_workers']
