# Copyright (c) 2018 NVIDIA Corporation
import asyncio
import numpy as np
import pandas as pd
import re
import traceback
from typing import Iterable, Mapping, Any, Tuple, Optional

from .backends.base import Backend, JobStatus, RetrievingJobLogsError, \
                           IsWorkerAvailableError, GettingJobStatusError, \
                           KillingJobError, LaunchingJobError
from .search_algorithms.base import SearchAlgorithm


class ExecutionManager:
  def __init__(self,
               backend_manager: Backend,
               search_algorithm: SearchAlgorithm,
               res_pattern: str,
               objective: str,
               constraints: Iterable[Mapping[str, Any]],
               output_file: str = None,
               verbose=0,
               sleep_time=5,
               wait_for_logs_time=10,
               max_retries=5) -> None:
    self._res_pattern = res_pattern
    self._search_algorithm = search_algorithm
    self._backend_manager = backend_manager
    self._num_workers = self._backend_manager.num_workers
    self._constraints = constraints
    self._output_file = output_file
    self._verbose = verbose
    self._sleep_time = sleep_time
    self._wait_for_logs_time = wait_for_logs_time
    self._max_retries = max_retries
    if objective.lower() not in ["minimize", "maximize"]:
      raise ValueError(
        'Objective has to be "minimize" or "maximize", '
        'but "{}" was provided'.format(objective)
      )
    self._objective = objective.lower()
    if self._objective == "minimize":
      self._failure_score = np.inf
    else:
      self._failure_score = -np.inf
    self.final_results = None
    self._cnt = 0

  def _parse_result(self, log: str) -> Optional[float]:
    """This method takes a log string and parses it to produce resulting float.
    More precisely it looks for the last occurrence of ``self._res_pattern``
    in the log file and takes the float which supposed to be present right after
    the sought pattern. If no pattern was found, None is returned.
    """
    res_pos = log.rfind(self._res_pattern)
    # -1 means the res_pattern is not found, returning None
    if res_pos == -1:
      return None
    res_pos += len(self._res_pattern)
    return float(log[res_pos:].split(maxsplit=1)[0])

  def _check_constraints(self, log: str) -> bool:
    """This method returns True if all constraints are satisfied
    and False otherwise. We default to False in case of exception
    """
    def formatter(value_str: str) -> float:
      return float(value_str)
    pattern = None
    value = None
    try:
      for constraint_dict in self._constraints:
        pattern = constraint_dict["pattern"]
        for i, pos in enumerate(re.finditer(pattern, log)):
          if i < constraint_dict.get("skip_first", 0):
            continue
          val_pos = pos.end()
          value_string = log[val_pos:].split(maxsplit=1)[0]
          cur_formatter = constraint_dict.get("formatter", formatter)
          value = cur_formatter(value_string)
          rng = constraint_dict["range"]
          if value < rng[0] or value > rng[1]:
            if self._verbose > 2:
              print('Constraint "{}" not satisfied with value = {}'.format(
                pattern, value,
              ))
            return False
      return True
    except:
      print('Constraint checking with pattern "{}" and with value = {} threw exeption. Setting not satisfied.'.format(
        pattern, value,
      ))
      return False

  async def get_available_worker(self) -> int:
    """This method returns the first available worker.
    The backend is queried for the next worker every sleep_time second.
    """
    while True:
      available_id = -1
      for worker_id in range(self._num_workers):
        try:
          worker_available = self._backend_manager.is_worker_available(worker_id)
        except IsWorkerAvailableError as e:
          worker_available = False
          if self._verbose > 1:
            print("IsWorkerAvailableError raised for worker {}: {}".format(
              worker_id, e.message,
            ))
        if worker_available:
          available_id = worker_id
          break
      if available_id == -1:
        await asyncio.sleep(self._sleep_time)
      else:
        return available_id

  async def _handle_running_job(self,
                                job_info: object,
                                job_params: str,
                                worker_id: int) -> Tuple[Optional[str],
                                                         Optional[float]]:
    """Helper function that handles running jobs."""
    result, job_status = None, None
    try:
      log = self._backend_manager.get_logs_for_job(job_info)
    except RetrievingJobLogsError:
      log = None
    if log is not None:
      if not self._check_constraints(log):
        for i in range(self._max_retries):
          try:
            self._backend_manager.kill_job(job_info)
            break
          except KillingJobError as e:
            if i == self._max_retries - 1:
              if self._verbose > 1:
                print('Could not kill job "{}" on worker {}: {}'.format(
                  job_params, worker_id, e.message,
                ))
              # continuing execution, since worker can't become available
              return None, None

        result = self._failure_score
        job_status = "Some constraints are not satisfied"
        if self._verbose > 1:
          print(
            'Killed job "{}" on worker {}: constraints are not satisfied'
            .format(job_params, worker_id)
          )
    return job_status, result

  async def _handle_succeeded_job(self,
                                  job_info: object,
                                  job_params: str,
                                  worker_id: int) -> Tuple[str, float]:
    """Helper function that handles succeeded jobs."""
    # trying 5 times and than returning None as if the job failed
    log = None
    for i in range(self._max_retries):
      # waiting here in order to let the backend time to finalize results
      await asyncio.sleep(self._wait_for_logs_time)
      try:
        log = self._backend_manager.get_logs_for_job(job_info)
        break
      except RetrievingJobLogsError as e:
        if i == self._max_retries - 1:
          if self._verbose > 1:
            print('Could not access logs for job "{}" on worker {}: {}'.format(
              job_params, worker_id, e.message,
            ))
          return "Job failed: could not access logs", self._failure_score

    # log was successfully retrieved, trying to parse results
    result = self._parse_result(log)
    if result is None:
      if self._verbose > 1:
        print('"{}" was not found in log for job {} on worker {}'.format(
          self._res_pattern, job_params, worker_id,
        ))
      return (
        "Job failed: {} was not found in job's log".format(self._res_pattern),
        self._failure_score
      )

    # got valid result, checking constraints
    if not self._check_constraints(log):
      if self._verbose > 1:
        print("Constraints not satisfied on job {}".format(job_params))
      return "Some constraints are not satisfied", self._failure_score

    # everything is correct, returning result
    if self._verbose > 1:
      print("Got {} {} for job \"{}\" on worker {}".format(
        self._res_pattern, result, job_params, worker_id,
      ))
    return "Job succeeded", result

  async def _handle_failed_job(self,
                               job_info: object,
                               job_params: str,
                               worker_id: int) -> Tuple[str, float]:
    """Helper function that handles failed jobs."""
    if self._verbose > 1:
      print('Job "{}" failed on worker {}'.format(job_params, worker_id))
    return "Job failed", self._failure_score

  async def _start_job_and_push_results(self,
                                        job_params: str,
                                        worker_id: int,
                                        results_queue: asyncio.Queue) -> None:
    """This method is responsible for starting job and pushing result to queue.
    It will launch the job with ``job_params`` on worker ``worker_id`` and
    then wait until job status becomes "succeeded" or "failed", periodically
    checking that job's log satisfies constraints. The backend
    is queried for the job status every sleep_time seconds. As soon as the backend
    reports success or failure the ``(job_params, result, job_status)`` tuple
    is pushed into the ``results_queue``. The result is obtained by getting the
    job log from the backend (trying a few times if something goes wrong) and
    searching for the ``self._res_pattern``. In case of
    failure or when ``self._res_pattern`` was not found in job log, result is
    equal to ``np.inf`` (or ``-np.inf``, depending on the objective).
    """
    # making the function exception-safe, since they are not going to
    # be handled or stop execution of the main program flow
    try:
      for i in range(self._max_retries):
        try:
          job_info = self._backend_manager.launch_job(worker_id, job_params)
          break
        except LaunchingJobError as e:
          if i == self._max_retries - 1:
            if self._verbose > 1:
              print("Backend can't start job {} on worker {}: {}".format(
                job_params, worker_id, e.message,
              ))
            elif self._verbose == 1:
              self._cnt += 1
              print("Processed {} jobs".format(self._cnt), end="\r")
            await results_queue.put((self._failure_score, job_params,
                                     "Job failed: can't launch job on backend"))
            return

      if self._verbose > 1:
        print("Started job \"{}\" on worker {}".format(job_params, worker_id))
      while True:
        for i in range(self._max_retries):
          try:
            status = self._backend_manager.get_job_status(job_info)
            break
          except GettingJobStatusError as e:
            if i == self._max_retries - 1:
              # setting status to JobStatus.RUNNING, since it's unclear
              # what state the job is in currently
              status = JobStatus.RUNNING
              if self._verbose > 1:
                print("Can't get status for job {} on worker {}: {}".format(
                  job_params, worker_id, e.message,
                ))

        if status == JobStatus.RUNNING or status == JobStatus.PENDING:
          job_status, result = await self._handle_running_job(
            job_info, job_params, worker_id,
          )
          if result is None:
            # everything is ok, can continue running this job
            await asyncio.sleep(self._sleep_time)
            continue
        elif status == JobStatus.SUCCEEDED:
          job_status, result = await self._handle_succeeded_job(
            job_info, job_params, worker_id,
          )
        elif status == JobStatus.FAILED or status == JobStatus.KILLED or status == JobStatus.NOTFOUND:
          job_status, result = await self._handle_failed_job(
            job_info, job_params, worker_id,
          )
        else:
          raise RuntimeError("Got unknown status from job: {}".format(status))

        if self._verbose == 1:
          self._cnt += 1
          print("Processed {} jobs".format(self._cnt), end="\r")
        await results_queue.put((result, job_params, job_status))
        return
    except Exception as e:
      if self._verbose > 1:
        print("Job {} on worker {} failed with unhandled exception:".format(
          job_params, worker_id,
        ))
        print(traceback.format_exc())
      elif self._verbose == 1:
        self._cnt += 1
        print("Processed {} jobs".format(self._cnt), end="\r")
      await results_queue.put((self._failure_score, job_params,
                               "Job failed: unhandled exception"))

  async def _process_jobs(self,
                          jobs_queue: asyncio.Queue,
                          results_queue: asyncio.Queue) -> None:
    """Main routine for processing jobs.
    This method will query the ``jobs_queue`` for the jobs parameters and
    launch jobs as soon as new parameters are pushed in the queue. The jobs
    are launched with :meth:`_start_job_and_push_results` calls which are
    wrapped with ``asyncio.ensure_future`` so that they don't block code
    execution. In order to ensure that all jobs are finished, the futures
    objects are stored in ``jobs_dispatched`` list and the method waits for
    all of them before finishing. The main loop will query for the new job
    parameters every sleep_time seconds and will stop as soon as it gets None.
    """
    jobs_dispatched = []
    while True:
      job_params = await jobs_queue.get()
      if job_params is None:
        break
      # converting dictionary into cmd arguments
      job_params = " ".join(
        ["{}={}".format(name, val) for name, val in job_params.items()]
      )

      worker_id = await self.get_available_worker()
      jobs_dispatched.append(asyncio.ensure_future(
        self._start_job_and_push_results(job_params, worker_id, results_queue)
      ))
      # waiting for job to start and make worker busy
      await asyncio.sleep(self._sleep_time)

    for job_dispatched in jobs_dispatched:
      await asyncio.wait_for(job_dispatched, timeout=None)
    await results_queue.put(None)

  async def _generate_jobs(self,
                           jobs_queue: asyncio.Queue,
                           results_queue: asyncio.Queue) -> None:
    """This method is used to generate all search jobs.
    It uses ``self._search_algorithm`` to get the first set of jobs by calling
    ``self._search_algorithm.gen_initial_jobs()`` and pushes all of them to the
    ``jobs_queue``. It then enters the loop until it gets None from the
    ``results_queue``. On each iteration of the loop it will wait for the new
     result to appear in the results_queue and ask the
    ``self._search_algorithm`` to generate new jobs based on the last result
    retrieved using ``self._search_algorithm.gen_new_jobs``. It will then push
    all new jobs into the ``jobs_queue`` and save the current ``results.csv``.
    """
    init_jobs = self._search_algorithm.gen_initial_params()
    for job_params in init_jobs:
      await jobs_queue.put(job_params)

    results = []
    cnt = 0
    while True:
      result_tuple = await results_queue.get()
      if result_tuple is None:
        break
      cnt += 1
      results.append(result_tuple + (cnt,))
      new_jobs = self._search_algorithm.gen_new_params(
        result=result_tuple[0],
        params=dict([arg_val.split('=') for arg_val in result_tuple[1].split()]),
        evaluation_succeeded=(not result_tuple[2].startswith("Job failed"))
      )
      for job_params in new_jobs:
        await jobs_queue.put(job_params)

      if self._objective == "minimize":
        sorted_results = sorted(results)
      else:
        sorted_results = sorted(results, reverse=True)

      if self._output_file:
        pd.DataFrame(
          data=sorted_results,
          columns=[self._res_pattern, "params", "status", "job_id"],
        ).to_csv(self._output_file)

    if self._verbose > 1:
      print(
        "\nTop-10 parameters:\n    {}".format(
          "\n    ".join(["{} {} for job \"{}\"".format(
            self._res_pattern, value, cmd,
          ) for value, cmd, status, job_id in sorted_results[:10]])
        )
      )
    self.final_results = pd.DataFrame(
      data=sorted_results,
      columns=[self._res_pattern, "params", "status", "job_id"],
    )

  def start_tuning(self) -> None:
    """This is the main function that should be called to start tuning."""
    self._cnt = 0
    loop = asyncio.get_event_loop()
    jobs_queue = asyncio.Queue(loop=loop)
    results_queue = asyncio.Queue(loop=loop)
    generate_jobs_coroutine = self._generate_jobs(
      jobs_queue=jobs_queue, results_queue=results_queue,
    )
    process_jobs_coroutine = self._process_jobs(
      jobs_queue=jobs_queue, results_queue=results_queue,
    )
    loop.run_until_complete(asyncio.gather(generate_jobs_coroutine,
                                           process_jobs_coroutine))
    loop.close()
