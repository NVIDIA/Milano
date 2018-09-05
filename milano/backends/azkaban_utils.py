# Copyright (c) 2018 NVIDIA Corporation
import requests
import io
import zipfile
from typing import Iterable, Tuple, Sequence


class AzkabanConnectionError(Exception):
  """An exception that is going to be raised if something fails
  with Azkaban access.
  """

  def __init__(self, message):
    self.message = message


class AzkabanManager:
  def __init__(self) -> None:
    self.session_id = None
    self.url_port = None

  def connect(self, url="http://127.0.0.1", port="8081",
              username="azkaban", password="azkaban") -> None:
    data = {
      "action": "login",
      "username": username,
      "password": password,
    }
    url_port = '{}:{}'.format(url, port)
    response = requests.post(url_port, data=data).json()
    if "error" in response:
      raise AzkabanConnectionError(response['error'])
    self.session_id = response['session.id']
    self.url_port = url_port

  def _check_connection(self) -> None:
    if self.session_id is None:
      raise AzkabanConnectionError(
        "AzkabanManager is not connected to server. "
        "Make sure you ran self.connect()."
      )

  def create_project(self, name: str, description: str) -> None:
    self._check_connection()
    data = {
      "action": "create",
      "session.id": self.session_id,
      "name": name,
      "description": description,
    }
    response = requests.post(self.url_port + '/manager', data=data).json()
    # TODO: figure out how to handle this warnings that project exists, since
    #       we usually don't worry about them, but they will interfere with
    #       other important logs that are being printed
    # if "message" in response:
    #   print("WARNING: {}".format(response['message']))

  def upload_zip(self, project_name: str, zipped_file: io.BytesIO) -> None:
    self._check_connection()
    data = {
      "ajax": "upload",
      "session.id": self.session_id,
      "project": project_name,
    }
    files = {"file": ("jobs.zip", zipped_file, "application/zip", {})}
    response = requests.post(self.url_port + '/manager',
                             files=files, data=data).json()
    if "error" in response:
      raise AzkabanConnectionError(response['error'])

  def get_project_flows(self, project_name: str) -> list:
    self._check_connection()
    data = {
      "ajax": "fetchprojectflows",
      "session.id": self.session_id,
      "project": project_name,
    }
    response = requests.get(self.url_port + '/manager', params=data).json()
    return [flow["flowId"] for flow in response["flows"]]

  def run_flow(self, project_name: str, flow_id: str,
               properties: Iterable[Tuple[str, str]] = None) -> dict:
    self._check_connection()
    data = {
      "ajax": "executeFlow",
      "session.id": self.session_id,
      "project": project_name,
      "flow": flow_id,
    }
    for name, value in properties:
      data["flowOverride[{}]".format(name)] = value

    job_info = requests.get(self.url_port + '/executor', params=data).json()
    if "error" in job_info:
      raise AzkabanConnectionError(
        "Got error for flow {} with properties \"{}\": {}".format(
          flow_id, properties, job_info['error'],
        )
      )
    return job_info

  def get_logs_for_job(self, job_info: dict) -> str:
    # TODO: for now this assumes that there is 1 job in the flow
    self._check_connection()
    data = {
      "ajax": "fetchExecJobLogs",
      "session.id": self.session_id,
      "execid": job_info["execid"],
      "jobId": job_info["flow"],
      "offset": 0,
      "length": 10000000,
    }
    response = requests.get(self.url_port + '/executor', params=data)
    if response.status_code != 200:
      raise AzkabanConnectionError(
        'Job "flow={}, exeid={}" returned with status {} and error "{}"'.format(
          job_info["flow"], job_info["execid"],
          response.status_code, response.reason,
        )
      )
    response = response.json()

    if "error" in response:
      raise AzkabanConnectionError(response['error'])
    return response["data"]

  def get_run_status(self, run_info: dict) -> str:
    self._check_connection()
    data = {
      "ajax": "fetchexecflow",
      "session.id": self.session_id,
      "execid": run_info['execid'],
    }
    response = requests.get(self.url_port + '/executor', params=data).json()
    if "error" in response:
      raise AzkabanConnectionError(response['error'])
    return response["status"]

  def kill_flow_execution(self, run_info: dict) -> None:
    self._check_connection()
    data = {
      "ajax": "cancelFlow",
      "session.id": self.session_id,
      "execid": run_info['execid'],
    }
    response = requests.get(self.url_port + '/executor', params=data).json()
    if "error" in response:
      raise AzkabanConnectionError(response['error'])

  def get_flow_executions(self, project_name: str, flow_id: str) -> list:
    self._check_connection()
    data = {
      "ajax": "getRunning",
      "session.id": self.session_id,
      "project": project_name,
      "flow": flow_id,
    }
    response = requests.get(self.url_port + '/executor', params=data).json()
    if "error" in response:
      raise AzkabanConnectionError(response['error'])

    if "execIds" in response:
      return response["execIds"]
    else:
      return []

  def is_flow_running(self, project_name: str, flow_id: str) -> bool:
    exec_ids = self.get_flow_executions(project_name, flow_id)
    return len(exec_ids) > 0


def strings_to_zipped_file(strings_dict: dict) -> io.BytesIO:
  """name: content"""
  zipped_file = io.BytesIO()
  with zipfile.ZipFile(zipped_file, 'w') as f:
    for name, content in strings_dict.items():
      f.writestr(name, content)
  zipped_file.seek(0)
  return zipped_file


def commands_to_job(commands: Sequence[str], job_name="job.job",
                    envs=()) -> str:
  job = "# {}\n".format(job_name)
  for env in envs:
    job += "env.{}\n".format(env)
  job += "type=command\ncommand={}\n".format(commands[0])
  if len(commands) > 1:
    for idx, cmd in enumerate(commands[1:]):
      job += "command.{}={}".format(idx + 1, cmd)
  return job
