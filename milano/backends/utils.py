# Copyright (c) 2018 NVIDIA Corporation
import paramiko
import time

class RemoteCommand:
  """Represents a command run via ssh"""
  def __init__(self, channel):
    self._channel = channel
    self._stdout_buffer = ""
    self._stderr_buffer = ""
    self._exit_code = None

  def is_running(self):
    return self._exit_code is None and not self._channel.exit_status_ready()

  def exit_code(self):
    if self.is_running():
      return None
    if self._exit_code is None:
      self._exit_code = self._channel.recv_exit_status()
    return self._exit_code

  def poll(self):
    while self._channel.recv_ready():
      chunk = self._channel.recv(1024 * 64)
      self._stdout_buffer = self._stdout_buffer + chunk.decode("utf-8")
    while self._channel.recv_stderr_ready():
      chunk = self._channel.recv_stderr(1024 * 64)
      self._stderr_buffer = self._stderr_buffer + chunk.decode("utf-8")

  def stdout(self):
    self.poll()
    return self._stdout_buffer

  def stderr(self):
    self.poll()
    return self._stderr_buffer

  def close(self):
    self._channel.close()


class SSHClient:
  """SSH client to run commands on the backend."""
  def __init__(self, private_key_path):
    self.private_key_path = private_key_path
    self.client = paramiko.SSHClient()

  def connect(self, address, username):
    self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    self.client.connect(
        address, username=username, key_filename=self.private_key_path)

  def exec_command(self, command):
    channel = self.client.get_transport().open_session()
    channel.exec_command(command)
    return RemoteCommand(channel)

  def exec_command_blocking(self, command):
    rc = self.exec_command(command)
    while rc.is_running():
      rc.poll()
      time.sleep(0.1)
    ec = rc.exit_code()
    stdout = rc.stdout()
    stderr = rc.stderr()
    rc.close()
    return ec, stdout, stderr

  def close(self):
    self.client.close()