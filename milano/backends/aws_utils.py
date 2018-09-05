# Copyright (c) 2018 NVIDIA Corporation
import boto3
import botocore
import time
import threading
import json
import hashlib
from milano.backends.utils import SSHClient


class EC2Instance:
  def __init__(self, resource, username, private_key_path):
    self._resource = resource
    self._private_key_path = private_key_path
    self._username = username
    self._ssh_client = None
    self._terminated = False

  def is_running(self):
    return self.state() == 'running'

  def is_terminated(self):
    s = self.state()
    return s != 'pending' and s != 'running'

  def state(self):
    self._reload()
    s = self._resource.state['Name']
    if s == 'terminated':
      self._terminated = True
    return s

  def public_ip(self):
    self._reload()
    return self._resource.public_ip_address

  def instance_id(self):
    return self._resource.instance_id

  def _reload(self):
    if not self._terminated:
      self._resource.reload()

  def __try_connect(self):
    if self._resource.state['Name'] != 'running':
      raise Exception("instance not running")
    if self._ssh_client is None:
      client = SSHClient(self._private_key_path)
      client.connect(self.public_ip(), self._username)
      self._ssh_client = client

  def exec_command(self, command):
    self.__try_connect()
    return self._ssh_client.exec_command(command)

  def exec_command_blocking(self, command, retries=3):
    for i in range(retries):
      try:
        self.__try_connect()
        return self._ssh_client.exec_command_blocking(command)
      except Exception as e:
        if i < retries - 1:
          try:
            if self._ssh_client is not None:
              self._ssh_client.close()
          except:
            pass
          self._ssh_client = None
        else:
          raise e

  def keep_alive(self):
    # As long as this file remains less than 5 minutes old, the instance
    # won't terminate.
    try:
      self.exec_command_blocking("touch /home/ubuntu/.milano_keep_alive")
    except:
      pass

  def is_driver_working(self):
    try:
      ec, _, _ = self.exec_command_blocking("nvidia-smi")
      return ec == 0
    except:
      return False

  def datasets_present(self, datasets):
    try:
      for i in range(len(datasets)):
        ec, _, _ = self.exec_command_blocking("ls /home/ubuntu/data/" + str(i))
        if ec != 0:
          return False
    except:
      return False
    return True

  def terminate(self):
    return self._resource.terminate()


def startup_script(datasets):
  dataset_mounts = "\n"
  for i in range(len(datasets)):
    if datasets[i]['type'] == 's3':
      dataset_mounts += "aws s3 sync {src} {dst}\n".format(
        src="s3://{bucket}/{prefix}".format(
          bucket=datasets[i]['bucket'],
          prefix=datasets[i].get('prefix', "")),
        dst="/home/ubuntu/data/" + str(i),
      )
    else:
      raise Exception("unrecognized dataset source type '{}'".format(
        datasets[i]['type']))
  # TODO All of the software installation should be baked into an AMI instead,
  # this is pretty slow.
  return """#!/bin/bash
touch /home/ubuntu/.milano_keep_alive
chmod 777 /home/ubuntu/.milano_keep_alive
eval "while true; do find /home/ubuntu/.milano_keep_alive -mmin +5 -exec shutdown -h now {} + && sleep 10; done" &>/dev/null &disown;
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
groupadd docker
usermod -aG docker ubuntu
apt-get update
apt-get install -y awscli
""" + dataset_mounts +  """
apt-get install -y docker-ce
apt-get install -y nvidia-docker2
apt-get install -y nvidia-384
modprobe nvidia
systemctl restart docker
"""


class EC2InstanceManager:

  def __init__(self, count, key_name, private_key_path, region_name,
               spot_instances, datasets, iam_role, user_params):
    self._desired_count = count
    self._key_name = key_name
    self._private_key_path = private_key_path
    self._region_name = region_name
    self._spot_instances = spot_instances
    self._datasets = datasets
    self._iam_role = iam_role
    self._user_params = user_params
    self._instances = {}
    self._active_instance_ids = []
    self._thread = None
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._management_thread_main)
    self._thread.start()

  def _ami_for_region(self):
    # ubuntu 16.04 HVM SSD
    ami = {
      "us-east-1": "ami-5c150e23",
      "us-west-1": "ami-4d6a852e",
      "ap-northeast-1": "ami-e5b3ca08",
      "sa-east-1": "ami-01316f8dfe32c01e2",
      "ap-southeast-1": "ami-01fde464a811ead8a",
      "ca-central-1": "ami-4975f82d",
      "ap-south-1": "ami-0dcc9657fd6ff85bc",
      "eu-central-1": "ami-9fbfb174",
      "eu-west-1": "ami-0a8458313ef39d6f6",
      "cn-north-1": "ami-0510c868",
      "cn-northwest-1": "ami-f96c7b9b",
      "us-gov-west-1": "ami-3a4dd15b",
      "ap-northeast-2": "ami-09960a24a97b8087b",
      "ap-southeast-2": "ami-fc26869e",
      "us-west-2": "ami-529fb82a",
      "us-east-2": "ami-0eb3ba416aed8a6a4",
      "eu-west-2": "ami-52d12435",
      "ap-northeast-3": "ami-0d5d86281edca346f",
      "eu-west-3": "ami-0a06fa501d424d43f"
    }
    return ami.get(self._region_name, "")

  def _launch(self, launch_count):
    s = boto3.Session(region_name=self._region_name)
    iam_client = s.client('iam')
    iam = s.resource("iam")
    ec2 = s.resource("ec2")
    # unique role per dataset config
    if self._iam_role is None:
      self._iam_role, _ = get_or_create_role(
          "milano-" + sha1short(json.dumps(self._datasets)),
          self._datasets, iam, iam_client)
    profile_name, _ = get_or_create_instance_profile(
        self._iam_role + "-ip", self._iam_role, iam)
    sg_id = get_or_create_ssh_security_group("milano-worker-ssh", ec2)
    create_params = {
        'InstanceType': "p3.2xlarge",
        'ImageId': self._ami_for_region(),
        'KeyName': self._key_name,
        'MinCount': launch_count,
        'MaxCount': launch_count,
        'SecurityGroupIds': [sg_id],
        'BlockDeviceMappings': [{
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "DeleteOnTermination": True,
                # TODO expose this as a top level config option?
                "VolumeSize": 64
            }
        }],
        'TagSpecifications': [{
            'ResourceType': 'instance',
            'Tags': [{
                'Key': 'Name',
                'Value': 'milano-worker',
            }]
        }],
        "IamInstanceProfile": {
          "Name": profile_name,
        },
        # If ~/.milano_keep_alive isn't touched every 5 minutes, the instance
        # will auto terminate.
        'InstanceInitiatedShutdownBehavior': "terminate",
        'UserData': startup_script(self._datasets),
    }
    if self._spot_instances:
      create_params['InstanceMarketOptions'] = {
          'MarketType': 'spot',
          'SpotOptions': {
              'SpotInstanceType': 'one-time',
              'InstanceInterruptionBehavior': 'terminate'
          }
      }
    create_params.update(self._user_params)
    instance_resources = ec2.create_instances(**create_params)
    with self._lock:
      for instance_resource in instance_resources:
        self._instances[instance_resource.instance_id] = EC2Instance(
            instance_resource, "ubuntu", self._private_key_path)

  def active_instance_ids(self):
    with self._lock:
      return self._active_instance_ids.copy()

  def get_instance(self, instance_id):
    with self._lock:
      return self._instances[instance_id]

  def terminate(self):
    self._stop_event.set()
    self._thread.join()
    for _, instance in self._instances.items():
      instance.terminate()

  def _management_thread_main(self):
    while not self._stop_event.is_set():
      next_active_ids = []
      alive_count = 0
      for instance_id, instance in self._instances.items():
        if not instance.is_terminated():
          alive_count += 1
        if instance.is_running():
          instance.keep_alive()
          if instance.is_driver_working() and instance.datasets_present(
              self._datasets):
            next_active_ids.append(instance_id)
      if alive_count < self._desired_count:
        needed_count = self._desired_count - alive_count
        print("launching {count} EC2 instances and mounting datasets. this may take a few minutes...".
              format(count=needed_count))
        try:
          self._launch(needed_count)
        except Exception as e:
          print(e)
          pass
      with self._lock:
        self._active_instance_ids = next_active_ids
      time.sleep(10)


def get_or_create_ssh_security_group(name, ec2):
  try:
    groups = ec2.security_groups.filter(GroupNames=[name])
    for group in groups:
      return group.group_id
  except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] != 'InvalidGroup.NotFound':
      raise e
  # No existing security group, create one.
  sg = ec2.create_security_group(Description=name, GroupName=name)
  sg.authorize_ingress(
      IpProtocol='tcp', CidrIp='0.0.0.0/0', FromPort=22, ToPort=22)
  return sg.group_id

def get_or_create_role(name, datasets, iam, client):
  try:
    role = iam.Role(name)
    return role.role_name, role.role_id
  except Exception as e:
    pass
  role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps({
    "Statement": [{
      "Effect": "Allow",
      "Principal": {
        "Service": ["ec2.amazonaws.com"]
      },
      "Action": ["sts:AssumeRole"]
    }]
  }))
  for i in range(len(datasets)):
    bucket = bucket=datasets[i]['bucket']
    prefix = datasets[i].get('prefix', "")
    resp = client.put_role_policy(
      RoleName=name,
      PolicyName=name + "-policy-" + str(i),
      PolicyDocument=json.dumps({
          "Statement":[
            {
              "Action": ["s3:ListBucket"],
              "Effect": "Allow",
              "Resource": ["arn:aws:s3:::{}".format(bucket)],
              "Condition":{"StringLike":{"s3:prefix":["{}/*".format(prefix)]}}
            },
            {
              "Effect": "Allow",
              "Action": ["s3:Get*"],
              "Resource": ["arn:aws:s3:::{}/{}*".format(bucket, prefix)]
            }
          ]
        }
      )
    )
  return role.role_name, role.role_id

def get_or_create_instance_profile(name, role, iam):
  try:
    instance_profile = iam.InstanceProfile(name)
    return name, instance_profile.instance_profile_id
  except Exception as e:
    pass
  instance_profile = iam.create_instance_profile(InstanceProfileName=name)
  instance_profile.add_role(RoleName=role)
  # create_instances will fail if we try to use this instance profile too soon.
  time.sleep(10)
  return name, instance_profile.instance_profile_id

def sha1short(str):
  return hashlib.sha1(str.encode()).hexdigest()[:6]