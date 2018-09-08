# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import AWSBackend
from milano.search_algorithms import RandomSearch


# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
script_to_run = "examples/pytorch/wlm/start_wlm_aws.sh"

# These configurations will be tried first
params_to_try_first = {
  "--model": ["LSTM", "GRU"],
  "--emsize": [1504, 1504],
  "--nlayers": [2, 2],
  "--lr": [20, 25],
  "--bptt": [35, 35],
  "--clip": [0.25, 0.35],
  "--dropout": [0.2, 0.2],
}

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
  "--emsize": {
    "type": "values", "values": [256, 512, 650, 1024, 1504, 2048]
  },
  "--nlayers": {
    "type": "values", "values": [1, 2, 3]
  },
  "--lr": {
    "type": "values", "values": [10, 20, 5, 30, 25, 40],
  },
  "--nhid": {
    "type": "values", "values": [256, 512, 650, 1024, 1504, 2048]
  },
  "--bptt": {
    "type": "values", "values": [15, 20, 30, 35, 45]
  },
  "--clip": {
    "type": "range", "min": 0.1, "max": 2.0
  },
  "--dropout": {
    "type": "range", "min": 0.0, "max": 0.9
  },
}

# specify result pattern used to parse logs
result_pattern = "valid ppl"

# maximize or minimize
objective = "minimize"

# specify backend information and workers configuration
backend = AWSBackend
backend_params = {
   # TODO maybe automate the creation of a keypair if one isn't supplied
  "config": {
    "num_workers": 1,
    "spot_instances": False,
    "key_name": "milano-test",
    "private_key_path": "/home/okuchaiev/.aws/milano-test.pem", # FILL THIS IN WITH YOUR .pem FILE
    "region_name": "us-west-2",
    "docker_image_name": "pytorch/pytorch:0.4_cuda9_cudnn7",
    # "iam_role": "..." # if omitted, a role with read access to the dataset bucket/prefixes is created.
    "datasets": [
      {
        "type": "s3",
        "bucket": "milano-test-data",
        "prefix": "cifar-10",
        "mount": "/workdir",
      },
    ],
    "instance_params": {
      "InstanceType": "p3.2xlarge",
    }
  }
}

# specify search algorithm to use
search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 3,
}
