# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import AWSBackend
from milano.search_algorithms import RandomSearch


# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
script_to_run = "examples/os2s/cifar10/start_aws.sh"

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
  "--lr_policy_params/learning_rate": {
    "type": "log_range", "min": 0.00001, "max": 0.1
  },
  "--lr_policy_params/power": {
    "type": "values", "values": [0.5, 1.0, 2.0]
  },
  "--batch_size_per_gpu": {
    "type": "values", "values": [32, 64, 128, 256, 512, 1024]
  },
  "--regularizer_params/scale": {
    "type": "log_range", "min": 0.00001, "max": 0.001
  },
  "--optimizer_params/momentum": {
    "type": "range", "min": 0.5, "max": 0.99
  },
}

# specify result pattern used to parse logs
result_pattern = "Validation top-1:"

# maximize or minimize
objective = "maximize"

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
    "docker_image_name": "tensorflow/tensorflow:1.9.0-gpu-py3",
    # "iam_role": "..." # if omitted, a role with read access to the dataset bucket/prefixes is created.
    "datasets": [
      {
        "type": "s3",
        "bucket": "milano-test-data",
        "prefix": "cifar-10",
        "mount": "/data",
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
