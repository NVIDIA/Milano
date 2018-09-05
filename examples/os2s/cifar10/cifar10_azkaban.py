# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import AzkabanBackend
from milano.search_algorithms import RandomSearch


# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
script_to_run = "examples/os2s/cifar10/start_azk.sh"

params_to_try_first = {
  "--lr_policy_params/learning_rate": [0.01, 0.02, 0.03],
  "--lr_policy_params/power": [2.0, 2.0, 1.5],
  "--batch_size_per_gpu": [64, 64, 64],
  "--regularizer_params/scale": [0.0001, 0.0002, 0.0002],
  "--optimizer_params/momentum": [0.71, 0.72, 0.73],
}

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
  "--lr_policy_params/learning_rate": {
    "type": "log_range", "min": 0.00001, "max": 0.1
  },
  "--lr_policy_params/power": {
    "type": "values", "values": [0.5, 1.0, 2.0]
  },
  "--batch_size_per_gpu": {
    "type": "values", "values": [32, 64, 128, 256]
  },
  "--regularizer_params/scale": {
    "type": "log_range", "min": 0.00001, "max": 0.001
  },
  "--optimizer_params/momentum": {
    "type": "range", "min": 0.45, "max": 0.99
  },
}

# specify result pattern used to parse logs
result_pattern = "Validation top-1:"

# maximize or minimize
objective = "maximize"

# specify backend information and workers configuration
backend = AzkabanBackend
backend_params = {
  "url": "http://192.168.42.149", # URL of your Azkaban UI
  "port": "8081", # Azkaban port. You should see Azkaban UI at url:port
  "username": "azkaban",
  "password": "azkaban",
  # If you are using Azkaban solo server on a single machine, set this to the number of GPUs you have
  "workers_config": [
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
  ],
}

# specify search algorithm to use
search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 1024,
}
