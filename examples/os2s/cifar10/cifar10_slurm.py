# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import SLURMBackend
from milano.search_algorithms import RandomSearch

# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
script_to_run = "examples/os2s/cifar10/start_slurm.sh"

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

# BACKEND parameters. We will use SLURMBackend to run on DB Cluster
backend = SLURMBackend
backend_params = {
  "workers_config": {
    "num_workers": 2, # NUMBER OF SLURM *NODES* to run at a time.
    "partition": "batch", # PARTITION name
    "username": "okuchaiev", # CHANGE THIS to your username
    "key_path": "/home/okuchaiev/.ssh/id_rsa", # CHANGE THIS to your id_rsa path for pasword-less ssh to the cluster
    "entrypoint": "dbcluster", # CHANGE THIS to your cluster entrypoint name
  },
}

search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 10, # TOTAL EXPERIMENTS TO RUN. You can set it arbitrary high
}
