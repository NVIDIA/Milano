# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import SLURMBackend
from milano.search_algorithms import RandomSearch

# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
# this should be on a driver machine
script_to_run = "examples/pytorch/wlm/start_wlm_slurm.sh"

# You can have this section to user-prespecify which configurations to explore first
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
  "--model": {
    "type": "values", "values": ["LSTM", "GRU"]
  },
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

constraints = [
  {"pattern": 'valid ppl   ',
   "range": [0, 310.0],
   "skip_first": 1,
   "formatter": lambda x: float(x[:-1])},
]

# specify result pattern used to parse logs
result_pattern = "valid ppl"

# maximize or minimize
objective = "minimize"

# BACKEND parameters. We will use SLURMBackend to run on DB Cluster
backend = SLURMBackend
backend_params = {
  "workers_config": {
    "num_workers": 2, # NUMBER OF SLURM *NODES* to run at a time.
    "partition": "batch", # PARTITION
    "username": "okuchaiev", # CHANGE THIS
    "key_path": "/home/okuchaiev/.ssh/id_rsa", # CHANGE THIS
    "entrypoint": "prom.nvidia.com", # CHANGE THIS
  },
}

search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 3,
}
