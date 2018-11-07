# Copyright (c) 2018 NVIDIA Corporation
from milano.backends import AzkabanBackend
from milano.search_algorithms import RandomSearch

# specify path to the script that is going to be tuned
# path has to be absolute or relative to tune.py script
# this should be on a driver machine
script_to_run = "examples/pytorch/wlm/start_wlm_azkaban.sh"

# specify the tunable parameters as cmd arguments and their possible ranges
params_to_tune = {
  "--model": {
    "type": "values", "values": ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]
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
   "skip_first": 4,
   "formatter": lambda x: float(x[:-1])},
]

# specify result pattern used to parse logs
result_pattern = "valid ppl"

# maximize or minimize
objective = "minimize"

# specify backend information and workers configuration
backend = AzkabanBackend
backend_params = {
  "url": "http://127.0.0.1", # URL of your Azkaban UI
  "port": "8081", # Azkaban port. You should see Azkaban UI at url:port
  "username": "azkaban",
  "password": "azkaban",
  # If you are using Azkaban solo server on a single machine, set this to the number of GPUs you have
  # PRO TIP: If your workload isn't too heavy, you can allocate more than one worker per GPU as
  # is done below:
  "workers_config": [
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
  ],
}

# specify search algorithm to use
search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 8,
}
