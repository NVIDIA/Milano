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
    "type": "values", "values": ["LSTM", "GRU", "QRNN"]
  },
  "--emsize": {
    "type": "values", "values": [128, 160, 196, 224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024]
  },
  "--nlayers": {
    "type": "values", "values": [1, 2, 3]
  },
  "--lr": {
    "type": "values", "values": [5, 10, 12, 15, 18, 19, 20, 22, 25, 30, 40],
  },
  "--nhid": {
    "type": "values", "values": [224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1280]
  },
  "--bptt": {
    "type": "values", "values": [16, 32, 48, 64, 96, 128, 160, 196]
  },
  "--clip": {
    "type": "range", "min": 0.1, "max": 2.0
  },
  "--dropout": {
    "type": "range", "min": 0.0, "max": 0.9
  },
  "--dropouth": {
    "type": "range", "min": 0.0, "max": 0.9s
  },
  "--dropoute": {
    "type": "range", "min": 0.0, "max": 0.9
  },
  "--dropouti": {
    "type": "range", "min": 0.0, "max": 0.9
  },
  "--wdrop": {
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
  "url": "http://10.110.40.104", # URL of your Azkaban UI
  "port": "8081", # Azkaban port. You should see Azkaban UI at url:port
  "username": "azkaban",
  "password": "azkaban",
  # If you are using Azkaban solo server on a single machine, set this to the number of GPUs you have
  # PRO TIP: If your workload isn't too heavy, you can allocate more than one worker per GPU as
  # is done below:
  "workers_config": [
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=0"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=1"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=2"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=2"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=2"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=2"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=3"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=3"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=3"]},
    {"num_workers": 1, "env_vars": ["CUDA_VISIBLE_DEVICES=3"]},
  ],
}

# specify search algorithm to use
search_algorithm = RandomSearch
search_algorithm_params = {
  "num_evals": 1000,
}