# Copyright (c) 2018 NVIDIA Corporation
from milano.search_algorithms import RandomSearch

# For benchmarks only search algorithm need to be specified.
# Optionally you can also specify a custom backend.
# If not specified, AzkabanBackend with default parameters and
# 10 identical workers will be used.

search_algorithm = RandomSearch
# note that you don't need to provide "num_evals" parameter,
# as it will be overwritten by benchmarking script
search_algorithm_params = {}
