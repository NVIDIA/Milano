## Contents
* [Quick Start](Quick_start.md)
* [How to add new search algorithm](how-to-add-new-search-algorithm.md)
* [Benchmarking](benchmarking.md)

## Getting started

You can use this toolkit with any script that has some
hyperparameters that need to be tuned. 

In order to use Milano you need to do the following things:

1. Prepare your script (or wrap your function in a script) such that it will
accept all hyperparameters as command
line arguments and print the resulting validation score in some "recognizable"
way (i.e. it should print the result after some pattern, e.g.
"Validation score: 0.23"). It should be possible to execute your script with
`./<script_name>` in the backend environment (see below), but there are no other
requirements for what your script should do or in what language it should
be written.

2. Prepare the Python configuration file. Look at the comments in
   `examples/os2s/cifar10_azkaban.py` or at other provided configs for
    examples on how to write the configuration file. In general it should
    specify the following:
  * Path to the script that need to be tuned.
  * Which parameters to tune and in which ranges. Supported parameter types are
     * "range": this parameter should be sampled uniformly from min to max values.
     * "log_range": this parameter should be sampled "logarithmically"
     from min to max values. This means that a uniform value will be
     sampled between [log(min), log(max)] and then it will be
     exponentiated.
     * "values": this parameter can be one of the supplied values.
  * Backend through which your script is going to be executed and workers configuration. Make sure
    that it is possible to execute your script in the backend, for example, you
    can run it from the environment where Azkaban is launched.

3. Start `tune.py` script to tune your hyperparameters and look at the
`results.csv` (can be changed with `--output_file` cmd argument) file for the
results. You don't need to wait for all jobs to
finish, `results.csv` will be updated iteratively on the go.
Run `python tune.py --help` to see the list of all available configurations.
Example command to train toy speech-to-text model with
[OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq) using Azkaban:

```
    python tune.py --config=examples/os2s/cifar10/cifar10_azkaban.py --verbose 3
```

## Config options

For the full list of supported config options see !!TODO!!

All of the SearchAlgorithm's support `num_evals` parameter which specifies how
many jobs you want to run (or equivalently, how many function evaluations you
want to do) and `random_seed`. Currently the following SearchAlgorithm's are
available:



## Requirements
Python >= 3.5 with packages listed in `requirements.txt` file.
To execute `run_benchmarks.py` or `build_images.py` additionally `matplotlib`
Python package is required.