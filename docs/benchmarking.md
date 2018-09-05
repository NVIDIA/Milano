# Benchmarking

Milano has a number of simple benchmarks available to test new optimization
algorithms. Currently there are benchmarks adopted from
[BBOB workshop](https://numbbo.github.io/workshops/BBOB-2017/index.html) and
a simple cifar10 benchmark based on
[OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq).
For BBOB benchmakrs, we have "sphere", "elipsoidal", "rastrigin" and
"rosenbrock" function implemented.

To benchmark a new algorithm, you will need to write a simplified
configuration file (only specifying search algorithm and it's parameters) and
run `benchmark_algo.py` script from `benchmarking` directory. Note that if you
don't specify backend explicitly, you will need to have Azkaban launched with
default settings for BBOB benchmarks. For example, to test a Bayesian optimization
algorithm on a 4D "sphere" benchmark, run:

```
python benchmark_algo.py --bench_name=sphere --bench_dim=4 --config=benchmarking_configs/gp_search.py
```

To run all benchmarks and compare different algorithms you can use
`run_benchmarks.py` script which will evaluate all passed algorithms on all
passed benchmarks with a range of different dimensions and plot you nice
looking graphs with improvements all algorithms have against random search.
Note that it might take a long time to run full benchmarking. You can also
build images from existing results, if they are in the same format as generated
with `run_benchmarks.py` by running `build_images.py` and specifying directory
with results csv files.

There are additional parameters available for all scripts. Add `--help` flag
to see all options and their description.

For examples of what kind of output will be generated during benchmarking, have
a look at the [benchmarking_results](https://gitlab-dl.nvidia.com/dl-algo/Milano/tree/master/benchmarking/benchmarking_results). 
The same csv files are also generated during the usual run of the `tune.py` script.
You can also notice that for some of the images, there are 
"aggr_first" and "aggr_second" version of the same image. The "aggr_first" version
means that the each algorithm's result is **first** divided by the performance of
the random search and then aggregated across different runs. For the "aggr_second"
version the algorithms are first aggregated across different runs and **second**,
are divided by the aggregated performance of the random search.