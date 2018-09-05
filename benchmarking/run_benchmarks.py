# Copyright (c) 2018 NVIDIA Corporation
import os
import argparse
import subprocess
import sys

from build_images import build_images


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs_dir', default="benchmarking_configs",
                      help='Directory with all benchmarking configs '
                           'that should be evaluated.')
  parser.add_argument('--output_dir', default='benchmarking_results',
                      help="Directory to store results in.")
  parser.add_argument('--reuse_results', dest='reuse_results',
                      action='store_true',
                      help="Whether to reuse existing results")
  parser.add_argument('--num_evals', default=100, type=int,
                      help="Maximum number of evaluations allowed "
                           "for each algorithm")
  parser.add_argument('--num_algo_runs', default=5, type=int,
                      help="Number of times each algorithm is "
                           "going to be applied to a single benchmark. "
                           "Results will be later averaged.")
  parser.add_argument('--benchmarks', nargs='+', required=False,
                      help='Benchmarks to evaluate algorithms on. By default'
                           'all available benchmarks will be used')
  parser.add_argument('--dims', nargs='+', required=False, type=int,
                      help='Dimensions of benchmarks (each benchmark is '
                           'going to be tested with all dims listed here)')
  parser.add_argument('--python_bin', default="python",
                      help="python3 executable, e.g. python or python3")
  args = parser.parse_args()

  configs = [os.path.join(args.configs_dir, cfg)
             for cfg in os.listdir(args.configs_dir)]

  if "random_search" not in [os.path.basename(cfg)[:-3] for cfg in configs]:
    raise ValueError("random_search.py config has to be present "
                     "for full benchmarking")

  if args.benchmarks is None:
    benchmarks = ['sphere', 'elipsoidal', 'rastrigin', 'rosenbrock', 'cifar10']
  else:
    benchmarks = args.benchmarks
  num_algo_runs = args.num_algo_runs
  num_evals = args.num_evals
  if args.dims is None:
    dims = [4, 8, 16, 32, 64]
  else:
    dims = args.dims

  base_dir = args.output_dir

  if not args.reuse_results:
    if os.path.exists(base_dir):
      print("Directory {} already exists, did you want ".format(base_dir) +
            "to specify --reuse_results flag?")
      sys.exit(1)

  os.makedirs(os.path.join(base_dir, 'results_csvs'), exist_ok=True)

  for config in configs:
    for bench in benchmarks:
      for i in range(num_algo_runs):
        cur_dims = dims if bench != "cifar10" else [5]
        for dim in cur_dims:
          algo_name = os.path.basename(config)[:-3]

          dir_name = os.path.join(
            base_dir, 'results_csvs',
            'bench-{}'.format(bench), 'dim-{}'.format(dim),
          )
          os.makedirs(dir_name, exist_ok=True)

          out_name = os.path.join(
            dir_name,
            "{}__{}.csv".format(algo_name, i)
          )
          if args.reuse_results and os.path.isfile(out_name):
            continue
          run_cmd = "{} benchmark_algo.py --bench_name={} ".format(
            args.python_bin, bench
          )
          run_cmd += "--config={} --num_evals={} ".format(config, num_evals)
          run_cmd += "--bench_dim={} --output_file={}".format(dim, out_name)
          print('Testing "{}" on "{}" with dim={}, run #{}'.format(
            algo_name, bench, dim, i
          ))
          subprocess.run(run_cmd, shell=True, check=True)

  build_images(args.output_dir)
