# Copyright (c) 2018 NVIDIA Corporation
import argparse
import runpy
import os
import sys
import pandas as pd
sys.path.insert(0, "../")

from milano.exec_utils import ExecutionManager
from milano.backends import AzkabanBackend


def gen_bench_funcs(bench_type):
  funcs_dict = {
    "bbob": (prepare_bbob, finalize_bbob),
    "cifar10": (prepare_cifar10, finalize_cifar10),
  }
  return funcs_dict[bench_type]


def prepare_bbob(args, config):
  script_to_run = "bbob_func_eval.py"
  params_to_tune = {
    "func_name": {"type": "values", "values": [args.bench_name]}
  }
  for i in range(args.bench_dim):
    params_to_tune["x{}".format(i)] = {"min": -5, "max": 5, "type": "range"}

  if 'backend' not in config:
    backend_manager = AzkabanBackend(
      script_to_run=script_to_run,
      workers_config=[{"num_workers": args.num_workers, "env_vars": []}],
    )
  else:
    backend_manager = config['backend'](
      script_to_run=script_to_run,
      **config['backend_params'],
    )
  exp_params = {
    'script_to_run': script_to_run,
    'params_to_tune': params_to_tune,
    'result_pattern': "Result:",
    'objective': "minimize",
    'backend_manager': backend_manager,
    'sleep_time': 0.1,
    'wait_for_logs_time': 0.5,
  }
  return exp_params


def finalize_bbob(out_file):
  return


def prepare_cifar10(args, config):
  script_to_run = "cifar10_eval.sh"
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

  if 'backend' not in config:
    backend_manager = AzkabanBackend(
      script_to_run=script_to_run,
      workers_config=[{"num_workers": args.num_workers, "env_vars": []}],
    )
  else:
    backend_manager = config['backend'](
      script_to_run=script_to_run,
      **config['backend_params'],
    )
  exp_params = {
    'script_to_run': script_to_run,
    'params_to_tune': params_to_tune,
    'result_pattern': "Validation top-1:",
    'objective': "maximize",
    'backend_manager': backend_manager,
    'sleep_time': 5,
    'wait_for_logs_time': 10,
  }
  return exp_params


def finalize_cifar10(out_file):
  # making the resulting csv in the required format for image generation
  df = pd.read_csv(out_file, index_col=0)
  df['Validation top-1:'] = 1.0 - df['Validation top-1:']
  df = df.rename(index=str, columns={"Validation top-1:": "Result:"})
  df.to_csv(out_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Benchmarking parameters')
  parser.add_argument("--bench_name", required=True,
                      help="Benchmark name, e.g. sphere, rastrigin, etc.")
  parser.add_argument("--config", required=True, help="Config to use.")
  parser.add_argument("--bench_dim", type=int, default=2,
                      help="Benchmarking dimensionality")
  parser.add_argument("--num_evals", type=int, default=100,
                      help="Maximum number of evaluations "
                           "allowed for algorithm.")
  parser.add_argument("--num_workers", type=int, default=10,
                      help="Number of workers to use.")
  parser.add_argument("--verbose", type=int, default=1,
                      help="How much output to print. Setting to 0 mutes "
                           "the script, 3 is the highest level.")
  parser.add_argument("--output_file", required=False,
                      help="Output file to save the results to.")
  args = parser.parse_args()
  config = runpy.run_path(args.config)

  if args.bench_name != "cifar10":
    prepare_func, finalize_func = gen_bench_funcs("bbob")
  else:
    prepare_func, finalize_func = gen_bench_funcs("cifar10")

  exp_params = prepare_func(args, config)

  # a hack to make 2x_random_search work
  algo_name = os.path.basename(args.config)[:-3]  # stripping off .py
  if algo_name == '2x_random_search':
    args.num_evals *= 2

  search_algorithm = config['search_algorithm'](
    params_to_tune=exp_params['params_to_tune'],
    objective=exp_params['objective'],
    num_evals=args.num_evals,
    **config['search_algorithm_params'],
  )

  if args.output_file is not None:
    out_file = args.output_file
  else:
    out_file = "{}__{}.csv".format(args.bench_name, algo_name)

  exec_mng = ExecutionManager(
    backend_manager=exp_params['backend_manager'],
    search_algorithm=search_algorithm,
    res_pattern=exp_params['result_pattern'],
    objective=exp_params['objective'],
    constraints=[],
    output_file=out_file,
    verbose=args.verbose,
    sleep_time=exp_params['sleep_time'],
    wait_for_logs_time=exp_params['wait_for_logs_time'],
  )
  exec_mng.start_tuning()
  print("\nScore: {:.6f}".format(exec_mng.final_results.iloc[0, 0]))
  finalize_func(out_file)
