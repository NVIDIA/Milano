# Copyright (c) 2018 NVIDIA Corporation
import argparse
import runpy
from milano.exec_utils import ExecutionManager


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Tuning parameters')
  parser.add_argument("--config", required=True,
                      help="Path to the configuration file.")
  parser.add_argument("--output_file", default="results.csv",
                      help="Path to the output file containing results.")
  parser.add_argument("--verbose", type=int, default=1,
                      help="How much output to print. Setting to 0 mutes "
                           "the script, 3 is the highest level.")

  args = parser.parse_args()
  config = runpy.run_path(args.config)

  backend_manager = config['backend'](
    script_to_run=config['script_to_run'],
    **config['backend_params'],
  )
  search_algorithm = config['search_algorithm'](
    params_to_tune=config['params_to_tune'],
    params_to_try_first=config.get('params_to_try_first', None),
    objective=config['objective'],
    **config['search_algorithm_params'],
  )
  exec_mng = ExecutionManager(
    backend_manager=backend_manager,
    search_algorithm=search_algorithm,
    res_pattern=config['result_pattern'],
    objective=config['objective'],
    constraints=config.get('constraints', []),
    output_file=args.output_file,
    verbose=args.verbose,
  )
  exec_mng.start_tuning()
