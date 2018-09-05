# Copyright (c) 2018 NVIDIA Corporation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse


def build_images(results_dir, img_format="png"):
  num_steps = 100
  results_dict = {}

  for bench_dir in os.listdir(os.path.join(results_dir, 'results_csvs')):
    bench = bench_dir[6:]
    results_dict[bench] = {}
    cur_dir = os.path.join(results_dir, 'results_csvs', bench_dir)
    for dim_dir in os.listdir(cur_dir):
      dim = int(dim_dir[4:])
      results_dict[bench][dim] = {}
      cur_dir = os.path.join(results_dir, 'results_csvs', bench_dir, dim_dir)
      for algo_run_csv in os.listdir(cur_dir):
        algo, run = algo_run_csv.split('__')
        run = int(run[:-4])
        results_df = pd.read_csv(os.path.join(cur_dir, algo_run_csv),
                                 index_col=0)
        results_df = results_df.sort_values(by="job_id")["Result:"]
        # assuming all runs in same algo/bench/dim
        # tuple have same number of steps, i.e. results_df.shape[0] is the same
        num_evals = results_df.shape[0]
        if algo == '2x_random_search':
          num_evals /= 2
        plot_steps = np.linspace(10, num_evals, num_steps, dtype=np.int)
        values = np.empty(plot_steps.shape)
        for i, plot_step in enumerate(plot_steps):
          if algo == "2x_random_search":
            values[i] = results_df[:plot_step * 2].min()
          else:
            values[i] = results_df[:plot_step].min()

        if algo not in results_dict[bench][dim]:
          results_dict[bench][dim][algo] = [plot_steps, values[:, np.newaxis]]
        else:
          if not np.allclose(plot_steps, results_dict[bench][dim][algo][0]):
            raise RuntimeError(
              "Check that all runs for bench={}, dim={}, algo={} ".format(
                bench, dim, algo
              ) +
              "have the same number of samples"
            )
          results_dict[bench][dim][algo][1] = np.hstack((
            results_dict[bench][dim][algo][1],
            values[:, np.newaxis],
          ))

  img_dir = os.path.join(results_dir, 'results_images')
  os.makedirs(img_dir, exist_ok=True)

  joint_plot_steps = None
  steps_matched = True

  for bench, dims in results_dict.items():
    cur_dir = os.path.join(img_dir, 'bench-{}'.format(bench))
    os.makedirs(cur_dir, exist_ok=True)
    for dim, algos in dims.items():
      cur_dir = os.path.join(
        img_dir, 'bench-{}'.format(bench), 'dim-{}'.format(dim)
      )
      os.makedirs(cur_dir, exist_ok=True)
      for aggr_mode in ["first", "second"]:
        plt.figure()
        for algo, steps_results in algos.items():
          plot_steps, results = steps_results
          rs_results = results_dict[bench][dim]["random_search"][1]
          if algo == "random_search":
            assert np.allclose(results, rs_results)

          if joint_plot_steps is None:
            joint_plot_steps = plot_steps
          else:
            if not np.allclose(joint_plot_steps, plot_steps):
              steps_matched = False

          if aggr_mode == "first":
            means = np.mean(results, axis=1) / np.mean(rs_results, axis=1)
          else:
            means = np.mean(results / rs_results, axis=1)
            stds = np.std(results / rs_results, axis=1)

          if aggr_mode != "first":
            plt.errorbar(plot_steps, means, yerr=stds, errorevery=10,
                         label=algo, alpha=0.8, capsize=3)
          else:
            plt.plot(plot_steps, means, label=algo)

        plt.legend()
        plt.title("bench={}, dim={}".format(bench, dim))
        plt.xlabel("Number of evaluations")
        plt.ylabel("Improvement over random search")
        im_name = 'bench-{}__dim-{}__aggr_{}.{}'.format(
          bench, dim, aggr_mode, img_format,
        )
        full_path = os.path.join(cur_dir, im_name)
        plt.savefig(full_path, bbox_inches="tight")
        plt.close()

  if not steps_matched:
    print("Different benchmarks/dims have different number of steps, "
          "can't draw joint plots.")
    return

  means_b = {}
  nums_b = {}
  means_d = {}
  nums_d = {}
  means_all = {}
  nums_all = {}

  for bench, dims in results_dict.items():
    for dim, algos in dims.items():
      for algo, steps_results in algos.items():
        res = steps_results[1]
        rs_res = results_dict[bench][dim]["random_search"][1]

        if algo not in means_all:
          means_all[algo] = np.zeros(joint_plot_steps.shape[0])
          nums_all[algo] = 0

        if bench not in means_b:
          means_b[bench] = {}
          nums_b[bench] = {}
        if algo not in means_b[bench]:
          means_b[bench][algo] = np.zeros(joint_plot_steps.shape[0])
          nums_b[bench][algo] = 0

        if dim not in means_d:
          means_d[dim] = {}
          nums_d[dim] = {}
        if algo not in means_d[dim]:
          means_d[dim][algo] = np.zeros(joint_plot_steps.shape[0])
          nums_d[dim][algo] = 0

        means_b[bench][algo] += np.mean(res, axis=1) / np.mean(rs_res, axis=1)
        means_d[dim][algo] += np.mean(res, axis=1) / np.mean(rs_res, axis=1)
        means_all[algo] += np.mean(res, axis=1) / np.mean(rs_res, axis=1)

        nums_b[bench][algo] += 1
        nums_d[dim][algo] += 1
        nums_all[algo] += 1

  # drawing plots aggregated across dims
  for bench, algo_means in means_b.items():
    for algo, means in algo_means.items():
      means_cur = means / nums_b[bench][algo]
      plt.plot(joint_plot_steps, means_cur, label=algo)

    plt.legend()
    plt.title("bench={}, dim=all".format(bench))
    plt.xlabel("Number of evaluations")
    plt.ylabel("Improvement over random search")
    im_name = 'bench-{}__dim-all.{}'.format(bench, img_format)
    full_path = os.path.join(img_dir, im_name)
    plt.savefig(full_path, bbox_inches="tight")
    plt.close()

  # drawing plots aggregated across benchmarks
  for dim, algo_means in means_d.items():
    for algo, means in algo_means.items():
      means_cur = means / nums_d[dim][algo]
      plt.plot(joint_plot_steps, means_cur, label=algo)

    plt.legend()
    plt.title("bench=all, dim={}".format(dim))
    plt.xlabel("Number of evaluations")
    plt.ylabel("Improvement over random search")
    im_name = 'bench-all__dim-{}.{}'.format(dim, img_format)
    full_path = os.path.join(img_dir, im_name)
    plt.savefig(full_path, bbox_inches="tight")
    plt.close()

  # drawing one plot aggregated across everything
  for algo, means in means_all.items():
    means_cur = means / nums_all[algo]
    plt.plot(joint_plot_steps, means_cur, label=algo)

  plt.legend()
  plt.title("bench=all, dim=all".format(dim))
  plt.xlabel("Number of evaluations")
  plt.ylabel("Improvement over random search")
  im_name = 'bench-all__dim-all.{}'.format(img_format)
  full_path = os.path.join(img_dir, im_name)
  plt.savefig(full_path, bbox_inches="tight")
  plt.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--results_dir', required=True,
                      help='Directory with .csv files with results')
  parser.add_argument('--img_format', default="png",
                      help='Format to generate images in. '
                           'E.g. png, jpg, pdf, etc.')
  args = parser.parse_args()
  build_images(args.results_dir, args.img_format)
