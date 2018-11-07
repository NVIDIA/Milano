import argparse
import os
import math
import random
import re

from matplotlib import pyplot as plt
import numpy as np

def get_names(params):
    names = []
    for part in params:
        names.append(part[2:part.find('=')])
    return names

def is_number(s):
    if is_float(s):
        return True, float(s)
    if is_int(s):
        return True, int(s)
    return False, s

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    return re.match("^\d+?\.\d+?$", s) is not None


def colorplot(results, xlabel, ylabel, values, benchmark, graph_folder):
    fig, ax = plt.subplots()

    scat = ax.scatter(results[xlabel], results[ylabel], c=values)
    fig.colorbar(scat, label=benchmark)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(graph_folder, xlabel + '_' + ylabel + '.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='visualizing Milano results')
    parser.add_argument('--file', type=str, default='results.csv',
                        help='location of the result file')
    parser.add_argument('--n', type=int, default=-1,
                        help='number of results to visualize. -1 for all')
    parser.add_argument('--wml', type=bool, default=True,
                        help='whether it has number of hidden units or nah')

    args = parser.parse_args()

    result_lines = open(args.file, 'r').readlines()
    benchmark = result_lines[0].split(',')[1]
    if len(result_lines) <= 1:
        raise ValueError('No experiments recorded')
    lines = [line.split(',') for line in result_lines[1:]]
    params = lines[0][2].split()
    
    param_names = get_names(params)
    raw_benchmarks = [float(line[1]) for line in lines]
    raw_benchmarks = [b if b < 300 else float('inf') for b in raw_benchmarks]
    print(len([b for b in raw_benchmarks if b < 300]))
    max_ = max([v for v in raw_benchmarks if v != float('inf')])
    benchmarks = [v if v != float('inf') else max_ * 1.2 + random.uniform(-max_ * 0.05, max_ * 0.05) for v in raw_benchmarks]

    results = {name: [] for name in param_names}
    for line in lines:
        for part in line[2].split():
            idx = part.find('=')
            results[part[2:idx]].append(is_number(part[idx + 1:])[1])

    
    samples = args.n if (args.n != -1 and args.n <= len(lines)) else len(lines)
    graph_folder = 'graphs_{}'.format(samples)

    os.makedirs(graph_folder, exist_ok=True)
    if args.wml:
        results['nhiddenunits'] = np.asarray(results['emsize']) * np.asarray(results['nlayers'])
        param_names.append('nhiddenunits')

    n_params = len(param_names)
    n_rows = math.ceil(n_params / 2)
    n_cols = 2

    for i, name in enumerate(param_names):
        print(name, n_params, i % 2 + 1, i // 2 + 1)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(results[name][:samples], benchmarks[:samples], 'bo')
        plt.title(name)
        plt.ylabel(benchmark)
    plt.savefig(os.path.join(graph_folder, 'single_params.png'))
    plt.close()

    for i, xlabel in enumerate(param_names):
        for j, ylabel in enumerate(param_names):
            if i >= j:
                continue
            colorplot(results, xlabel, ylabel, benchmarks, benchmark, graph_folder)




if __name__ == '__main__':
    main()