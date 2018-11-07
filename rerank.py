import argparse

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
    if len(result_lines) <= 1:
        raise ValueError('No experiments recorded')
    lines = [line.split(',') for line in result_lines[1:]]
    raw_benchmarks = [float(line[1]) for line in lines]
    results = sorted(list(zip(raw_benchmarks, result_lines[1:])))
    with open(args.file + '.sorted', 'w') as f:
    	f.write(result_lines[0])
    	for val, line in results:
    		f.write(line)


if __name__ == '__main__':
	main()