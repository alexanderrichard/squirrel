#!/usr/bin/python2.7

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('label_file', help='hmm state alignment to estimate the state prior from')
    parser.add_argument('prior_file', help='file to write the prior to')
    args = parser.parse_args()

    with open(args.label_file, 'r') as f:
        content = f.read().split('\n')[0:-1]
        n_states = int( content[1].split()[1] )
        prior = [0] * n_states
        for line in content[2:]:
            if not line == '#':
                prior[int(line)] += 1

    with open(args.prior_file, 'w') as f:
        f.write(str(n_states) + '\n')
        n_total = float(sum(prior))
        for p in prior:
            f.write(str( p/n_total ) + '\n')
