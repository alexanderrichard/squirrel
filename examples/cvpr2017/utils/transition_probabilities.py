#!/usr/bin/python2.7

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('label_file', help='hmm state alignment to estimate transition probabilities from')
    parser.add_argument('transition_probability_file', help='write estimated transition probabilities to this file')
    args = parser.parse_args()

    with open(args.label_file, 'r') as f:
        content = f.read().split('\n')[0:-1]
        n_states = int( content[1].split()[1] )
        loops = [0] * n_states
        total = [0] * n_states
        for i in range(2, len(content)-1): # skip header and last '#' symbol
            if content[i] == '#':
                continue
            total[int(content[i])] += 1
            if content[i] == content[i+1]:
                loops[int(content[i])] += 1

    # save loop probabilities
    with open(args.transition_probability_file, 'w') as f:
        f.write(str(n_states) + '\n')
        for state in range(n_states):
            f.write(str( float(loops[state]) / max(1, total[state]) ) + '\n')
