#!/usr/bin/python2.7

import argparse
import numpy as np

def read_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        content = f.read().split('\n')[2:-1] # skip header
        for line in content:
            if not line == '#':
                labels.append( int(line) )
    return np.array(labels)


parser = argparse.ArgumentParser()
parser.add_argument('decoded_label_file')
parser.add_argument('ground_truth_label_file')
args = parser.parse_args()

recognized = read_labels(args.decoded_label_file)
reference  = read_labels(args.ground_truth_label_file)

print 'Mof: %f' % ( float(sum(recognized == reference)) / len(reference) )
