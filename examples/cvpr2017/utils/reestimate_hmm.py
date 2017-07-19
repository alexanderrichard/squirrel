#!/usr/bin/python2.7

import argparse
import math


def parseLogFile(filename):
    transcripts = []
    lengths = []
    with open(filename, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for i in range(len(content)):
            if content[i].strip() == '<recognized>':
                transcripts.append( [ int(s.split(':')[0]) for s in content[i+1].split() ] )
                lengths.append( [ int(s.split(':')[1]) for s in content[i+1].split() ] )
    return transcripts, lengths


def reestimate_hmm(transcripts, lengths, n_classes, frames_per_state):
    class_instances = [0] * n_classes
    class_frames = [0] * n_classes
    for i in range(len(transcripts)):
        for j in range(len(transcripts[i])):
            class_instances[transcripts[i][j]] += 1
            class_frames[transcripts[i][j]] += lengths[i][j]
    start_states = [0]
    for c in range(n_classes):
        print '%d %d %d %d' % (c, class_frames[c] , class_instances[c], frames_per_state)
        start_states.append( int( float(class_frames[c]) / class_instances[c] / frames_per_state ) + start_states[-1] )
    hmm = []
    for cls in range(len(start_states)-1):
        hmm.append( range(start_states[cls], start_states[cls+1]) )
    return hmm


def generate_hmm_alignment(transcripts, lengths, hmm, alignment_file):
    n_frames = sum( sum(l) for l in lengths )
    n_states = sum( len(l) for l in hmm )
    n_sequences = len(transcripts)
    with open(alignment_file, 'w') as f:
        f.write('#sequencelabels\n' + str(n_frames) + ' ' + str(n_states) + ' ' + str(n_sequences) + '\n')
        for i in range(n_sequences):
            for j in range(len(lengths[i])):
                # align hmm states for label transcripts[i][j] to the j-th segment of the i-th video
                t = 0
                segment_length = float(lengths[i][j]) / len(hmm[transcripts[i][j]])
                while t < lengths[i][j]:
                    state = hmm[transcripts[i][j]][ int(t / segment_length) ]
                    f.write( str(state) + '\n' )
                    t = t+1
            f.write('#\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('realign_log_file', help='log file of the realignment as input')
    parser.add_argument('n_classes', help='the number of action classes', type=int)
    parser.add_argument('frames_per_state', help='the average number of frames per hmm state', type=int)
    parser.add_argument('hmm_file', help='filename for the new hmm file')
    parser.add_argument('hmm_alignment_file', help='filename for the new hmm alignment')
    args = parser.parse_args()

    transcripts, lengths = parseLogFile(args.realign_log_file)
    hmm = reestimate_hmm(transcripts, lengths, args.n_classes, args.frames_per_state)
    # save hmm states
    with open(args.hmm_file, 'w') as f:
        f.write(str(args.n_classes) + '\n')
        f.write( '\n'.join([ str(len(s)) for s in hmm ] ) + '\n')
    # generate new alignment
    generate_hmm_alignment(transcripts, lengths, hmm, args.hmm_alignment_file)
