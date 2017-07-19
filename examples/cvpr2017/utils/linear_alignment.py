#!/usr/bin/python2.7

import argparse


def get_sequence_lengths(labelfile):
    lengths = [0]
    with open(labelfile, 'r') as f:
        content = f.read().split('\n')[2:-2] # skip header and '#' at the end of the file
        for line in content:
            if line == '#':
                lengths.append(0)
            else:
                lengths[-1] += 1
    return lengths


def get_transcripts(transcriptfile):
    transcripts = [ [] ]
    with open(transcriptfile, 'r') as f:
        content = f.read().split('\n')[2:-2] # skip header and '#' at the end of the file
        for line in content:
            if line == '#':
                transcripts.append([])
            else:
                transcripts[-1] += [int(line)]
    return transcripts


def get_hmm(hmm_file):
    start_states = [0]
    with open(hmm_file, 'r') as f:
        content = f.read().split('\n')[1:-1]
        for line in content:
            start_states.append( int(line) )
        for i in range(1, len(start_states)):
            start_states[i] += start_states[i-1]
    hmm = []
    for cls in range(len(start_states)-1):
        hmm.append( range(start_states[cls], start_states[cls+1]) )
    return hmm


def linear_alignment(transcripts, lengths, hmm, outputfile):
    n_hmm_states = hmm[-1][-1] + 1
    n_frames_total = sum(lengths)
    n_sequences = len(lengths)
    with open(outputfile, 'w') as f:
        f.write('#sequencelabels\n' + str(n_frames_total) + ' ' + str(n_hmm_states) + ' ' + str(n_sequences) + '\n')
        for i in range(len(lengths)):
            # generate hmm state transcript for i-th video
            hmm_transcript = []
            for label in transcripts[i]:
                hmm_transcript += hmm[label]
            # generate linear alignment
            t = 0
            segment_length = float(lengths[i]) / len(hmm_transcript)
            while t < lengths[i]:
                state = hmm_transcript[ int(t / segment_length) ]
                f.write( str(state) + '\n' )
                t = t+1
            f.write('#\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_labels', help='framewise action labels, only used to determine the length of each video')
    parser.add_argument('transcripts', help='action label transcripts')
    parser.add_argument('hmm_file', help='hmm model file')
    parser.add_argument('output_label_file', help='write the output alignment to this file')
    args = parser.parse_args()

    lengths = get_sequence_lengths(args.ground_truth_labels)
    transcripts = get_transcripts(args.transcripts)
    hmm = get_hmm(args.hmm_file)
    linear_alignment(transcripts, lengths, hmm, args.output_label_file)

