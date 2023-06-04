import argparse
from Bio.SeqIO import FastaIO
import sys
import re
import numpy as np

def accuracy(pred, actual):
    """
        pred:   predicted sequence
        actual: reference sequence
        
        returns: mismatches, total_characters
    """

    if len(pred) != len(actual):
        raise ValueError("Predicted and actual sequences must be the same length.")
    
    mismatches = 0
    total = 0
    for i in range(len(pred)):
        if pred[i] != actual[i]:
            mismatches += 1
        total += 1

    return mismatches, total
    
    

def calc_mismatch_matrix(pred, actual, symbol_to_index):
    """
        pred:   predicted sequence
        actual: reference sequence
        symbol_to_index: dictionary mapping symbols to indices
        
        returns: matrix of predicted tokens vs actual tokens [pred, actual]
    """
    pred_vs_actual = np.zeros((len(symbol_to_index), len(symbol_to_index)), dtype=int)

    for pos_i in range(len(pred)):
        actual_token = actual[pos_i]
        pred_token = pred[pos_i]
        pred_vs_actual[symbol_to_index[pred_token], symbol_to_index[actual_token]] += 1

    return pred_vs_actual




def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, required=True,
                    help="Fasta input file of predicted sequences.")
    parser.add_argument("-r", "--ref", type=str, default=None, required=True,
                    help="Fasta input file of reference sequences.")
    parser.add_argument("-m", "--matrix", type=str, default=None, required=False,
                    help="mismatch matrix output file.")

    

    params = parser.parse_args(argv)

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = dict((c, i) for i, c in enumerate(alphabet))

    mismatch_matrix = np.zeros((len(alphabet), len(alphabet)), dtype=int)

    with open(params.ref, "r") as ref_h, open(params.input, "r") as in_h:
        ref_iter = FastaIO.SimpleFastaParser(ref_h)
        in_iter = FastaIO.SimpleFastaParser(in_h)
        seqs_seen = 0
        total_mismatches = 0
        total_chars = 0
        while 1:
            try:
                ref_name, ref_seq = next(ref_iter)
                in_name, in_seq = next(in_iter)
            except StopIteration:
                break
            seqs_seen += 1
            if ref_name != in_name:
                raise ValueError("Reference and input files are not in the same order.")
            if len(ref_seq) != len(in_seq):
                raise ValueError("Reference and input sequences are not the same length.")
            mismatches, total = accuracy(in_seq, ref_seq)
            total_mismatches += mismatches
            total_chars += total
            if params.matrix:
                mismatch_matrix += calc_mismatch_matrix(in_seq, ref_seq, aa_to_int)
        print("Total sequences: {}".format(seqs_seen))
        print("Total mismatches: {}".format(total_mismatches))
        print("Total characters: {}".format(total_chars))
        print("Accuracy: {}".format(1 - (total_mismatches / total_chars)))

    if params.matrix:
        with open(params.matrix, "w") as out_h:
            out_h.write("\t" + "\t".join(alphabet) + "\n")
            for i in range(len(alphabet)):
                out_h.write(alphabet[i] + "\t" + "\t".join([str(x) for x in mismatch_matrix[i,:]]) + "\n")

if __name__ == '__main__':
    main(sys.argv[1:])