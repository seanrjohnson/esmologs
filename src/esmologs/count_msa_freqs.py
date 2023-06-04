"""
    Reads one or more fasta MSA files, writes the column-wise letter counts. 
    That is:
        for every column in every input MSA, the number of each amino acid is counted and written to a line in the output file.
        Thus the number of lines in the output file is equal to the total number of columns over all of the input MSAs.
    The overall frequency of each amino acid is printed to stdout.
"""

import argparse
import sys
import numpy as np
from esmologs import utils
from esmologs import __version__
import warnings
ALPHABET="ACDEFGHIKLMNPQRSTVWY"
CHAR_TO_IDX = {ALPHABET[i]:i for i in range(len(ALPHABET))}


def count_msa_freqs(input, output, param_lb):
    total_freq = np.zeros(len(ALPHABET))
    with open(output,"w") as outfile:
        for fname in input:
            msa = utils.parse_fasta(fname, clean='upper')
            if len(msa) == 0:
                warnings.warn(f"MSA: {fname} contains no sequences!")
                continue
            for pos_idx in range(len(msa[0])):
                line_ct = [0] * len(ALPHABET)
                for seq_idx in range(len(msa)):
                    char = msa[seq_idx][pos_idx]
                    if char in CHAR_TO_IDX:
                        line_ct[CHAR_TO_IDX[char]] += 1
                
                if np.sum(line_ct) >= param_lb:
                    total_freq += np.array(line_ct)
                    print(" ".join([str(x) for x in line_ct]), file=outfile)
    total_freq = total_freq/np.sum(total_freq)
    print( " ".join([str(x) for x in total_freq]) )

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('-i', '--input', nargs="+", default=None, required=True, help="Input files. Aligned fastas.")

    parser.add_argument('-o', '--output', default=None, required=True,
                        help="frequency table output file.")
    parser.add_argument("--lb", default=10, type=int, help="Skip columns with fewer than this many non-gap rows. [10]")
    params = parser.parse_args(argv)
    
    count_msa_freqs(params.input, params.output, params.lb)



if __name__ == '__main__':
    main(sys.argv[1:])
