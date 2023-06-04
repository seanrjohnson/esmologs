"""
    Reads one or more fasta files, counts the number of amino acids of each type in the file.

"""

import argparse
import sys
import numpy as np
from esmologs import utils
from esmologs import __version__
import warnings
ALPHABET="ACDEFGHIKLMNPQRSTVWY"
CHAR_TO_IDX = {ALPHABET[i]:i for i in range(len(ALPHABET))}


def count_msa_freqs(input):
    total_freq = np.zeros(len(ALPHABET))
    for fname in input:
        msa = utils.parse_fasta(fname, clean='upper')
        if len(msa) == 0:
            warnings.warn(f"MSA: {fname} contains no sequences!")
            continue
        for seq in msa:
            for char in seq:
                if char in CHAR_TO_IDX:
                    total_freq[CHAR_TO_IDX[char]] += 1
    total_freq = total_freq/np.sum(total_freq)
    print( " ".join([str(x) for x in ALPHABET]) )
    print( " ".join([str(x) for x in total_freq]) )

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)
    parser.add_argument('-i', '--input', nargs="+", default=None, required=True, help="Input files. Aligned fastas.")
    params = parser.parse_args(argv)
    count_msa_freqs(params.input)

if __name__ == '__main__':
    main(sys.argv[1:])
