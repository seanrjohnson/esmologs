"""

Converts foldseek databases into pairs of 3Di and amino acid fasta files.

"""

import argparse
import sys
from esmologs import __version__

def foldseekdb2fasta(input_basename, output_basename):
    with open(f"{input_basename}","rb") as aa_h, open(f"{input_basename}_ss","rb") as tdi_h, open(f"{input_basename}_h","rb") as header_h, \
         open(f"{output_basename}.3di.fasta", "wb") as tdi_out_h, open(f"{output_basename}.pep.fasta", "wb") as pep_out_h:
        for aa_line in aa_h:
            aa_line = aa_line.strip(b'\x00\x0a')
            tdi_line = tdi_h.readline().strip(b'\x00\x0a')
            header_line = header_h.readline().strip(b'\x00\x0a')
            if len(header_line) > 0: # skip sequences that don't have names (hopefully there aren't any of these), and also the last line
                tdi_out_h.write(b'>' + header_line + b'\n' + tdi_line + b'\n')
                pep_out_h.write(b'>' + header_line + b'\n' + aa_line + b'\n')

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('-i', '--input', default=None, required=True, help="Basename for input files. Expected input files will be: *, *_ss, *_h for the protein, 3di and headers, respectively.")

    parser.add_argument('-o', '--output', default=None, required=True,
                        help="Basename for output files. Output files will be named: *.3di.fasta and *.pep.fasta")
    params = parser.parse_args(argv)
    
    foldseekdb2fasta(params.input, params.output)



if __name__ == '__main__':
    main(sys.argv[1:])
