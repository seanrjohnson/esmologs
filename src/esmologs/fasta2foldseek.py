"""

Converts pairs of 3Di and amino acid fasta files into foldseek databases.

"""
import argparse
import sys
from esmologs import __version__
from esmologs.utils import iter_fasta

def fasta2foldseek(aa_fasta, tdi_fasta, output_basename):
    # write the dbtypes
    
    #TODO: do I need to make the .lookup file?

    # pep dbtype
    with open(output_basename+".dbtype","wb") as pep_dbtype:
        pep_dbtype.write(b'\x00\x00\x00\x00')

    # 3Di dbtype
    with open(output_basename+"_ss.dbtype","wb") as tdi_dbtype:
        tdi_dbtype.write(b'\x00\x00\x00\x00')

    # headers dbtype
    with open(output_basename+"_h.dbtype","wb") as h_dbtype:
        h_dbtype.write(b'\x00\x0c\x00\x00')
        
    
    with open(f"{output_basename}","wb") as aa_h, open(f"{output_basename}_ss","wb") as tdi_h, open(f"{output_basename}_h","wb") as header_h, \
         open(f"{output_basename}.index","wb") as aa_index_h, open(f"{output_basename}_ss.index","wb") as tdi_index_h, open(f"{output_basename}_h.index","wb") as header_index_h, \
         open(f"{output_basename}.lookup","wb") as lookup_h, \
         open(tdi_fasta, "r") as tdi_in, open(aa_fasta, "r") as pep_in:
        tdi_iter = iter_fasta(tdi_in, full_name=True)

        seq_index = -1
        for pep_header, pep_seq in iter_fasta(pep_in, full_name=True):
            pep_name = pep_header.split(' ')[0]
            tdi_header, tdi_seq = next(tdi_iter)
            tdi_name = tdi_header.split(' ')[0]
            assert pep_header == tdi_header, f"Headers do not match: {pep_header} != {tdi_header}"
            assert pep_name == tdi_name, f"Names do not match: {pep_name} != {tdi_name}"
            assert len(pep_seq) == len(tdi_seq), f"Sequences do not match in length: {len(pep_seq)} != {len(tdi_seq)}"

            seq_index += 1
            
            # write the pep sequence
            aa_start_pos = aa_h.tell()
            aa_index_h.write(f"{seq_index}\t{aa_start_pos}\t".encode())
            aa_h.write(pep_seq.encode())
            aa_h.write(b'\x0a\x00')
            aa_index_h.write(f"{aa_h.tell() - aa_start_pos}\n".encode())

            # write the tdi sequence
            tdi_start_pos = tdi_h.tell()
            tdi_index_h.write(f"{seq_index}\t{tdi_start_pos}\t".encode())
            tdi_h.write(tdi_seq.encode())
            tdi_h.write(b'\x0a\x00')
            tdi_index_h.write(f"{tdi_h.tell() - tdi_start_pos}\n".encode())

            # write the header
            header_start_pos = header_h.tell()
            header_index_h.write(f"{seq_index}\t{header_start_pos}\t".encode())
            header_h.write(pep_header.encode())
            header_h.write(b'\x0a\x00')
            header_index_h.write(f"{header_h.tell() - header_start_pos}\n".encode())

            # write the lookup
            lookup_h.write(f"{seq_index}\t{pep_name}\t{seq_index}\n".encode())

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('--aa', default=None, required=True, help="Input amino acid fasta file.")
    parser.add_argument('--tdi', default=None, required=True, help="Input 3Di fasta file.")
    parser.add_argument('-o', '--output', default=None, required=True,
                        help="Basename for output files. output files will be: *, *_ss, *_h for the protein, 3di and headers, respectively.")
    params = parser.parse_args(argv)

    fasta2foldseek(params.aa, params.tdi, params.output)



if __name__ == '__main__':
    main(sys.argv[1:])
