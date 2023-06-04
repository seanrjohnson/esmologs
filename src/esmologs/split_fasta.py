"""Partitions the sequences in a fasta file

Given an input sequence file, write new text files consisting of the sequences partitioned into groups.
Also prints the total number of records in the file.
"""
from Bio import AlignIO, Align, SeqIO
import argparse



def main(input_path, output_prefix, ids_per_partition):
    cds_count = 0
       
    max_digits = 3
    out_index = 0
    out_file = None

    for position, rec in enumerate(SeqIO.parse(input_path, "fasta")):
        cds_count += 1
        if position % ids_per_partition == 0:
            out_index += 1
            if out_file is not None:
                out_file.close()
            out_file = open(output_prefix + str(out_index).zfill(max_digits)+ ".fasta", "w")
        SeqIO.write([rec],out_file,"fasta")
    
    out_file.close()

    print(cds_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument('-i', '--input', default=None, type=str,
                       help="the fasta file to split.")
    parser.add_argument('--output_prefix', required=True, type=str,
                        help="Output files will be named [output_prefix][0-9]+.fasta")
                        
    overwrite_group = parser.add_mutually_exclusive_group(required=True)
    overwrite_group.add_argument('--ids_per_partition', type=int, default=None,
                        help="The number of ids to write to each partition.")
    params = parser.parse_args()


    main(params.input, params.output_prefix, params.ids_per_partition)
