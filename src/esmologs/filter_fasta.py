import sys
import argparse
from Bio import SeqIO, BiopythonWarning, BiopythonParserWarning
import sys
import re
import random
import statistics
import warnings


def clean_name_func(name):
  """
    converts all: " " (space), ";" (semicolon), ":" (colon), "," (comma), "()" (parentheses), "'" (quote) characters to "_" in a string
  """

  bad_chars = " ;:,()'"
  chars = ["_" if x in bad_chars else x for x in name]
  return "".join(chars)


def subset_fasta(input_handle, output_handle, length_lb, length_ub, seq_type, aa_filter, random_draw_number, length_stdev_filter, start_codon_filter, random_seed, remove_stop_codon, clean_name):
  if ((length_lb is None) or (length_lb < 0)):
    length_lb = 0
  if (length_ub is None):
    length_ub = sys.maxsize
  if ((length_ub < 1) or (length_ub < length_lb)):
    sys.exit("Invalid upper bound. Must be > 0, and > lower bound")

  out = list()
  
  seqs = list(SeqIO.parse(input_handle, seq_type))

  lengths = list()
  for i in range(len(seqs)):
    seqs[i]
  
    seqs[i] = seqs[i].upper()
    if str(seqs[i].seq)[-1] == "*":
      seqs[i] = seqs[i][:-1]  # delete stop codon if present
    lengths.append(len(seqs[i]))

  if length_stdev_filter is not None:
    mean_length = statistics.mean(lengths)
    median_length = statistics.median(lengths)
    stdev_length = statistics.stdev(lengths, mean_length)
    stdev_lb = median_length - (length_stdev_filter * stdev_length)
    stdev_ub = median_length + (length_stdev_filter * stdev_length)
    
    # if the stdev bounds are more restrictive than whatever other bounds have been set, then use the stdev bounds.
    if stdev_lb > length_lb:
      length_lb = stdev_lb
    if stdev_ub < length_ub:
      length_ub = stdev_ub

  for seq in seqs:
    keep = True

    length = len(seq)

    if start_codon_filter:
      if str(seq.seq)[0] != "M":
        keep = False


    if ((length < length_lb) or (length > length_ub)):
      keep = False
    
    if aa_filter:
      if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY\*]+$",str(seq.seq)):
        print(str(seq.seq))
        keep = False

    if remove_stop_codon:
      if str(seq.seq)[-1] == "*":
        seq.seq = seq.seq[:-1]
    
    if clean_name:
        seq.id = clean_name_func(seq.id) 
        

    if keep:
      out.append(seq)
  
  if random_draw_number is not None:
    if random_draw_number > len(out):
      print(f"After applying filters, only {len(out)} sequences remain, which is less than the random draw number, so all sequences will be returned.", file=sys.stderr)
    else:
      random.Random(random_seed).shuffle(out)
      out = out[0:random_draw_number]

  SeqIO.write(out, output_handle, seq_type)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", default=None)
  parser.add_argument("-o", default=None)
  parser.add_argument("--length_ub", type=int, default=None, help="discard sequences longer than this")
  parser.add_argument("--length_lb", type=int, default=None, help="discard sequences shorter than this")
  parser.add_argument("--length_stdev_filter", type=float, default=None, help="discard sequences with a length more than this many standard deviations from the median.")
  parser.add_argument("--random_draw_number", type=int, default=None, help="after applying other filters, randomly choose this many sequences to output.")
  parser.add_argument("--random_seed", type=int, required=False, default=None, help="Seed to use for the random number generator.")
  parser.add_argument('--filter_by_aa', action='store_true', default=False, help="if true then throw out sequences containing AAs not in the canonical 20")
  parser.add_argument('--filter_by_start', action='store_true', default=False, help="if true then throw out sequences that don't start with an M.")
  parser.add_argument('--remove_stop_codon', action='store_true', default=False, help="if true then remove the stop codon from the end of the sequence.")
  parser.add_argument('--clean_name', action='store_true', default=False, help="if true then make the name compatible with newick format.")


  args = parser.parse_args()

  if args.i is not None:
    input_handle=open(args.i, "r")
  else:
    input_handle = sys.stdin

  if args.o is not None:
    output_handle = open(args.o, "w")
  else:
    output_handle = sys.stdout

  subset_fasta(input_handle, output_handle, args.length_lb, args.length_ub, "fasta", args.filter_by_aa, args.random_draw_number, args.length_stdev_filter, args.filter_by_start, args.random_seed, args.remove_stop_codon, args.clean_name)