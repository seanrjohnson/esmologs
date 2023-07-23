import sys
import argparse
from Bio import SeqIO, BiopythonWarning, BiopythonParserWarning
import sys
import re
import random
import statistics
from Bio.SeqIO.InsdcIO import GenBankWriter
import warnings


def clean_name_func(name):
  """
    converts all: " " (space), ";" (semicolon), ":" (colon), "," (comma), "()" (parentheses), "'" (quote) characters to "_" in a string
  """

  bad_chars = " ;:,()'"
  chars = ["_" if x in bad_chars else x for x in name]
  return "".join(chars)

warnings.filterwarnings("ignore", category=BiopythonParserWarning)

### Monkey patching the genbank writer in biopython so that it can handle long peptide names.
### hopefully someday they will merge my pull request and this won't be necessary... :-(
    
def patched_write_first_line(self, record):
    """Write the LOCUS line (PRIVATE)."""
    locus = record.name
    if not locus or locus == "<unknown name>":
        locus = record.id
    if not locus or locus == "<unknown id>":
        locus = self._get_annotation_str(record, "accession", just_first=True)
    if len(locus) > 16:
        if len(locus) + 1 + len(str(len(record))) > 28:
            # Locus name and record length to long to squeeze in.
            # Per updated GenBank standard (Dec 15, 2018) 229.0
            # the Locus identifier can be any length, and a space
            # is added after the identifier to keep the identifier
            # and length fields separated
            # warnings.warn(
            #     "Increasing length of locus line to allow "
            #     "long name. This will result in fields that "
            #     "are not in usual positions.",
            #     BiopythonWarning,
            # )
            pass

    if len(locus.split()) > 1:
        raise ValueError(f"Invalid whitespace in {locus!r} for LOCUS line")
    if len(record) > 99999999999:
        # As of the GenBank release notes 229.0, the locus line can be
        # any length. However, long locus lines may not be compatible
        # with all software.
        # warnings.warn(
        #     "The sequence length is very long. The LOCUS "
        #     "line will be increased in length to compensate. "
        #     "This may cause unexpected behavior.",
        #     BiopythonWarning,
        # )
        pass

    # Get the molecule type
    mol_type = self._get_annotation_str(record, "molecule_type", None)
    if mol_type is None:
        raise ValueError("missing molecule_type in annotations")
    if mol_type and len(mol_type) > 7:
        # Deal with common cases from EMBL to GenBank
        mol_type = mol_type.replace("unassigned ", "").replace("genomic ", "")
        if len(mol_type) > 7:
            warnings.warn(f"Molecule type {mol_type} too long", BiopythonWarning)
            mol_type = "DNA"
    if mol_type in ["protein", "PROTEIN"]:
        mol_type = ""

    if mol_type == "":
        units = "aa"
    else:
        units = "bp"

    topology = self._get_topology(record)

    division = self._get_data_division(record)

    # Accommodate longer header, with long accessions and lengths
    if len(locus) > 16 and len(str(len(record))) > (11 - (len(locus) - 16)):
        name_length = locus + " " + str(len(record))

    # This is the older, standard 80 position header
    else:
        name_length = str(len(record)).rjust(28)
        name_length = locus + name_length[len(locus) :]
        assert len(name_length) == 28, name_length
        assert " " in name_length, name_length

    assert len(units) == 2
    assert len(division) == 3
    line = "LOCUS       %s %s    %s %s %s %s\n" % (
        name_length,
        units,
        mol_type.ljust(7),
        topology,
        division,
        self._get_date(record),
    )
    # Extra long header
    if len(line) > 80:
        splitline = line.split()
        if splitline[3] not in ["bp", "aa"]:
            raise ValueError(
                "LOCUS line does not contain size units at "
                "expected position:\n" + line
            )

        if not (
            splitline[3].strip() == "aa"
            or "DNA" in splitline[4].strip().upper()
            or "RNA" in splitline[4].strip().upper()
        ):
            raise ValueError(
                "LOCUS line does not contain valid "
                "sequence type (DNA, RNA, ...):\n" + line
            )

        self.handle.write(line)

    # 80 position header
    else:
        assert len(line) == 79 + 1, repr(line)  # plus one for new line

        # We're bending the rules to allow an identifier over 16 characters
        # if we can steal spaces from the length field:
        # assert line[12:28].rstrip() == locus, \
        #     'LOCUS line does not contain the locus at the expected position:\n' + line
        # assert line[28:29] == " "
        # assert line[29:40].lstrip() == str(len(record)), \
        #     'LOCUS line does not contain the length at the expected position:\n' + line
        assert line[12:40].split() == [locus, str(len(record))], line

        # Tests copied from Bio.GenBank.Scanner
        if line[40:44] not in [" bp ", " aa "]:
            raise ValueError(
                "LOCUS line does not contain size units at "
                "expected position:\n" + line
            )
        if line[44:47] not in ["   ", "ss-", "ds-", "ms-"]:
            raise ValueError(
                "LOCUS line does not have valid strand "
                "type (Single stranded, ...):\n" + line
            )
        if not (
            line[47:54].strip() == ""
            or "DNA" in line[47:54].strip().upper()
            or "RNA" in line[47:54].strip().upper()
        ):
            raise ValueError(
                "LOCUS line does not contain valid "
                "sequence type (DNA, RNA, ...):\n" + line
            )
        if line[54:55] != " ":
            raise ValueError(
                "LOCUS line does not contain space at position 55:\n" + line
            )
        if line[55:63].strip() not in ["", "linear", "circular"]:
            raise ValueError(
                "LOCUS line does not contain valid "
                "entry (linear, circular, ...):\n" + line
            )
        if line[63:64] != " ":
            raise ValueError(
                "LOCUS line does not contain space at position 64:\n" + line
            )
        if line[67:68] != " ":
            raise ValueError(
                "LOCUS line does not contain space at position 68:\n" + line
            )
        if line[70:71] != "-":
            raise ValueError(
                "LOCUS line does not contain - at position 71 in date:\n" + line
            )
        if line[74:75] != "-":
            raise ValueError(
                "LOCUS line does not contain - at position 75 in date:\n" + line
            )

        self.handle.write(line)


GenBankWriter._write_the_first_line = patched_write_first_line




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
    if seq_type == "genbank":
      if 'molecule_type' not in seqs[i].annotations:
        seqs[i].annotations['molecule_type'] = 'protein'
  
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
  parser.add_argument("--seq_type", default="fasta", help="")
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

  subset_fasta(input_handle, output_handle, args.length_lb, args.length_ub, args.seq_type, args.filter_by_aa, args.random_draw_number, args.length_stdev_filter, args.filter_by_start, args.random_seed, args.remove_stop_codon, args.clean_name)