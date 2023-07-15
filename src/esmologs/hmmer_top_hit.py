"""
uses hmmer3 to find the top hit for each sequence in a fasta file

"""
import argparse
import sys
import os
import subprocess
from esmologs import __version__
from Bio.SeqIO import FastaIO
import tempfile
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import warnings
import logging
import importlib.resources as importlib_resources

def iter_hmm(hmm_file):
    out_list = []
    name = ""
    for line in hmm_file:
        out_list.append(line.strip())
        if line.startswith("NAME  "):
            name = line.split()[1]
        if line[0:2] == "//":
            yield name, "\n".join(out_list)
            out_list = []



# {command: (swap_input, swap_output)}
swap_io = {"phmmer": (False, False), "hmmscan": (True, False), "hmmsearch": (False, False)}

def get_top_hit(domtblout_name):
    """
    gets the top hit from a domtblout file
    returns the name of the query sequence, the name of the hit sequence, and the score
    """

    top_hit = None
    with open(domtblout_name, "r") as domtblout:
        for line in domtblout:
            if line[0] == "#":
                continue
            else:
                top_hit = line
                break
    
    if top_hit is None:
        return None
    else:
        top_hit = top_hit.split()
        return top_hit[3], top_hit[0], top_hit[7]
        
        
def worker(name, sequence, database, command_line_args, max_on_fail=True, swap_input=False):
    """
        returns the top hit for a sequence
    """

    logger = logging.getLogger(__name__)

    tmp_file_extension = ".fasta"
    if command_line_args[0] == "hmmsearch":
        tmp_file_extension = ".hmm"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile_name = os.path.join(tmpdirname, "tmp" + tmp_file_extension)
        with open(tmpfile_name, "w") as tmp:
            if command_line_args[0] == "hmmsearch":
                #print(sequence)
                print(sequence, file=tmp)
            else:
                print(f">{name}\n{sequence}", file=tmp)
        domtblout_name = os.path.join(tmpdirname, "tmp.tblout")
        
        query_db = [tmp.name, database]

        if swap_input:
            query_db = query_db[::-1]

        subprocess.run(command_line_args + ["--domtblout", domtblout_name] + query_db)

        out =  get_top_hit(domtblout_name)

        if out is None:
            logger.info(f"Could not find a hit for {name}. Trying with --max, and lowered thresholds.")

        if out is None and max_on_fail:
            subprocess.run(command_line_args + ["--max", "-Z", "1", "--domZ", "1", "-E", "1000000", "--domE", "1000000", "--domtblout", domtblout_name] + query_db)
            out =  get_top_hit(domtblout_name)
        
            if out is None:
                logger.info(f"After --max, still could not find a hit for {name}. Skipping.")

    return out
    
    #subprocess.run(["hmmsearch", "--tblout", params.output, "--noali", "--cpu", "1", "--notextw", "--domtblout", params.output + ".domtblout", params.database, params.input])

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('-i', '--input', default=None, required=True, help="Input fasta file.")
    parser.add_argument('-o', '--output', default=None, required=True, help="Output file.")
    parser.add_argument('-d', '--database', default=None, required=True, help="Database to search.")
    parser.add_argument('-t', '--threads', default=1, type=int, required=False, help="Number of threads to use.")
    #parser.add_argument('--swap_output', default=False, action="store_true", help="Swap the query and target output columns. Default is False.")
    # parser.add_argument('-k',  default=1, required=False, type=int, help="number of top hits to keep for each sequence. Default is 1.")
    parser.add_argument("--args", default=None, required=False, help="Extra arguments to pass to the command. Must be a string of space separated arguments. Put two sets of quotes around it.")
    parser.add_argument('--command', default="hmmsearch", required=False, help="Command to use. Default is hmmsearch.")
    parser.add_argument('--tdi', action='store_true', default=False, required=False, help="If set, use 3Di_[command], and for phmmer, use --mxfile [3Di substitution matrix file]")
    parser.add_argument('--log_file', default=None, required=False, help="Log file to write to. Default is stdout.")
    params = parser.parse_args(argv)
    

    # Configure the logging handler to write to the specified log file
    if params.log_file is not None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename=params.log_file)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    input_handle = open(params.input, "r")
    if params.command == "hmmsearch":
        input_iter = iter_hmm(input_handle)
    else:
        input_iter = FastaIO.SimpleFastaParser(input_handle)
    
    extra_args = []
    if params.args is not None:
        extra_args = params.args.strip("\"' ").split()
    
    swap_input, swap_output = swap_io[params.command]

    pkg = importlib_resources.files("esmologs")
    tdi_matrix_file = str(pkg / "data" / "mat3di.out")

    if params.tdi:
        params.command = "3Di_" + params.command
        if params.command == "3Di_phmmer":
            extra_args += ["--mxfile", tdi_matrix_file]

    command_line_args = [params.command, "--cpu", "1", "--noali", "--notextw", "-o", "/dev/null"] + extra_args
    
    pool = ThreadPool(params.threads)
    with open(params.output, "w") as out:
        for result in tqdm(pool.imap_unordered(lambda x: worker(x[0], x[1], params.database, command_line_args, swap_input=swap_input), input_iter)):
            if result is not None:
                if swap_output:
                    result = [result[1], result[0], result[2]]
                print("\t".join(result), file=out)
    input_handle.close()
    
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main(sys.argv[1:])