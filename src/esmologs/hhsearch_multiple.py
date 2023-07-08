"""
    Reads an hhsuite hhm file (containing one or more profiles) or a list of hhm files and runs hhblits of all of the profiles against a formatted hhsuite database.
"""
from threedifam import __version__
from typing import List
from collections.abc import Iterable
import subprocess
import argparse
import sys
import re
from dataclasses import dataclass, fields
from typing import Iterable

#####



@dataclass
class HHRResult:
    query_id: str
    query_length: int
    query_neff: float
    template_id: str
    template_length: int
    template_info: str
    template_neff: float
    query_ali: str
    query_start: int
    query_end: int
    template_ali: str
    template_start: int
    template_end: int  
    probability: float
    evalue: float
    score: float
    aligned_cols: int
    identity: float
    similarity: float
    sum_probs: float

def parse_hhr(input_string:str)-> Iterable[HHRResult]:
    query_id = ""
    query_length = 0
    query_neff = 0

    # reset the template variables
    template_id = None
    template_length = 0
    template_info = ""
    template_neff = 0
    query_ali = []
    query_start = 0
    query_end = 0
    template_ali = []
    template_start = 0
    template_end = 0
    probability = 0
    evalue = 0
    score = 0
    aligned_cols = 0
    identity = 0
    similarity = 0
    sum_probs = 0

    lines = input_string.split("\n")
    lines.reverse() # so we can use pop() to get the next line
    
    while lines:
        line = lines.pop()
        if line.startswith("Query"):
            query_id = line.split()[1]
        elif line.startswith("Match_columns"):
            query_length = int(line.split()[1])
        elif line.startswith("Neff"):
            query_neff = float(line.split()[1])
        elif line.startswith('>'):
            if template_id is not None:
                yield HHRResult(
                    query_id=query_id,
                    query_length=query_length,
                    query_neff=query_neff,
                    template_id=template_id,
                    template_length=template_length,
                    template_info=template_info,
                    template_neff=template_neff,
                    query_ali="".join(query_ali),
                    query_start=query_start,
                    query_end=query_end,
                    template_ali="".join(template_ali),
                    template_start=template_start,
                    template_end=template_end,
                    probability=probability,
                    evalue=evalue,
                    score=score,
                    aligned_cols=aligned_cols,
                    identity=identity,
                    similarity=similarity,
                    sum_probs=sum_probs
                )
                # reset the template variables
                template_id = None
                template_length = 0
                template_info = ""
                template_neff = 0
                query_ali = []
                query_start = 0
                query_end = 0
                template_ali = []
                template_start = 0
                template_end = 0
                probability = 0
                evalue = 0
                score = 0
                aligned_cols = 0
                identity = 0
                similarity = 0
                sum_probs = 0

            template_info = line[1:]
            template_id = template_info.split(' ')[0]
            
            line = lines.pop()
            parameters = re.findall(r'\S+[=]\S+', line)
            for parameter in parameters:
                name, value = parameter.split('=')
                if name == 'Probab':
                    probability = float(value)
                elif name == 'E-value':
                    evalue = float(value)
                elif name == 'Score':
                    score = float(value)
                elif name == 'Aligned_cols':
                    aligned_cols = int(value)
                elif name == 'Identities':
                    identity = float(value[:-1])
                elif name == 'Similarity':
                    similarity = float(value)
                elif name == 'Sum_probs':
                    sum_probs = float(value)
                elif name == 'Template_Neff':
                    template_neff = float(value)
        elif line.startswith("Q "):
            parts = line.split()
            if len(parts) == 5: # if the line has 5 parts, it's a query alignment line. We only want the first one. The second one (if present) is consensus.
                query_ali.append(parts[3])
                if len(query_ali) == 1:
                    query_start = int(parts[2])
                    query_length = int(parts[5][1:-1])
                query_end = int(parts[4])
                
            
                #find the last line with 5 parts in the next consecutive block of lines starting with 'T '
                t_seen = False
                parts = None
                while lines:
                    next_line = lines.pop()
                    next_line_parts = next_line.split()
                    if next_line.startswith('T '):
                        t_seen = True
                        if len(next_line_parts) == 5:
                            parts = next_line_parts
                    elif t_seen: # if we've seen a T line, and the next line is not a T line, we've reached the end of the block, so take the final T line with 5 parts.
                        template_ali.append(parts[3])
                        if len(template_ali) == 1:
                            template_start = int(parts[2])
                        template_end = int(parts[4])
                        template_length = int(parts[5][1:-1])
                        break
                    
    if template_id is not None:
        yield HHRResult(
            query_id=query_id,
            query_length=query_length,
            query_neff=query_neff,
            template_id=template_id,
            template_length=template_length,
            template_info=template_info,
            template_neff=template_neff,
            query_ali="".join(query_ali),
            query_start=query_start,
            query_end=query_end,
            template_ali="".join(template_ali),
            template_start=template_start,
            template_end=template_end,
            probability=probability,
            evalue=evalue,
            score=score,
            aligned_cols=aligned_cols,
            identity=identity,
            similarity=similarity,
            sum_probs=sum_probs
        )

def run_search(hhm_strings: List[bytes], dbpath:str, cores:int = 1, program:str = "hhblits") -> Iterable[HHRResult]:
    if program == "hhblits":
        run_opts = ["hhblits","-i", "stdin", "-d", dbpath, "-tags", "-n", "1", "-v", "0", "-cpu", str(cores), "-o", "stdout"]
    elif program == "hhsearch":
        run_opts = ["hhsearch","-i", "stdin", "-d", dbpath, "-v", "0", "-tags", "-cpu", str(cores), "-o", "stdout"]
    else:
        raise Exception("Unknown program: {}".format(program))
    

    for hhm in hhm_strings:
        hhblits_out = subprocess.Popen(run_opts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = hhblits_out.communicate(input=hhm)
        if hhblits_out.returncode != 0:
            raise Exception("hhblits failed with return code {}\n{}".format(hhblits_out.returncode, err.decode("utf-8")))
        out = out.decode("utf-8")
        # print(out)
        for rec in parse_hhr(out):
            yield rec


def parse_hhms(files:List[str]):
    """
        from a list of hhm files, reads each file and yields the hhm profiles one by one as strings.

    Args:
        files (List[str]): a list of paths to hhm files.
    """

    for f in files:
        lines_buffer = list()
        with open(f, 'rb') as fh:
            for line in fh:
                lines_buffer.append(line)
                if line.startswith(b'//'):
                    yield b''.join(lines_buffer)
                    lines_buffer = list()

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('-i', '--input', default=None, nargs="+", type=str, required=True, help="Input hhm files.") #TODO: allow fasta input files and call single_seq_to_hmm.

    parser.add_argument('-d', '--database', default=None, required=True, help="pre-formatted hhsuite database") 

    parser.add_argument('-o', '--output', default=None, required=False,
                        help="a profile hmm file. If not supplied then stdout.")

    parser.add_argument('--out_fmt', default="full", choices={"full","triple"}, 
                        help="Output in full, or triples (headerless tab-separated: query,hit,score) format. [full]")

    parser.add_argument('--program', default="hhblits", choices={"hhblits", "hhsearch"},)
    
    parser.add_argument('-k', type=float, default=float("inf"), help="Number of hits to return. [all]")

    parser.add_argument('--cpu', default=1, type=int,  
                        help="How many cores to use. [1]")

    params = parser.parse_args(argv)

    if params.output is None:
        out = sys.stdout
    else:
        out = open(params.output, "w")

    HHRResult_fields = [field.name for field in fields(HHRResult)]
    if params.out_fmt == "full":
        print("\t".join(HHRResult_fields), file=out)

    for i, rec in enumerate(run_search(parse_hhms(params.input), params.database, params.cpu, params.program)):
        if i >= params.k:
            break
        if params.out_fmt == "full":
            print("\t".join((str(getattr(rec, field)) for field in HHRResult_fields)), file=out)
        elif params.out_fmt == "triple":
            print(f"{rec.query_id}\t{rec.template_id}\t{rec.score}", file=out)


if __name__ == '__main__':
    main(sys.argv[1:])
