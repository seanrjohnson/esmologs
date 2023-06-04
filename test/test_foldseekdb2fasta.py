import tempfile
from esmologs import foldseek2fasta
import helpers

def test_foldseek2fasta_1(shared_datadir):
    with tempfile.TemporaryDirectory() as output_dir:
        #output_dir = "tmp"
        foldseek2fasta.main(["-i", str(shared_datadir / "af2_examples"), "-o", output_dir + "/out"])
        helpers.compare_files(shared_datadir / "af2_examples.3di.fasta", output_dir + "/out.3di.fasta")
        helpers.compare_files(shared_datadir / "af2_examples.pep.fasta", output_dir + "/out.pep.fasta")
