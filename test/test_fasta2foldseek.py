import tempfile
from esmologs import fasta2foldseek
import helpers

def test_fasta2foldseek_1(shared_datadir):
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = "tmp"
        fasta2foldseek.main(["--aa", str(shared_datadir / "af2_examples.pep.fasta"), "--tdi", str(shared_datadir / "af2_examples.3di.fasta"), "-o", output_dir + "/out_foldseek"])
        #pep
        helpers.compare_files(shared_datadir / "af2_examples", output_dir + "/out_foldseek")
        helpers.compare_files(shared_datadir / "af2_examples.index", output_dir + "/out_foldseek.index")
        
        #3Di
        helpers.compare_files(shared_datadir / "af2_examples_ss", output_dir + "/out_foldseek_ss")
        helpers.compare_files(shared_datadir / "af2_examples_ss.index", output_dir + "/out_foldseek_ss.index")

        #headers
        helpers.compare_files(shared_datadir / "af2_examples_h", output_dir + "/out_foldseek_h")
        helpers.compare_files(shared_datadir / "af2_examples_h.index", output_dir + "/out_foldseek_h.index")
        
        #lookup
        helpers.compare_files(shared_datadir / "af2_examples.lookup", output_dir + "/out_foldseek.lookup")
        #helpers.compare_files(shared_datadir / "af2_examples.pep.fasta", output_dir + "/out.pep.fasta")


