import tempfile
from esmologs import single_seq_to_hmm
import helpers

def test_foldseek2fasta_1(shared_datadir):
    with tempfile.TemporaryDirectory() as output_dir:
        # output_dir = "tmp"
        single_seq_to_hmm.main(["-i", str(shared_datadir / "SOD_examples.fasta"), "-o", output_dir + "/out.hhm", "--model", "esm2_8M"])
        helpers.compare_files(shared_datadir / "SOD_examples.hhm", output_dir + "/out.hhm")
