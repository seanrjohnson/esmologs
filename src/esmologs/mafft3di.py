"""
mafft3di.py

Run MAFFT with the 3di matrix. Directly passes arguments to MAFFT, but specifies the matrix as the 3di matrix

Running with:
--maxiterate 1000 --globalpair
is a good idea for alignments of fragments of single domains

--------------------------------------------------------------------------

"""

import sys
from esmologs import __version__
import subprocess
import importlib.resources as importlib_resources


def main(argv):
    pkg = importlib_resources.files("threedifam")
    matrix_file = str(pkg / "data" / "mat3di.out")
    if ("-h" in argv) or ("--help" in argv) or (len(argv) == 0):
        print(__doc__, file=sys.stderr)
    subprocess.run(["mafft","--aamatrix", matrix_file] + argv)




if __name__ == '__main__':
    main(sys.argv[1:])
