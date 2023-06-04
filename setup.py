import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

EXTENSIONS = []

setuptools.setup(
    name="esmologs",
    version=get_version("src/esmologs/__init__.py"),
    author="Sean Johnson",
    description="Local homology search powered by ESM-2 language model, foldseek, and hhsuite.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "https://github.com/seanrjohnson/esmologs/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={
        "esmologs":["data/*"]
    },
    #extensions=EXTENSIONS,
    scripts=[
        "src/esmologs/foldseek2fasta.py",
        "src/esmologs/mafft3di.py",
        "src/esmologs/hmmer_compare3di.py",
        "src/esmologs/count_msa_freqs.py",
        "src/esmologs/count_aa_freqs.py",
        "src/esmologs/single_seq_to_hmm.py",
        "src/esmologs/split_fasta.py",
        "src/esmologs/pred_pfamN.py",
        "src/esmologs/hhsearch_multiple.py",
        "src/esmologs/hmmer_top_hit.py",
    ],
    install_requires=[
        "setuptools>=60.7",
        "pytest",
        "pytest-datadir"
    ],
    python_requires=">=3.9"

)
