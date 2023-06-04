# esmologs
Local homology search powered by the ESM-2 protein language model, foldseek, and hhsuite.

# Install

```
conda env create -f conda_env.yml
```

Will create a conda environment called `esmologs`





# all to all hhsearch
Starting from a fasta file, we'll use ESM2 to generate simulated hhm files and msas.
`single_seq_to_hmm.py -i input.fasta --model esm2_150M --out_fmt hhsearch --device cuda:0 --msa_outdir simulated_msas --msa_size 40 --profile_outdir profiles`

now we have to build indexes for the profiles, for the msas, and for the column state sequences. See the hhsearch documentation for details
[https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases](https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases)


## profiles
`ffindex_build -s db_hhm.ffdata db_hhm.ffindex profile_outdir`

## msas
`ffindex_build -s db_a3m.ffdata db_a3m.ffindex simulated_msas`

## column state sequences
`cstranslate -f -x 0.3 -c 4 -I a3m -i db_a3m -o db_cs219`

## run a search
`hhsearch -i profile_outdir/profile -o hits.hhr -d db`
or
`hhblits -i profile -d db -tags -n 1 -v 0 -o hits.hhr`

## search multiple profiles at once

## generate profiels from single sequences using ESM-2

# References

https://www.nature.com/articles/s41587-021-01179-w

https://github.com/facebookresearch/esm

https://github.com/steineggerlab/foldseek

https://github.com/soedinglab/hh-suite

https://github.com/EddyRivasLab/hmmer

https://github.com/althonos/pyhmmer

