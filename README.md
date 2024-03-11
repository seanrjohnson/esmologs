# esmologs
Local homology search powered by the ESM-2 protein language model, Foldseek, HMMER, and hhsuite.

See also the paper associated with this code:

Johnson, Sean R., et al. “Sensitive Remote Homology Search by Local Alignment of Small Positional Embeddings from Protein Language Models.” eLife, vol. 12, Feb. 2024. elifesciences.org, https://doi.org/10.7554/eLife.91415.2.



# Install with conda

```
conda env create -f conda_env.yml
```

Creates a conda environment called `esmologs`

Activate it with
```
conda activate esmologs
```

For HMMER3Di, see the separate repository:

[https://github.com/seanrjohnson/hmmer3di](https://github.com/seanrjohnson/hmmer3di)

Note that HMMER3Di will work attrociously on amino acids, and, apparently, not much differently from standard HMMER3 on 3Di sequences.

# Data analysis
Code for generating the plots and tables in the manuscript can be found here:
[https://github.com/seanrjohnson/esmologs/blob/main/notebooks/Data_analysis.ipynb](https://github.com/seanrjohnson/esmologs/blob/main/notebooks/Data_analysis.ipynb)

# Examples

Note that we developed and tested these scripts on an Nvidia A100 with 40 Gb of VRAM. Some scripts may use a lot of VRAM, particularly for long protein sequences.

## Generate HH-suite .hhm files and HMMER .hmm files from fasta protein files

The following command will create two new directories: "generated_msa" and "generated_hhm". For every protein sequence in peptides.fasta, an MSA with 40 sequences sampled from the ESM-2 3B masked probabilities will be created in the generated_msa directory, and an HH-suite-compatible .hhm will be created in the generated_hhm directory.
```bash 
single_seq_to_hmm.py -i peptides.fasta --model esm2_3B --out_fmt hhsearch --device cuda:1 --msa_outdir generated_msa --msa_size 40 --profile_outdir generated_hhm  --mask_interval 7
```

Generate hmmer profiles

```bash
# use Dirichlet priors
mkdir -p hmm
for filename in generated_msa/*
do
    stem=$( basename "${filename}" )
    hmmbuild -n $stem --amino hmm/${stem}.hmm $filename
done

# use raw frequencies
mkdir -p hmm_pnone
for filename in generated_msa/*
do
    stem=$( basename "${filename}" )
    hmmbuild --pnone -n $stem --amino hmm_pnone/${stem}.hmm $filename
done
```

(Note that you can generate a profile similar to that generated from `hmmbuild --pnone` directly from `single_seq_to_hmm.py` using the option `--out_fmt hmmer`, but those profiles perform much worse than profiles built from `hmmbuild` with default settings on the sampled MSAs)


## Build HH-suite database

We have to build indexes for the profiles, for the msas, and for the column state sequences. See the hhsearch documentation for details
[https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases](https://github.com/soedinglab/hh-suite/wiki#hh-suite-databases)


```bash
ffindex_build -s db_hhm.ffdata db_hhm.ffindex generated_hhm
ffindex_build -s db_a3m.ffdata db_a3m.ffindex generated_msa
cstranslate -f -x 0.3 -c 4 -I a3m -i db_a3m -o db_cs219
```

## Run a search with HH-suite

For a single query

`hhsearch -i profile.hhm -o hits.hhr -d db`

or

`hhblits -i profile.hhm -d db -tags -n 1 -v 0 -o hits.hhr`

For multiple queries

```bash
#concatenate multiple .hhm files into an .hhm file containing multiple profiles.
cat generated_hhm/*.hhm > queries.hhm
# use the hhsearch_multiple.py script from esmologs
hhsearch_multiple.py -i queries.hhm -d db --cpu 8 --program hhblits --out_fmt triple -k 100 > top_100_hits.tsv
```


## Predict 3Di sequences from protein sequences


```bash
# download the ESM-2 3B 3Di weights
wget https://zenodo.org/record/8174960/files/ESM-2_3B_3Di.pt

# predict 3Di sequences from peptide sequences in peptides.fasta
predict_from_ESM2_to_3Di.py --weights ESM-2_3B_3Di.pt -i peptides.fasta -o threedi.fasta
```


## Create a Foldseek database from predicted 3Di sequences

`fasta2foldseek.py --aa peptides.fasta --tdi threedi.fasta -o foldseek_db`


## Run foldseek searches

```bash 
# this is for an all-to-all comparison of foldseek_db, you could also use different databases for query and target
foldseek search foldseek_db foldseek_db aln tmpFolder
foldseek convertalis foldseek_db foldseek_db aln all_to_all.tsv --format-output query,target,bits
```


## Train ESM-2 3B 3Di starting from the ESM-2 3B pre-trained weights


Download the training data and make the splits
```bash
#Download UniProt50 foldseek database
foldseek databases Alphafold/UniProt50 afdb50  tmp

#extract peptide and 3di fastas
foldseek2fasta.py -i afdb50 -o afdb50

#filter proteins less than 120 aa or greater than 1000 aa
filter_fasta.py -i afdb50.pep.fasta -o afdb50.pep.120_1000aa.fasta
filter_fasta.py -i afdb50.3di.fasta -o afdb50.3di.120_1000aa.fasta

#split afdb50 into random splits
fasta_train_test_dev_split.py --pep afdb50.pep.120_1000aa.fasta --3di afdb50.3di.120_1000aa.fasta --split 90 5 5 --out afdb50.120_1000aa
```

Train the CNN
```bash
train_ESM2_to_3Di.py --train afdb50.120_1000aa.train --val afdb50.120_1000aa.val --device cuda:1 --epochs 1 --validation_interval 50 --validation_batches 10 --batch_size 15 --checkpoint_dir 3di_afdb_cnn --log 3di_afdb_cnn.log
```

It converges to a loss of about 1.37 after about 2 hours (well before the end of the first epoch). Kill it and start again with the last ESM-2 layer unfrozen.

```bash
train_ESM2_to_3Di.py --train afdb50.120_1000aa.train --val afdb50.120_1000aa.val --device cuda:1 --epochs 1 --validation_interval 100 --validation_batches 10 --batch_size 10 --checkpoint_dir 3di_afdb_cnn_unfreeze --log 3di_afdb_cnn_unfreeze.log --starting_weights 3di_afdb_cnn/0_000000000014.pt --esm_layers_to_train 36
```

It converges after about 11 hours at loss of 1.09.

# References

https://www.nature.com/articles/s41587-021-01179-w

https://github.com/facebookresearch/esm

https://github.com/steineggerlab/foldseek

https://github.com/soedinglab/hh-suite

https://github.com/EddyRivasLab/hmmer

https://github.com/althonos/pyhmmer

