"""
    Reads a fasta file of individual protein sequences.
    Uses ESM2 to infer MSA column-wise probabilities.
    Writes a profile hmm in HHsearch 1.5 (.hhm) hmmer3 (.hmm) format.
"""
#TODO: figure out how to calculate cs219 files directly
#TODO: Apply Dirichlet priors to probabilities for hmmer3 output.

import torch
import argparse
import sys
import numpy as np
from esmologs import utils
from esmologs import __version__
import warnings
from textwrap import wrap
from scipy.stats import entropy
from pathlib import Path
import string
import tqdm
from collections import OrderedDict

#TODO: visualize the hmms (maybe as a LOGO, or something like Skyine http://skylign.org/)

MODELS=OrderedDict((("esm2_8M", "esm2_t6_8M_UR50D"),
         ("esm2_35M", "esm2_t12_35M_UR50D"),
         ("esm2_150M", "esm2_t30_150M_UR50D"),
         ("esm2_650M", "esm2_t33_650M_UR50D"),
         ("esm2_3B", "esm2_t36_3B_UR50D"),
         ("esm2_15B", "esm2_t48_15B_UR50D")))

# esm1_t34_670M_UR50S,
# esm1_t34_670M_UR50D,
# esm1_t34_670M_UR100,
# esm1_t12_85M_UR50S,
# esm1_t6_43M_UR50S,
# esm1b_t33_650M_UR50S,
# esm_msa1_t12_100M_UR50S,
# esm_msa1b_t12_100M_UR50S,
# esm1v_t33_650M_UR90S,
# esm1v_t33_650M_UR90S_1,
# esm1v_t33_650M_UR90S_2,
# esm1v_t33_650M_UR90S_3,
# esm1v_t33_650M_UR90S_4,
# esm1v_t33_650M_UR90S_5,
# esm_if1_gvp4_t16_142M_UR50,
# esm2_t6_8M_UR50D,
# esm2_t12_35M_UR50D,
# esm2_t30_150M_UR50D,
# esm2_t33_650M_UR50D,
# esm2_t36_3B_UR50D,
# esm2_t48_15B_UR50D,

class ProfileAlphabet():
    def __init__(self, symbols, background_frequencies, name):
        self.symbols = symbols
        self.background = background_frequencies
        self.name = name
    def __len__(self):
        return len(self.symbols)

#background frequencies taken from hhsearch (what hhmake puts as the background frequencies in hhm files)
# the order of the alphabet is lexical, which is the same as used in hmmer3 hmm files and in hhsuite hhm files.
# valid_aa_idx = sorted([model.alphabet.get_idx(tok) for tok in ESM_ALLOWED_AMINO_ACIDS])
def clean_name(name):
    clean_name.valid_chars = getattr(clean_name, 'valid_chars', "-_.()%s%s" % (string.ascii_letters, string.digits))
    return ''.join(c for c in name if c in clean_name.valid_chars)

def write_simulated_msa(probs, outpath, msa_size, sequence_alphabet): #TODO: I don't like that this is random sampling, I'd rather have a deterministic algorithm, like by multiplying the distribution by the number of desired sequences and then rounding.
    alphabet_size = len(sequence_alphabet.symbols)
    np.zeros((probs.shape[0], msa_size))

    chars = np.array([np.random.choice(alphabet_size, msa_size, True, probs[x,:]) for x in range(probs.shape[0])])
    with open(outpath, "w") as outfile:
        for i in range(chars.shape[1]):
            print(f">{i}", file=outfile)
            print("".join( [sequence_alphabet.symbols[symbol_index] for symbol_index in chars[:,i] ]), file=outfile)



def calculate_neff(probabilities): #TODO: maybe add a sliding window size, to do the calculation in a similar way as cstranslate?
    """calculates number of effective sequences for each row

    Args:
        probabilities (ndarray): [num_positions, alphabet_size]
    
    Returns:
        ndarray: [num_positions] number of effective sequences at each position
        float: average neff across the whole array
    """
    positional_neff = np.exp(entropy(probabilities, axis=1))
    sequence_neff = np.mean(positional_neff)

    return positional_neff, sequence_neff

def get_masked_probabilities(seq, model, model_alphabet, target_alphabet, device, fake_start=True, p=None): #TODO: should fake_start be default? Would it be better to append a <Mask> token and not resample it, than force an M?
    """_summary_

    Args:
        seq (str): protein sequence
        model : an ESM2 model instance
        model_alphabet : an ESM2 model_alphabet instance
        target_alphabet (ProfileAlphabet): _description_
        device: device where the model is stored, and where the batches should be sent.
        p (int, optional): Batch size. Increase to increase speed at the expense of memory. Defaults to None (run the entire sequence in a single batch).
        fake_start: append an M to the start of the sequence, so that the first AA isn't predicted to be M with high probability.
    Returns:
        ndarray: [input_length, 20]
    """
    #TODO: allow unmasked, or spaced masks, maybe by training a top model.
    # Adapted from code by Sergey Ovchinnikov for generating PSSMs.
    #'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C'
    if fake_start:
        seq = "M" + seq
    
    unmasked_batch = model_alphabet.get_batch_converter()([(None,seq)])[-1]
    indexes = [model_alphabet.tok_to_idx[c] for c in target_alphabet.symbols] # lookup table to go from target_alphabet index to model_alphabet index
    seq_len = len(seq)
    if p is None: p = seq_len
    with torch.no_grad():
        def forward(x):
            fx = model(x)["logits"][:,1:(seq_len+1),indexes]
            return fx
        logits = torch.zeros((seq_len,20))
        for n in range(0,seq_len,p):
            m = min(n+p,seq_len)
            x_h = torch.tile(torch.clone(unmasked_batch),(m-n,1))
            for i in range(m-n):
                x_h[i,n+i+1] = model_alphabet.mask_idx 
            fx_h = forward(x_h.to(device)) # Forward truncates off the <cls> token
            for i in range(m-n):
                logits[n+i] = fx_h[i,n+i] #
    softmax = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    
    if fake_start:
        return softmax[1:,:] #skip the first position if we faked it.
    return softmax

def get_masked_probabilities_interval(seq, model, model_alphabet, target_alphabet, device, fake_start=True, interval=None): #TODO: should fake_start be default? Would it be better to append a <Mask> token and not resample it, than force an M?
    """_summary_

    Args:
        seq (str): protein sequence
        model : an ESM2 model instance
        model_alphabet : an ESM2 model_alphabet instance
        target_alphabet (ProfileAlphabet): _description_
        device: device where the model is stored, and where the batches should be sent.
        interval (int): every nth position will be masked, and the masking will last n-rounds (instead of len(seq) rounds).
        fake_start: append an M to the start of the sequence, so that the first AA isn't predicted to be M with high probability.
    Returns:
        ndarray: [input_length, 20]
    """
    #TODO: allow unmasked, or spaced masks, maybe by training a top model.
    # Adapted from code by Sergey Ovchinnikov for generating PSSMs.
    #'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C'
    if fake_start:
        seq = "M" + seq
    
    unmasked_batch = model_alphabet.get_batch_converter()([(None,seq)])[-1]
    indexes = [model_alphabet.tok_to_idx[c] for c in target_alphabet.symbols] # lookup table to go from target_alphabet index to model_alphabet index
    seq_len = len(seq)
    if interval is None: interval = seq_len
    if interval > seq_len: interval = seq_len
    
    with torch.no_grad():
        def forward(x):
            fx = model(x)["logits"][:,1:(seq_len+1),indexes]  # [sequence, amino_acid_position, token_number] sequence is ':' because we want all of the sequences in the batch. amino acid position starts at 1 because we skip the start token, and goes to seq_len + 1 because we want the entire sequence. indexes is set to only the valid amino acid characters.
            return fx
        logits = torch.zeros((seq_len,20)) # to store the logits from the masked positions.

        batch = torch.tile(torch.clone(unmasked_batch),(interval,1)) # one copy of the sequence for each sampling pass.
        for step in range(interval):  # iterate through sampling passes
            for i in range(step, seq_len, interval): # shift the masked positions at each step.
                batch[step, i+1] = model_alphabet.mask_idx # the +1 is to skip the start token
        model_out = forward(batch.to(device))
        for step in range(interval):  # iterate through sampling passes
            for i in range(step, seq_len, interval): # shift the masked positions at each step.
                logits[i] = model_out[step, i]
    softmax = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            
    if fake_start:
        return softmax[1:,:] #skip the first position if we faked it.
    return softmax


def hmmer_transform(prob):
    """
        In hmmer all probabilities are stored as negative natural logs with five digits of precision to the right of the decimal,
        rounded. For example, a probability of 0.25 is stored as -logn(0.25) = 1.38629.
        The special case of a zero probability is stored as `*`.
    """

    if prob == 1:
        return 0
    elif prob == 0:
        return '*'
    elif prob > 0 and prob < 1:
        return -1 * np.log(prob)

def hmmer_num_format(num) -> str:
    if num == "*":
        return str(num)
    return str(round(float(num), 5)).ljust(7, '0')

def write_hmmer(out_handle, seq, probabilities, name, com="", filt="", date="", description=" ", alphabet=None):
    """
        Writes a probability matrix for a sequence to a hmmer3 hmm file.

        out_handle: an open file handle to write to
        seq: a sequence
        probabilities: a 2d numpy array of shape [len(seq), len(alphabet)] where entries are the probability of the character at that position of the sequence. So rows should sum to 1.
        name: the sequence name
        com: the command used to generate the sequence #TODO: maybe use argv?
        date: string to put for the date.
        description: string to put for the description (DESC) field.
        alphabet: a ProfileAlphabet object.
    """
    if alphabet is None:
        raise ValueError("alphabet must be specified")
    # date example: Sun Oct 23 12:38:48 2022
    # TODO: which parameters are actually necessary? Date?
    # TODO: add predicted secondary structure
    # if len(seq) > (float("inf")): #TODO: is there even a maximum size for hmm files?
    #     raise ValueError(f"Cannot make hhm file for sequence longer than 19999. {name}")
    # TODO: How to calculate stats for evalue calibration? right now they are just copied from an hmmbuild result file.
    # STATS LOCAL MSV       -9.8668  0.71232
    # STATS LOCAL VITERBI  -10.6217  0.71232
    # STATS LOCAL FORWARD   -4.3946  0.71232
    
    wrapped_seq = "\n".join(wrap(seq))
    # chr(9) is \t, \t cannot be used in f-expressions because \ is not allowed!
    # {chr(9).join([str(hhsearch_transform(x)) for x in alphabet.background])}
    average_probabilities = np.mean(probabilities, axis=0)
    formatted_alphabet_background = "          " + "  ".join( [hmmer_num_format(hmmer_transform(x)) for x in alphabet.background] )
    print(
f"""HMMER3/f [3.3.2 | Nov 2020]
NAME  {name}
LENG  {len(seq)}
ALPH  {alphabet.name}
CONS  yes
MAP   yes
STATS LOCAL MSV       -9.5376  0.71195
STATS LOCAL VITERBI  -10.7894  0.71195
STATS LOCAL FORWARD   -4.4127  0.71195
HMM          {"        ".join(alphabet.symbols)}
            m->m     m->i     m->d     i->m     i->i     d->m     d->d
  COMPO   {"  ".join( [hmmer_num_format(hmmer_transform(x)) for x in average_probabilities] )}
{formatted_alphabet_background}
          0.03438  3.78314  4.50549  0.61958  0.77255  0.00000        *""",
    file=out_handle
    )
    for i in range(probabilities.shape[0]):
        print(f"{str(i+1).rjust(7,' ')}   {'  '.join([hmmer_num_format(hmmer_transform(x)) for x in probabilities[i]])}{'      '}{str(i+1)} {seq[i]} - - -",
            file=out_handle)
        print(formatted_alphabet_background,
            file=out_handle)
        print("          0.03438  3.78314  4.50549  0.61958  0.77255  0.48576  0.95510", #TODO: is this optimal? Maybe can learn some of these from the model somehow?
            file=out_handle) 
    print("//\n", file=out_handle)


def hhsearch_transform(prob):
    if prob == 1:
        return 0
    elif prob == 0:
        return '*'
    elif prob > 0 and prob < 1:
        return int(round(-1000 * np.log2(prob),0))

def write_hhsearch(out_handle, seq, probabilities, name, com="", date="", description="", alphabet=None):
    """
        Writes a probability matrix for a sequence to an hhsearch hhm file.

        out_handle: an open file handle to write to
        seq: a sequence
        probabilities: a 2d numpy array of shape [len(seq), len(alphabet)] where entries are the probability of the character at that position of the sequence. So rows should sum to 1.
        name: the sequence name
        com: the command used to generate the sequence #TODO: maybe use argv?
        date: string to put for the date.
        description: not used
        alphabet: a ProfileAlphabet object.
    """

    if alphabet is None:
        raise ValueError("alphabet must be specified")

    # date example: Sun Oct 23 12:37:01 2022
    # TODO: which parameters are actually necessary? Date?
    # TODO: add predicted secondary structure
    if len(seq) > 19999:
        raise ValueError(f"Cannot make hhm file for sequence longer than 19999. {name}")
    positional_neff, sequence_neff = calculate_neff(probabilities)
    NEFF="{:.1f}".format(sequence_neff)
    wrapped_seq = "\n".join(wrap(seq))
    # chr(9) is \t, \t cannot be used in f-expressions because \ is not allowed!
    print(
f"""HHsearch 1.5
NAME  {name}
FAM   
COM   {com}
DATE  {date}
LENG  {len(seq)} match states, {len(seq)} columns in multiple alignment

FILT  
NEFF  {NEFF}
SEQ
>Consensus
{wrapped_seq.lower()}
>{name}
{wrapped_seq}
#
NULL   {chr(9).join([str(hhsearch_transform(x)) for x in alphabet.background])}
HMM    {chr(9).join(alphabet.symbols)}
       M->M	M->I	M->D	I->M	I->I	D->M	D->D	Neff	Neff_I	Neff_D
       0	*	*	0	*	0	*	*	*	*""",
    file=out_handle
    )
    for i in range(probabilities.shape[0]):
        aa_neff = int(positional_neff[i] * 1000)
        print(f"{seq[i]} {str(i+1).ljust(4,' ')} {chr(9).join([str(hhsearch_transform(x)) for x in probabilities[i]])}{chr(9)}{str(i+1)}",
                file=out_handle)
        print(f"       0	*	*	*	*	*	*	{aa_neff}	0	0	\n", #TODO: is this optimal? Maybe can learn some of these from the model somehow?
            file=out_handle) 
    print("//\n", file=out_handle)

def main(argv):
    parser = argparse.ArgumentParser(f"\nversion: {__version__}\n\n" + __doc__,)

    parser.add_argument('-i', '--input', default=None, required=True, help="Input file. An unaligned fasta.") #TODO: add option for specifying a table of extracting subsequences for the profiles.
    # parser.add_argument('--regions', default=None, 
    #                     help="headerless tab separated file with columns: sequence_name, region_start, region_end. sequence_name may be repeated, and should correspond to a sequence name in the input. coordinates are 1-indexed inclusive.")

    output_arggroup = parser.add_mutually_exclusive_group(required=False)
    output_arggroup.add_argument('-o', '--output', default=None, required=False,
                        help="a profile hmm file.")
    output_arggroup.add_argument('--profile_outdir', default=None, required=False,
                        help="Create a new directory with this name and write each profile as a separate file in this directory.")
    
    parser.add_argument('--out_fmt', default="hhsearch", choices={"hhsearch","hmmer"}, 
                        help="Output in hhsearch or hmmer format.")
    parser.add_argument('--msa_outdir', default=None, type=str, 
                         help="if supplied then write fake MSAs in fasta format to this directory (created if it doesn't exist). These fake msas are generated by sampling probabilities for each column independently, so they are useful for downstream tasks where columns are treated independently (like profile hmms), not for things like predicting structures/contact maps.")
    parser.add_argument('--msa_size', default=40, type=int,
                         help="Each fake MSA will have this many sequences.")
    parser.add_argument('--model', default="esm2_150M", choices=set(MODELS.keys()), # TODO: probably change default to a bigger model
                        help="Which ESM2 model to use for PSSM inferrence.")
    parser.add_argument('--device', default="cpu", 
                        help="Which device to use for model inferrence.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Increase this number to increase parallelization at the cost of memory (ignored when --mask_interval is set, in that case every sequence will be run in its own batch).")
    parser.add_argument('--mask_interval', default=None, type=int,
                        help="If set, then instead of masking positions one at a time, every nth position will be masked, and the masking will last n-rounds (instead of len(seq) rounds).")
    params = parser.parse_args(argv)
    
    device=params.device #"cuda:0"
    model, model_alphabet = torch.hub.load("facebookresearch/esm:main", MODELS[params.model])
    model.to(device)

    # TODO: maybe add support for 3Di or change background frequencies to SwissProt
    sequence_alphabet = ProfileAlphabet("ACDEFGHIKLMNPQRSTVWY", [0.076627179,  0.018866884,  0.053996137,  0.05978801,   0.034939433,  0.075415245,  0.036829356,  0.050485049,  0.059581159,  0.099925729,  0.021959667,  0.040107059,  0.045310839,  0.032644868,  0.051296351,  0.046617001,  0.071051061,  0.072644632,  0.012473412,  0.039418044],
                                        "amino")

    if params.msa_outdir is not None:
        msa_outdir = Path(params.msa_outdir)
        msa_outdir.mkdir(exist_ok=True)

    if params.profile_outdir is not None:
        profile_outdir = Path(params.profile_outdir)
        profile_outdir.mkdir(exist_ok=True)

    if params.profile_outdir is None and params.output is None:
        raise ValueError("Either --output or --output_dir must be specified.")
        
    names, seqs = utils.parse_fasta(params.input, clean='upper', return_names=True)
    
    if params.output is not None:
        out_handle = open(params.output, "w")

    for i in tqdm.trange(len(seqs)):
        if params.mask_interval is not None:
            probs = get_masked_probabilities_interval(seqs[i], model, model_alphabet=model_alphabet,
                                                target_alphabet=sequence_alphabet, device=device, interval=params.mask_interval)
        else:
            probs = get_masked_probabilities(seqs[i], model, model_alphabet=model_alphabet,
                                                target_alphabet=sequence_alphabet, device=device, p=params.batch_size)

       
        if params.msa_outdir is not None or params.profile_outdir is not None: #if the sequence name is becoming a file name we need to make sure it is a valid file name.
            name = clean_name(names[i])
        else:
            name = names[i]


        if params.output is None:
            out_handle = open(profile_outdir / name, "w")

        if params.out_fmt == "hhsearch":
            write_hhsearch(out_handle, seqs[i], probs, name, alphabet=sequence_alphabet)
        elif params.out_fmt == "hmmer":
            write_hmmer(out_handle, seqs[i], probs, name, alphabet=sequence_alphabet)
        else:
            raise ValueError(f"Output format {params.out_fmt} not supported.")
        if params.msa_outdir is not None:
            write_simulated_msa(probs,msa_outdir / name, params.msa_size, sequence_alphabet)
        if params.output is None:
            out_handle.close()
    if params.output is not None:
        out_handle.close()

if __name__ == '__main__':
    main(sys.argv[1:])
