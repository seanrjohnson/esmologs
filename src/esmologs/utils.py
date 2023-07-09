#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing required libraries
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import importlib.resources as importlib_resources

def _open_if_is_name(filename_or_handle, mode="r"):
    """
        if a file handle is passed, return the file handle
        if a Path object or path string is passed, open and return a file handle to the file.
        returns:
            file_handle, input_type ("name" | "handle")
    """
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle, mode)
        input_type = "name"
    except TypeError:
        pass
    except Exception as e:
        raise(e)

    return (out, input_type)


def get_data():
    pkg = importlib_resources.files("threedifam")
    train_set_pep = list(SeqIO.parse(str(pkg / "data" / "train_set_pep.fasta"), "fasta"))
    train_set_3di = list(SeqIO.parse(str(pkg / "data" / "train_set_3di.fasta"), "fasta"))
    test_set_pep = list(SeqIO.parse(str(pkg / "data" / "test_set_pep.fasta"), "fasta"))
    test_set_3di =list(SeqIO.parse(str(pkg / "data" / "test_set_3di.fasta"), "fasta"))
    val_set_pep = list(SeqIO.parse(str(pkg / "data" / "val_set_pep.fasta"), "fasta"))
    val_set_3di = list(SeqIO.parse(str(pkg / "data" / "val_set_3di.fasta"), "fasta"))
    
    #get only the sequences 
    train_set_pep_seqs = parse_seqs_list(train_set_pep)
    train_set_3di_seqs = parse_seqs_list(train_set_3di)
    test_set_pep_seqs = parse_seqs_list(test_set_pep)
    test_set_3di_seqs = parse_seqs_list(test_set_3di)
    val_set_pep_seqs = parse_seqs_list(val_set_pep)
    val_set_3di_seqs = parse_seqs_list(val_set_3di)
    
    return train_set_pep_seqs,train_set_3di_seqs,test_set_pep_seqs,test_set_3di_seqs,val_set_pep_seqs,val_set_3di_seqs

def parse_seqs_list(seqs_list):
    seqs = []
    #get a list of sequences
    for record in seqs_list:
        seqs.append(str(record.seq))
        
    return seqs
             
def seq2onehot(seq_records):
    #declaring the alphabet
    # - represents padding
    alphabet = 'ACDEFGHIKLMNPQRSTVWYBXZJUO-'

    # define a mapping of chars to integers
    aa_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_aa = dict((i, c) for i, c in enumerate(alphabet))
    
    one_hot_representations = []
    integer_encoded_representations = []

    for seq in seq_records:
        # integer encode seq data
        integer_encoded = torch.tensor(np.array([aa_to_int[aa] for aa in seq]))
        # convert tensors into one-hot encoding 
        one_hot_representations.append(F.one_hot(integer_encoded, num_classes=27))

    ps = pad_sequence(one_hot_representations,batch_first=True)
    output = torch.transpose(ps,1,2)
    return output

def seq2integer(seq_records):
    #declaring the alphabet
    # - represents padding
    alphabet = 'ACDEFGHIKLMNPQRSTVWYBXZJUO-'

    # define a mapping of chars to integers
    aa_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_aa = dict((i, c) for i, c in enumerate(alphabet))
    
    integer_encoded_representations = []

    for seq in seq_records:
        # integer encode seq data
        integer_encoded = torch.tensor([aa_to_int[aa] for aa in seq])
        integer_encoded_representations.append(integer_encoded)
    
    #pad result to get equal sized tensors
    output = pad_sequence(integer_encoded_representations,batch_first=True,padding_value=26.0)
    return output

class CleanSeq():
    def __init__(self, clean=None):
        self.clean = clean
        if clean == 'delete':
            # uses code from: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
            deletekeys = dict.fromkeys(string.ascii_lowercase)
            deletekeys["."] = None
            deletekeys["*"] = None
            translation = str.maketrans(deletekeys)
            self.remove_insertions = lambda x: x.translate(translation)
        elif clean == 'upper':
            deletekeys = {'*': None, ".": "-"}
            translation = str.maketrans(deletekeys)
            self.remove_insertions = lambda x: x.upper().translate(translation)
            

        elif clean == 'unalign':
            deletekeys = {'*': None, ".": None, "-": None}
            
            translation = str.maketrans(deletekeys)
            self.remove_insertions = lambda x: x.upper().translate(translation)
        
        elif clean is None:
            self.remove_insertions = lambda x: x
        
        else:
            raise ValueError(f"unrecognized input for clean parameter: {clean}")
        
    def __call__(self, seq):
        return self.remove_insertions(seq)

    def __repr__(self):
        return f"CleanSeq(clean={self.clean})"




def parse_fasta(filename, return_names=False, clean=None, full_name=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input:
            filename: the name of a fasta file or a filehandle to a fasta file.
            return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
            clean: {None, 'upper', 'delete', 'unalign'}
                    if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
                    if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
                    if 'unalign' then convert to upper, delete ".", "*", "-"
            full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.
        output: sequences or (names, sequences)
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    out_seqs = list()
    out_names = list()
    (input_handle, input_type) = _open_if_is_name(filename)

    seq_cleaner = CleanSeq(clean)

    for line in input_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == ">":
            if full_name:
                name = line[1:]
            else:
                parts = line.split(None, 1)
                name = parts[0][1:]
            out_names.append(name)
            if (prev_name is not None):
                out_seqs.append(prev_seq)
            prev_len = 0
            prev_name = name
            prev_seq = ""
        else:
            prev_len += len(line)
            prev_seq += line
    if (prev_name != None):
        out_seqs.append(prev_seq)

    if input_type == "name":
        input_handle.close()
    
    if clean is not None:
        for i in range(len(out_seqs)):
            out_seqs[i] = seq_cleaner(out_seqs[i])

    if return_names:
        return out_names, out_seqs
    else:
        return out_seqs
    

def iter_fasta(filename, clean=None, full_name=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input:
            filename: the name of a fasta file or a filehandle to a fasta file.
            return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
            clean: {None, 'upper', 'delete', 'unalign'}
                    if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
                    if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
                    if 'unalign' then convert to upper, delete ".", "*", "-"
            full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.
        output: names, sequences
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    (input_handle, input_type) = _open_if_is_name(filename)

    seq_cleaner = CleanSeq(clean)

    for line in input_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == ">":
            if full_name:
                name = line[1:]
            else:
                parts = line.split(None, 1)
                name = parts[0][1:]
            if (prev_name is not None):
                yield prev_name, seq_cleaner(prev_seq)
            prev_len = 0
            prev_name = name
            prev_seq = ""
        else:
            prev_len += len(line)
            prev_seq += line
    if (prev_name != None):
        yield prev_name, seq_cleaner(prev_seq)
        

    if input_type == "name":
        input_handle.close()
