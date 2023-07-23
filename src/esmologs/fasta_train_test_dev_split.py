#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:48:31 2022

@author: mpeshwa
"""

#Importing required libraries
from Bio import SeqIO
from tqdm import tqdm
import random
from itertools import chain
import math
import pandas as pd
import numpy as np
import sys
import argparse
import random

def main(pep, threedi, out, split, random_seed):
    # count records
    count = 0
    for rec in SeqIO.parse(pep, "fasta"):
        count += 1
    indexes = list(range(count))
    random.Random(random_seed).shuffle(indexes)
    
    train_size = int(len(indexes)*split[0]/100)
    test_size = math.ceil(len(indexes)*split[1]/100)
    val_size = len(indexes) - (train_size + test_size)
   
    train_indexes = set(indexes[0:train_size])
    test_indexes = set(indexes[train_size:(train_size+test_size)])
    val_indexes = set(indexes[(train_size+test_size):])
    
    assert len(train_indexes) + len(test_indexes) + len(val_indexes) == len(indexes)

    with open(out+".train.pep.fasta", "w") as train_pep, \
         open(out+".test.pep.fasta", "w") as test_pep, \
         open(out+".val.pep.fasta", "w") as val_pep, \
         open(out+".train.3di.fasta", "w") as train_3di, \
         open(out+".test.3di.fasta", "w") as test_3di, \
         open(out+".val.3di.fasta", "w") as val_3di:
        pep_in = SeqIO.parse(pep, "fasta")
        threedi_in = SeqIO.parse(threedi, "fasta")
        for i, pep_seq in enumerate(pep_in):
            threedi_seq = next(threedi_in)
            assert len(pep_seq) == len(threedi_seq)
            if i in train_indexes:
                SeqIO.write((pep_seq,),train_pep,"fasta")
                SeqIO.write((threedi_seq,),train_3di,"fasta")
            elif i in test_indexes:
                SeqIO.write((pep_seq,),test_pep,"fasta")
                SeqIO.write((threedi_seq,),test_3di,"fasta")
            elif i in val_indexes:
                SeqIO.write((pep_seq,),val_pep,"fasta")
                SeqIO.write((threedi_seq,),val_3di,"fasta")
            else:
                raise ValueError(f"index {i} not found in splits")
    print(f"Total sequences: {len(indexes)}")
    print(f"Train sequences: {len(train_indexes)}")
    print(f"Test sequences: {len(test_indexes)}")
    print(f"Validation sequences: {len(val_indexes)}")
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pep", help="peptide fasta input file. Expected to be in the same order as 3di input file", type=str)
    parser.add_argument("--3di", help="3di fasta input file. Expected to be in the same order as pep input file", type=str)
    parser.add_argument("--split", nargs=3, default=[90,5,5],  help="Train test validation  split percent, must add up to 100, default: 90 5 5", type=int)
    parser.add_argument("--out", help="output files will be named [out].[train|test|val].[pep|3di].fasta", type=str)
    parser.add_argument("--random_seed", default=72, help="seed to use for shuffling", type=int)
    params = parser.parse_args()

    if sum(params.split) != 100:
        raise ValueError("split must sum to 100")


    main(params.pep, getattr(params, '3di'), params.out, params.split, params.random_seed)
