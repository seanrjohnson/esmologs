#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#importing required libraries
import torch
import torch.nn as nn
#from utils import get_data, SeqsDataset
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from esm import pretrained
import esm
from typing import Union, Optional
from Bio import SeqIO
from esm.model.esm2 import ESM2


class ESM2_to_3Di(nn.Module):
    models_cfg = {"esm2_t48_15B_UR50D": {"num_layers": 48, "embed_dim": 5120, "attention_heads": 40, "token_dropout": True},
                  "esm2_t36_3B_UR50D": {"num_layers": 36, "embed_dim": 2560, "attention_heads": 40, "token_dropout": True},
                  "esm2_t33_650M_UR50D": {"num_layers": 33, "embed_dim": 1280, "attention_heads": 20, "token_dropout": True},
                  "esm2_t30_150M_UR50D": {"num_layers": 30, "embed_dim": 640, "attention_heads": 20, "token_dropout": True},
                  "esm2_t12_35M_UR50D": {"num_layers": 12, "embed_dim": 480, "attention_heads": 20, "token_dropout": True},
                  "esm2_t6_8M_UR50D": {"num_layers": 6, "embed_dim": 320, "attention_heads": 20, "token_dropout": True}
                }

    def __init__(self, esm_model="esm2_t36_3B_UR50D", weights=None):
        """
        Args:
            esm_model (str, optional): The ESM model to use. Defaults to "esm2_t36_3B_UR50D".
            weights (dict, optional): The weights to initialize the model with. Defaults to None.
                                        If None, then pretrained ESM2 weights are downloaded and used, and the top CNN layer is left with default weights.
        """
        super(ESM2_to_3Di, self).__init__()
        # load the model
        if weights is None:
            self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet(esm_model)
        else:
            self.esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            self.esm_model = ESM2(alphabet=self.esm_alphabet,
                                  **self.models_cfg[esm_model]
                                 )        
        
        self.esm_embedding_size = self.esm_model.embed_tokens.weight.shape[1]
        self.esm_repr_layer = self.esm_model.num_layers
        # batch converter takes a list of tuples: [(name, seq), ...]
        # and converts it to labels:List[str], strs:List[str], tokens:torch.tensor
        self.esm_converter = self.esm_alphabet.get_batch_converter(None) # TODO: might want to add a max length here.
        self.esm_eos_token = self.esm_alphabet.all_toks.index('<eos>')
        self.esm_pad_token = self.esm_alphabet.all_toks.index('<pad>')
        self.freeze_esm()

        # 21 channels is number of 3Di symbols plus a pad token.
        self.target_alphabet = 'ACDEFGHIKLMNPQRSTVWY-' #TODO: is padding needed?

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.esm_embedding_size, out_channels=300, kernel_size=5, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=300, out_channels=len(self.target_alphabet), kernel_size=5, stride=1, padding=1)
        )

        if weights is not None:
            self.load_state_dict(weights, strict=False)
        

    def freeze_esm(self):
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for module in self.esm_model.modules():
            module.train(False)


    def unfreeze_esm(self, *esm_layers):
        esm_layers = list(esm_layers)
        for i, layer in enumerate(self.esm_model.layers):
            if i+1 in esm_layers:
                layer.train(True)
                for param in layer.parameters():
                    param.requires_grad = True



    # Defining the forward pass    
    def forward(self, x):
        # x should be a tensor of shape: num_seqs, length_of_longest_sequence
        encoded = self.esm_model(x, repr_layers=(self.esm_repr_layer,),
                                            return_contacts=False)["representations"][self.esm_repr_layer][:,1:-1] # 1 for <cls>, -1 for <eos> of longest sequence

        # mask eos and pad tokens
        x = x[:,1:-1]
        mask = torch.logical_and(torch.not_equal(x, self.esm_eos_token), torch.not_equal(x, self.esm_pad_token))

        encoded = torch.multiply(encoded, mask.unsqueeze(2))


        
        encoded = torch.transpose(encoded,1,2)
        out = self.cnn_layers(encoded)
        return out
    


    def encode_seqs(self, seqs):
        names_and_seqs = list()
        # seq_lengths = list()
        for i, seq in enumerate(seqs):
            names_and_seqs.append((str(i), seq))
            # seq_lengths.append(len(seq))

        _, _, tokenized = self.esm_converter(names_and_seqs)

        
        return tokenized # [num_seqs, length_of_longest_sequence]
        
    def encode_target(self, seq_records):
        alphabet = self.target_alphabet

        # define a mapping of chars to integers
        aa_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_aa = dict((i, c) for i, c in enumerate(alphabet))

        integer_encoded_representations = []

        for seq in seq_records:
            # integer encode seq data
            integer_encoded = torch.tensor([aa_to_int[aa] for aa in seq])
            integer_encoded_representations.append(integer_encoded)

        #pad result to get equal sized tensors
        output = pad_sequence(integer_encoded_representations,batch_first=True,padding_value=float(len(alphabet)-1))
        
        return output

    def decode_prediction(self, pred, lengths):
        """
            pred:   [num_seqs, alphabet_size, length_of_longest]
            lengths: lengths of starting sequences
        """
        alphabet = self.target_alphabet
        gap_i = len(alphabet) - 1
        
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
        
        out_seqs = list()
        for i in range(pred.shape[0]):
            out_seqs.append("".join([self.target_alphabet[pred[i, seq_pos]] for seq_pos in range(lengths[i])]))
        
        return out_seqs
        
    def accuracy(self, pred, actual):
        """
            pred:   [num_seqs, alphabet_size, length_of_longest]
            actual: [num_seqs, length_of_longest]
            
            returns: accuracy (0-1)
        """

        gap_token = len(self.target_alphabet) - 1
        
        actual = actual.cpu().detach().numpy()
        
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy() # [num_seqs, length_of_longest]

        actual_mask = np.not_equal(actual, gap_token) # masks non-pad positions to 1 and pad positions to 0.
        total = np.sum(actual_mask)
        same = np.equal(pred, actual)
        same = np.sum(np.multiply(same, actual_mask))
        
        return same/total
        
    def mismatch_matrix(self, pred, actual):
        """
            pred:   [num_seqs, alphabet_size, length_of_longest]
            actual: [num_seqs, length_of_longest]
            
            returns: matrix of predicted tokens vs actual tokens [pred, actual]
        """
        alphabet = self.target_alphabet
        gap_i = len(alphabet) - 1
        
        actual = actual.cpu().detach().numpy()
        
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy() # [num_seqs, length_of_longest]

        pred_vs_actual = np.zeros((len(alphabet), len(alphabet)), dtype=int)

        for seq_i in range(pred.shape[0]):
            for pos_i in range(pred.shape[1]):
                actual_token = actual[seq_i, pos_i]
                if actual != gap_i:
                    pred_token = pred[seq_i, pos_i]
                    pred_vs_actual[pred_token, actual_token] += 1

        return pred_vs_actual


class Incrementer:
    """An incrementer class that increments a value by one each time it's called."""

    # Initializing the class with a default start value of 0
    def __init__(self, start_val=-1):
        """Initialize the Incrementer with a starting value.

        Args:
            start_val (int, optional): The value to start from. Defaults to -1, so that it will return 0 on the first call.
        """
        # Here we set the initial value to start_val - 1 so that the first increment brings it up to start_val
        self.val = start_val

    # Defining the __call__ function to make the class instances callable
    # This method is called when the instance is "called" like a function
    def __call__(self, *args, **kwargs):
        """Increment the value by one and return it.

        Note:
            The *args and **kwargs are not used here. 
            They are included to allow the method to accept (and ignore) arbitrary arguments.

        Returns:
            int: The incremented value.
        """
        # Increment the value
        self.val += 1
        # Return the value
        return self.val
    
#class to create the pytorch dataset
class SeqsDataset(Dataset):

    def __init__(self, seqs_pep_path, seqs_3di_path):
        self.seqs_pep = SeqIO.index(seqs_pep_path, "fasta", key_function=Incrementer())
        self.seqs_3di = SeqIO.index(seqs_3di_path, "fasta", key_function=Incrementer())

    def __getitem__(self, i):
        seq_pep = str(self.seqs_pep._proxy.get(self.seqs_pep._offsets[i]).seq)
        seq_3di = str(self.seqs_3di._proxy.get(self.seqs_3di._offsets[i]).seq) # self.seqs_3di[i]
        return seq_pep, seq_3di

    def __len__(self):
        return len(self.seqs_pep)