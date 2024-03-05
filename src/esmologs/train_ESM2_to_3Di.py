#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#importing required libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from esmologs.ESM2_to_3Di import ESM2_to_3Di, SeqsDataset
import argparse
import sys
import re
from pathlib import Path
import logging

def setup_logger(log_file=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addHandler(handler)

# torch.set_printoptions(edgeitems=52)

def train_step(X, Y, model, loss_fn, optimizer, device="cpu"):
        """
            X: list of protein sequences
            Y: list of 3Di sequences
            model: 
            loss_fn: 
            optimizer:
            
            
            returns:
                loss, accuracy
        """
        # Compute prediction and loss
        x = model.encode_seqs(X) # returns tensor of size: [num_seqs, length_of_longest]
        x = x.to(device)

        y = model.encode_target(Y) # returns tensor of size: [num_seqs, length_of_longest]
        y = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, y)
        with torch.no_grad():
            accuracy = model.accuracy(pred,y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item(), accuracy


def validation_loop(dataloader, model, loss_fn, validation_batches, device="cpu"):
    batch_iter = iter(dataloader)
    with torch.no_grad():
        running_loss = 0
        running_accuracy = 0
        
        batch_num = 0
        for (X, Y) in batch_iter:
            # Compute prediction and loss
            x = model.encode_seqs(X) # returns tensor of size: [num_seqs, embedding_size(2560), length_of_longest]
            x = x.to(device)

            y = model.encode_target(Y) # returns tensor of size: [num_seqs, length_of_longest]
            y = y.to(device)

            pred = model(x)
            running_loss += loss_fn(pred, y).item()
            running_accuracy += model.accuracy(pred,y)
            
            batch_num += 1
            
            if batch_num == validation_batches:
                break

    return running_loss / batch_num, running_accuracy / batch_num

def train_model(model, train_dataloader, val_dataloader, checkpoint_dir, epochs, validation_interval, validation_batches, learning_rate=0.001, weight_decay=0.01, gamma=0.95, device="cpu"):
    checkpoint_dir = Path(checkpoint_dir)

    # defining the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # weights taken from 0.1 * the diagonal of the 3Di substitution matrix: https://github.com/steineggerlab/foldseek/blob/master/data/mat3di.out
    # 
    weights = torch.tensor([0.6,0.6,0.4,0.9,0.7,0.6,0.6,0.8,0.9,0.6,1.0,0.7,0.4,0.5,0.6,0.6,0.8,0.3,0.8,0.9,0.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    
    metrics_names = ["step", "epoch", "val_iteration", "train_loss","train_accuracy","val_loss","val_accuracy"]
    # metrics = {n:list() for n in metrics_names}

    logging.info("\t".join(metrics_names))
    step = 0
    for epoch in range(epochs):
        train_iter = iter(train_dataloader)
        
        loop_iteration = -1
        val_iteration = 1
        interval_train_accuracy = 0
        interval_train_batches = 0
        interval_train_loss = 0
        while True:
            loop_iteration += 1
            step += 1
            
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            
            train_loss, train_accuracy = train_step(batch[0], batch[1], model, loss_fn, optimizer, device=device)
            interval_train_batches += 1
            interval_train_accuracy += train_accuracy
            interval_train_loss += train_loss
            
            #TODO: it doesn't make a whole lot of sense to do validation batches within an epoch (or at least within the first epoch), because the training data has only been seen once anyways.
            if (loop_iteration % validation_interval) == 0: 
                val_loss, val_accuracy = validation_loop(val_dataloader, model, loss_fn, validation_batches, device=device)
                logging.info( "\t".join([
                                  str(step), 
                                  str(epoch), 
                                  str(val_iteration).zfill(6), 
                                  "%.2f" % (interval_train_loss / interval_train_batches),
                                  "%.2f" % (interval_train_accuracy / interval_train_batches), 
                                  "%.2f" % val_loss, 
                                  "%.2f" % val_accuracy
                                 ])
                     )
                scheduler.step()
                torch.save(model.state_dict(), checkpoint_dir /  f'{epoch}_{str(val_iteration).zfill(12)}.pt')
                val_iteration += 1
                
                interval_train_batches = 0
                interval_train_accuracy = 0
                interval_train_loss = 0
                
            
def main(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None,
                        help="prefix for training data. Looks for [train].pep.fasta and [train].3di.fasta")
    parser.add_argument("--val", type=str, default=None, required=True,
                        help="prefix for validation data. Looks for [val].pep.fasta and [train].3di.fasta")
    parser.add_argument("--checkpoint_dir", type=str, default=None, required=True,
                        help="Directory to save checkpoints into")
    parser.add_argument("--device", type=str, default="cpu", required=False,
                        help="What device to use.")
    
    parser.add_argument("--esm_model", type=str, default="esm2_t36_3B_UR50D", required=True,
                        choices={"esm2_t48_15B_UR50D","esm2_t36_3B_UR50D","esm2_t33_650M_UR50D",
                                 "esm2_t30_150M_UR50D","esm2_t12_35M_UR50D","esm2_t6_8M_UR50D"})
    
    parser.add_argument("--starting_weights", type=str, default=None, required=False,
                        help="If set, then initialize the model weights to these values, otherwise use ESM2 pretrained weights and the default pytorch initialization for the top layers.")
    parser.add_argument("--epochs", type=int, default=1, required=False,
                        help="Run training for this many passes over the training data.")
    parser.add_argument("--validation_interval", type=int, default=5000, required=False,
                        help="Run a validation batch every this many training batches. default: 5000")
    parser.add_argument("--validation_batches", type=int, default=50, required=False,
                        help="how many random minibatches from the validation set to run each validation interval. Total number of sequences will be"
                            " approximately batch_size*validation_batches. default: 50")
    parser.add_argument("--batch_size", type=int, default=200, required=False,
                        help="How many sequences to train or validate on in each minibatch.")
    parser.add_argument("--learning_rate", type=float, default=0.001, required=False,
                        help="Learning rate to use for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False,
                        help="Weight decay to use for the optimizer.")
    parser.add_argument("--gamma", type=float, default=0.95, required=False,
                        help="Learning rate decay to use for the optimizer.")
    parser.add_argument("--esm_layers_to_train", type=int, default=None, required=False, nargs="+",
                        help="If set, then only train these layers of the ESM model. Otherwise only train the CNN.")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file name. If not specified, logs are printed to the console.")
    
    
    params = parser.parse_args(argv)
    params.device = params.device.lower().strip()
    
    setup_logger(params.log)
    logging.info(params)
    
    device = torch.device("cpu")
    if re.match("^cuda(:[0-9]+)?$", params.device):
        device = torch.device(params.device)
    elif params.device != "cpu":
        raise ValueError("Device must be cpu, cuda, or cuda:[integer]")
    
    # defining the model

    if params.starting_weights is not None:
        model = ESM2_to_3Di(params.esm_model, weights=torch.load(params.starting_weights, map_location=device))
    else:
        model = ESM2_to_3Di(params.esm_model)

    model.to(device)
    
    
    checkpoint_dir = Path(params.checkpoint_dir)
    checkpoint_dir.mkdir(parents=False, exist_ok=True)
    #weights = torch.tensor([1.]*26+[0.])


    #load datasets
    train_dataset = SeqsDataset(params.train + ".pep.fasta", params.train + ".3di.fasta")
    val_dataset = SeqsDataset(params.val + ".pep.fasta", params.val + ".3di.fasta")

    #generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)

    if params.esm_layers_to_train is not None:
        model.unfreeze_esm(*params.esm_layers_to_train)
    

    train_model(model, train_dataloader, val_dataloader, checkpoint_dir, params.epochs, params.validation_interval, params.validation_batches, params.learning_rate, params.weight_decay, params.gamma, device)

if __name__ == '__main__':
    main(sys.argv[1:])
