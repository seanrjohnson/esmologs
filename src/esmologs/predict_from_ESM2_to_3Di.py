from esmologs.ESM2_to_3Di import ESM2_to_3Di
import argparse
import torch
from Bio.SeqIO import FastaIO
import sys
import re
from tqdm import tqdm
from pathlib import Path

torch.set_grad_enabled(False)

def convert_batch(model, seqs, device="cpu"):
    lengths = [len(seq) for seq in seqs]
    x = model.encode_seqs(seqs)
    x = x.to(device)
    pred = model(x)
    softmax = torch.nn.functional.softmax(pred, 1) #TODO: is this necessary?
    softmax[:, len(model.target_alphabet) - 1, :] = -1 # set gap to impossible probability
    predicted_seqs = model.decode_prediction(softmax, lengths)
    
    return predicted_seqs


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, nargs="+", default=None, required=True,
                    help="Protein fasta input files.")
    output_group = parser.add_mutually_exclusive_group(required=True)
    
    output_group.add_argument("-o", "--output", type=str, default=None, required=False,
                    help="3Di fasta input file.")
    output_group.add_argument("--output_dir", type=str, default=None, required=False,
                              help="Directory to write 3Di fasta input files to. Filenames will be the same as the input fasta files.")
    
    parser.add_argument("--esm_model", type=str, default="esm2_t36_3B_UR50D", required=False,
                        choices={"esm2_t48_15B_UR50D","esm2_t36_3B_UR50D","esm2_t33_650M_UR50D",
                                 "esm2_t30_150M_UR50D","esm2_t12_35M_UR50D","esm2_t6_8M_UR50D"})
    parser.add_argument("--batch_size", default=1, type=int, 
                    help="How many of the input sequences to predict at once.")
    parser.add_argument("--weights", type=str, default=None, required=True,
                    help="Training checkpoint to use for prediction.")
    parser.add_argument("--device", type=str, default="cpu", required=False,
                        help="What device to use.")

    parser.add_argument("--skip_existing", action='store_true', default=False, help="If set, then skip converting files for which the output file already exists.")

    
    params = parser.parse_args(argv)
    params.device = params.device.lower().strip()
    device = torch.device("cpu")
    if re.match("^cuda(:[0-9]+)?$", params.device):
        device = torch.device(params.device)
    elif params.device != "cpu":
        raise ValueError("Device must be cpu, cuda, or cuda:[integer]")
    
    # defining the model
    model = ESM2_to_3Di(params.esm_model, torch.load(params.weights, map_location=device))
    #model.load_state_dict(torch.load(params.weights, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    out_h = None
    if params.output_dir is not None:
        Path(params.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        out_h = open(params.output, "w")
        
    
    for input_file in params.input:
        if params.output_dir is not None:
            outfile_path = Path(params.output_dir) / Path(input_file).name
            if params.skip_existing and outfile_path.exists():
                continue
            out_h = open(outfile_path, "w")
        with open(input_file, "r") as in_h:
            names = list()
            seqs = list()
            for name, seq in tqdm(FastaIO.SimpleFastaParser(in_h)):
                names.append(name)
                seqs.append(seq)
                if len(names) == params.batch_size:
                    converted_seqs = convert_batch(model, seqs, device=device)
                    for converted_seq_i in range(len(converted_seqs)):
                        print(f">{names[converted_seq_i]}\n{converted_seqs[converted_seq_i]}", file=out_h)
                    names = list()
                    seqs = list()
    
            if len(names) > 0:
                converted_seqs = convert_batch(model, seqs)
                print(converted_seqs)
                for converted_seq_i in range(len(converted_seqs)):
                    print(f">{names[converted_seq_i]}\n{converted_seqs[converted_seq_i]}", file=out_h)
        if params.output_dir is not None:
            out_h.close()
    if params.output_dir is None:
        out_h.close()

if __name__ == '__main__':
    main(sys.argv[1:])
