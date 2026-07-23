import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import copy

from nnet2torch import nnet_to_pytorch
from util import generate_property, get_weight_paths
from run_verify import run_neuralsat_from_src

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str, choices=["compas", "adult", "bank", "german"])
    p.add_argument("--model-dir", type=str, default="/storage/nguyenho/FairQuant-Artifact/models")
    # p.add_argument("--benchmark_dir", type=str, required=True, help="Benchmark directory containing h5 model")
    # p.add_argument("--bit_all", type=int, default=4, choices=[4, 8]) 
    # p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "acasxu"])
    # p.add_argument("--export_dir", type=str, default="./export_dir", help="Root directory for exporting converted model")
    # p.add_argument("--rad", type=int, default=2)
    p.add_argument("--verifier_dir", type=str, default="./neuralsat_src", help="Verifier directory")
    p.add_argument("--timeout", type=int, default=100, help="Timeout (seconds) for verification")

    args = p.parse_args()

    return args

class MergedFairnessDNN(nn.Module):
    def __init__(self, model_A, model_B):
        super().__init__()
        self.model_A = model_A  # PA = min
        self.model_B = model_B  # PA = max

    def forward(self, x):
        out_A = self.model_A(x)  # shape: (batch, 1)
        out_B = self.model_B(x)  # shape: (batch, 1)

        return torch.cat([out_A, out_B], dim=1)

def main():
    args = parse_args()
    tic = time.time()

    if args.dataset == "adult":
        pa_idx = 8 # proctected attribute index 
    elif args.dataset == "compas":
        pa_idx = 3 # race
    elif args.dataset == "bank":
        pa_idx = 0 # age
    elif args.dataset == "german":
        pa_idx = 11 # age
    else:
        raise ValueError("f{args.dataset=} is not valid dataset")
    weight_paths = get_weight_paths(args)

    # for idx, weight_path in enumerate(weight_paths):
    weight_path, idx = "/storage/nguyenho/FairQuant-Artifact/models/compas/compas-7.nnet", 6
    # weight_path, idx = "/storage/nguyenho/FairQuant-Artifact/models/adult/AC-1.nnet", 0
    print(f"-------- {args.dataset} -- {idx+1} -------------")
    model, input_mins, input_maxs = nnet_to_pytorch(weight_path)
    num_inputs = len(input_mins)
    merged_num_inputs = len(input_mins) - 1

    # print(f"Original: {num_inputs=}")
    # print(f"Merged: {merged_num_inputs=}")

    sens_A = input_mins[pa_idx]
    sens_B = input_maxs[pa_idx]
    model_A: nn.Module = model
    
    old_layer = model[0]
    weight = old_layer.weight.data
    bias = old_layer.bias.data

    new_bias_A = bias + weight[:, pa_idx] * sens_A
    new_bias_B = bias + weight[:, pa_idx] * sens_B
    new_weight = torch.cat([weight[:, :pa_idx], weight[:, pa_idx + 1:]], dim=1)

    # Fix the first layer for model_A (PA = sens_A)
    new_layer_A = nn.Linear(merged_num_inputs, model[0].out_features)
    new_layer_A.weight.data = new_weight  # pa_idx column removed
    new_layer_A.bias.data = new_bias_A
    model_A[0] = new_layer_A

    # Fix the first layer for model_B (PA = sens_B)
    model_B = copy.deepcopy(model)  # deepcopy AFTER you have the shared new_weight ready
    new_layer_B = nn.Linear(merged_num_inputs, model[0].out_features)
    new_layer_B.weight.data = new_weight
    new_layer_B.bias.data = new_bias_B
    model_B[0] = new_layer_B

    merged_model = MergedFairnessDNN(model_A, model_B)
    input_mins = np.delete(input_mins, pa_idx)
    input_maxs = np.delete(input_maxs, pa_idx)

    vnnlib_path = "hehe.vnnlib"
    generate_property(
        vnnlib_path,
        torch.tensor(input_mins),
        torch.tensor(input_maxs),
    )

    status = run_neuralsat_from_src(args, merged_model, merged_num_inputs, vnnlib_path, device="cuda", timeout_each=args.timeout)
    runtime = time.time() - tic

    print(f"{status=}")
    print(f"{runtime=}")

if __name__ == "__main__":
    main()
