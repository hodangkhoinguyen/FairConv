import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import copy

from nnet2torch import nnet_to_pytorch
from util import generate_property
from run_verify import run_neuralsat_from_src

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight-path", type=str, default="/storage/nguyenho/FairQuant-Artifact/models/adult/AC-1.nnet")
    # p.add_argument("--pa", type=str, help="Protected attribute", choices=["sex"])
    # p.add_argument("--benchmark_dir", type=str, required=True, help="Benchmark directory containing h5 model")
    # p.add_argument("--bit_all", type=int, default=4, choices=[4, 8]) 
    # p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "acasxu"])
    # p.add_argument("--export_dir", type=str, default="./export_dir", help="Root directory for exporting converted model")
    # p.add_argument("--rad", type=int, default=2)
    p.add_argument("--verifier_dir", type=str, default="./neuralsat_src", help="Verifier directory")

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

    pa_idx = 8 # proctected attribute index 

    model, input_mins, input_maxs = nnet_to_pytorch(args.weight_path)
    print(model)
    print(input_mins)
    print(input_maxs)
    sens_A = input_mins[pa_idx]
    sens_B = input_maxs[pa_idx]

    model_A: nn.Module = model
    
    old_layer = model[0]
    weight = old_layer.weight.data
    bias = old_layer.bias.data

    print(f"{weight[:, pa_idx]=}")
    print(f"{sens_A=} {weight[:, pa_idx] * sens_A=}")
    print(f"{sens_B=} {weight[:, pa_idx] * sens_B=}")

    new_bias_A = bias + weight[:, pa_idx] * sens_A
    new_bias_B = bias + weight[:, pa_idx] * sens_B
    new_weight = torch.cat([weight[:, :pa_idx], weight[:, pa_idx + 1:]], dim=1)

    # Fix the first layer for model_A (PA = sens_A)
    new_layer_A = nn.Linear(12, 16)
    new_layer_A.weight.data = new_weight  # pa_idx column removed
    new_layer_A.bias.data = new_bias_A
    model_A[0] = new_layer_A

    # Fix the first layer for model_B (PA = sens_B)
    model_B = copy.deepcopy(model)  # deepcopy AFTER you have the shared new_weight ready
    new_layer_B = nn.Linear(12, 16)
    new_layer_B.weight.data = new_weight
    new_layer_B.bias.data = new_bias_B
    model_B[0] = new_layer_B

    merged_model = MergedFairnessDNN(model_A, model_B)
    print(merged_model)
    batch = 5
    dummy_input = torch.rand(batch, 12)
    print(merged_model(dummy_input))
    input_mins = np.delete(input_mins, pa_idx)
    input_maxs = np.delete(input_maxs, pa_idx)

    vnnlib_path = "hehe.vnnlib"
    generate_property(
        vnnlib_path,
        torch.tensor(input_mins),
        torch.tensor(input_maxs),
    )

    status = run_neuralsat_from_src(args, merged_model, 12, vnnlib_path, device="cuda")
    print(f"{status=}")

if __name__ == "__main__":
    main()
