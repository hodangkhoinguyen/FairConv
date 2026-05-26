import torch
import torch.nn as nn
import numpy as np


def load_nnet(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if not line.startswith("//")]

    # Remove empty lines
    lines = [l for l in lines if l]

    idx = 0

    # Header
    num_layers, input_size, output_size, _ = map(int, lines[idx].split(",")[:-1])
    idx += 1

    layer_sizes = list(map(int, lines[idx].split(",")[:-1]))
    idx += 1

    symmetric = int(lines[idx].split(",")[0])
    idx += 1

    input_mins = np.array(list(map(float, lines[idx].split(",")[:-1])))
    idx += 1

    input_maxs = np.array(list(map(float, lines[idx].split(",")[:-1])))
    idx += 1

    means = np.array(list(map(float, lines[idx].split(",")[:-1])))
    idx += 1

    ranges = np.array(list(map(float, lines[idx].split(",")[:-1])))
    idx += 1

    weights = []
    biases = []

    for layer in range(num_layers):
        out_size = layer_sizes[layer + 1]
        in_size = layer_sizes[layer]

        W = []
        for _ in range(out_size):
            row = list(map(float, lines[idx].split(",")[:-1]))
            W.append(row)
            idx += 1

        b = []
        for _ in range(out_size):
            val = float(lines[idx].split(",")[0])
            b.append(val)
            idx += 1

        weights.append(np.array(W))
        biases.append(np.array(b))

    return weights, biases, input_mins, input_maxs


def nnet_to_pytorch(filename):
    weights, biases, input_mins, input_maxs = load_nnet(filename)

    layers = []

    for i in range(len(weights)):
        in_features = weights[i].shape[1]
        out_features = weights[i].shape[0]

        linear = nn.Linear(in_features, out_features)

        linear.weight.data = torch.tensor(weights[i], dtype=torch.float32)
        linear.bias.data = torch.tensor(biases[i], dtype=torch.float32)

        layers.append(linear)

        # ReLU after every hidden layer
        if i < len(weights) - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    return model, input_mins, input_maxs

if __name__ == "__main__":
    path = "/storage/nguyenho/FairQuant-Artifact/models/adult/AC-1.nnet"
    model, input_mins, input_maxs = nnet_to_pytorch(path)
    print(model)
