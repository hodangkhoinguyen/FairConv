import torch
import os

def create_vnnlib_str(data_lb: torch.Tensor, data_ub: torch.Tensor, prediction: torch.Tensor):
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    base_str = f"; Specification for class {int(y)}\n"
    base_str += f"\n; Definition of input variables\n"
    for i in range(len(x_ub)):
        base_str += f"(declare-const X_{i} Real)\n"

    base_str += f"\n; Definition of output variables\n"
    for i in range(n_class):
        base_str += f"(declare-const Y_{i} Real)\n"

    base_str += f"\n; Definition of input constraints\n"
    for i in range(len(x_ub)):
        base_str += f"(assert (<= X_{i} {x_ub[i]:.8f}))\n"
        base_str += f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n"

    base_str += f"\n; Definition of output constraints\n"
    spec_i = base_str
    spec_i += f"(assert (or\n"
    for i in range(n_class):
        if i == y:
            continue
        spec_i += f"\t(and (>= Y_{i} Y_{y}))\n"
    spec_i += f"))\n"
    return spec_i



def generate_property(vnnlib_path, data_lb, data_ub, n_class=2):
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    base_str = f"; Specification for fairness\n"
    base_str += f"\n; Definition of input variables\n"
    for i in range(len(x_ub)):
        base_str += f"(declare-const X_{i} Real)\n"

    base_str += f"\n; Definition of output variables\n"

    for i in range(n_class):
        base_str += f"(declare-const Y_{i} Real)\n"

    base_str += f"\n; Definition of input constraints\n"
    for i in range(len(x_ub)):
        base_str += f"(assert (<= X_{i} {x_ub[i]:.8f}))\n"
        base_str += f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n"

    base_str += f"\n; Definition of output constraints\n"
    base_str += f"(assert (or\n"

    # (Y_1 >= 0 and Y_0 <= 0)
    base_str += f"\t(and (>= Y_1 0) (<= Y_0 0))\n"
    # (Y_1 <= 0 and Y_0 >= 0)
    base_str += f"\t(and (<= Y_1 0) (>= Y_0 0))\n"

    base_str += f"))\n"

    with open(vnnlib_path, "w") as f:
        f.write(base_str)

WEIGHT_PATH_DICT = {
    "adult": "adult/AC-{}.nnet",
    "compas": "compas/compas-{}.nnet",
    "bank": "bank/BM-{}.nnet",
    "german": "german/GC-{}.nnet",
}

NUM_DICT = {
    "adult": 12,
    "compas": 7,
    "bank": 8,
    "german": 5,
}


def get_weight_paths(args) -> list:
    weight_path = os.path.join(args.model_dir, WEIGHT_PATH_DICT[args.dataset])
    nums = NUM_DICT[args.dataset]
    return [weight_path.format(idx) for idx in range(1, nums + 1)]
