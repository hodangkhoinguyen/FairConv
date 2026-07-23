import os
import time

def run_neuralsat(args, onnx_path, vnnlib_path, output_path, timeout):
    result_path = f'{output_path}.txt'
    log_path = f'{output_path}.log'
    if os.path.exists(result_path):
        status, runtime = open(result_path).read().strip().split(',')
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
            runtime = -1
        if not (status == 'error' or status == "timeout"):
            return status, runtime

    os.chdir(args.verifier_dir)

    cmd = f'timeout {timeout}s'
    cmd += f' python3 -W ignore main.py --verbosity=2'
    cmd += f' --net {onnx_path} --spec {vnnlib_path} --timeout {timeout}'
    cmd += f' --result_file {result_path}'

    # setting_path = os.path.join(args.home_dir, f'neuralsat_config.json')
    # assert os.path.exists(setting_path), f"Setting file does not exist: {setting_path=}"
    # print(setting_path)
    # cmd += f' --setting_file {setting_path}'

    cmd += f' --export_runtime'
    cmd += f' > {log_path} 2>&1'
    print(cmd)
    tic = time.time()
    os.system(cmd)
    toc = time.time()
    
    if os.path.exists(result_path):
        status, runtime = open(result_path).read().strip().split(',')
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
            runtime = -1
    else:
        status = 'error'
        with open(result_path, 'w') as f:
            print(f'{status},{toc - tic}', file=f)
    os.chdir(args.home_dir)
    return status, runtime

def run_neuralsat_from_src(args, new_model, num_inputs, vnnlib_path, input_split=True, device="cuda", timeout_each=100):
    import sys
    sys.path.insert(0, args.verifier_dir)
    from verifier.verifier import Verifier
    from setting import Settings
    from helper.spec.objective import parse_vnnlib

    input_shape = (1, num_inputs)
    objectives = parse_vnnlib(vnnlib_path, input_shape)

    Settings.setup(None)
    new_model.to(device)
    verifier = Verifier(
        net=new_model, 
        input_shape=input_shape, 
        batch=500,
        device=device,
    )

    if input_split:
        status = verifier.verify(objectives, timeout=timeout_each, force_split="input")
    else:
        status = verifier.verify(objectives, timeout=timeout_each)
    return status
