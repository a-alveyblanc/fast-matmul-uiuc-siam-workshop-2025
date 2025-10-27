import numpy as np
import cupy as cu

import sys
import os

def main():
    kernel_name, N = sys.argv[1:]
    N = int(N)

    if kernel_name not in ["naive", "shared_memory", "register_tiled"]:
        raise ValueError(
            "Pre-defined kernels are `naive`, `shared-memory`, and "
            f"`register-tiled`. Got {kernel_name}")

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    with open(f"{cur_dir}/../kernels/full/{kernel_name}.cu", "r") as f:
        src = f.read()

    if kernel_name == "naive":
        template = "<float>"
    elif kernel_name == "shared_memory":
        template = "<float, 32, 32, 32>"
    else:
        template = "<float, 128, 128, 8, 8, 8>"

    wrapper_name = f"{kernel_name}_float"

    code = src + f"""
extern "C" __global__
void {wrapper_name}(const float *__restrict__ A,
                    const float *__restrict__ B,
                    float *__restrict__ C,
                    const int N) {{
  {kernel_name}_matmul{template}(A, B, C, N);
}}
    """

    mod = cu.RawModule(code=code,
                       options=("-std=c++14",),
                       name_expressions=[wrapper_name])

    func = mod.get_function(wrapper_name)

    A = cu.random.rand(N, N, dtype=cu.float32)
    B = cu.random.rand(N, N, dtype=cu.float32)
    C = cu.zeros_like(A)

    if kernel_name == "register_tiled":
        BM, BN, TM, TN = 128, 128, 8, 8
        block_dim = (BN // TN, BM // TM)
        grid_dim = ((N + BM - 1) // BM,
                    (N + BN - 1) // BN)
    else:
        BM = BN = 32
        block_dim = (BN, BM)
        grid_dim = (N // BN, N // BM)

    func(grid_dim, block_dim, (A, B, C, N))
    cu.cuda.runtime.deviceSynchronize()

    C_true = A @ B
    print(f"Error = {cu.linalg.norm(C_true - C) / cu.linalg.norm(C):.4f}")

if __name__ == "__main__":
    main()
