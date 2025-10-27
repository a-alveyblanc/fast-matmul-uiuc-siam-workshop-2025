from runners import (
    NaiveKernelRunner,
    SharedMemoryKernelRunner,
    RegisterTiledKernelRunner
)

import sys
import os

def main():
    kernel_name, N = sys.argv[1:]
    N = int(N)

    if kernel_name not in ["naive", "shared_memory", "register_tiled"]:
        raise ValueError(
            "Pre-defined kernels are `naive`, `shared-memory`, and "
            f"`register-tiled`. Got {kernel_name}")

    if kernel_name == "register_tiled":
        BM, BN, TM, TN = 128, 128, 8, 8
        block_dim = (BN // TN, BM // TM)
        grid_dim = ((N + BM - 1) // BM,
                    (N + BN - 1) // BN)
    else:
        BM = BN = 32
        block_dim = (BN, BM)
        grid_dim = (N // BN, N // BM)

    args = (N,)
    if kernel_name == "naive":
        runner = NaiveKernelRunner()
    elif kernel_name == "shared_memory":
        runner = SharedMemoryKernelRunner()
    else:
        runner = RegisterTiledKernelRunner()

    runner(block_dim, grid_dim, args, read_full_src=True, niterations=20)

if __name__ == "__main__":
    main()
