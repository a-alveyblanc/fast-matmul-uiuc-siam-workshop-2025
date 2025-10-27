import cupy as cu
import numpy as np

import os

from dataclasses import dataclass

@dataclass(frozen=True)
class KernelRunner:
    kernel_name: str
    template: str

    def _get_kernel_src(self, read_full_src: bool = False) -> str:
        cur_dir = os.getcwd()

        kernel_dir = "full" if read_full_src else "blank"

        with open(
            f"{cur_dir}/../src/kernels/{kernel_dir}/{self.kernel_name}.cu",
            "r") as f:
            src = f.read()

        return src

    def __call__(self,
                 block_dim: tuple[int, ...],
                 grid_dim: tuple[int, ...],
                 args: tuple,
                 read_full_src: bool = False,
                 niterations: int = 1) -> np.ndarray:

        N, = args

        src = self._get_kernel_src(read_full_src)

        wrapper_name = f"{self.kernel_name}_float"
        code = src + f"""
extern "C" __global__
void {wrapper_name}(const float *__restrict__ A,
                    const float *__restrict__ B,
                    float *__restrict__ C,
                    const int N) {{
  {self.kernel_name}_matmul{self.template}(A, B, C, N);
}}
        """

        mod = cu.RawModule(code=code,
                           options=("-std=c++14",),
                           name_expressions=[wrapper_name])

        func = mod.get_function(wrapper_name)

        A = cu.random.rand(N, N, dtype=cu.float32)
        B = cu.random.rand(N, N, dtype=cu.float32)
        C = cu.zeros_like(A)

        C_true = A @ B

        if niterations == 1:
            func(grid_dim, block_dim, (A, B, C, N))
            cu.cuda.runtime.deviceSynchronize()
            print(f"Error = {cu.linalg.norm(C_true - C) / cu.linalg.norm(C):.4f}")
        else:
            # warm up calls
            for _ in range(5):
                func(grid_dim, block_dim, (A, B, C, N))
            cu.cuda.runtime.deviceSynchronize()

            start = cu.cuda.Event()
            end = cu.cuda.Event()

            start.record()
            for _ in range(niterations):
                func(grid_dim, block_dim, (A, B, C, N))
            end.record()
            end.synchronize()

            ms = cu.cuda.get_elapsed_time(start, end)

            print(10*"=", self.kernel_name, 10*"=")
            print(f"Total time  : {ms:.4f} ms")
            print(f"Average time: {ms / niterations:.4f} ms")

            s_per_iteration = (ms / niterations) * 1e-3
            nflops = float(2 * N**3)
            gflops = (nflops * 1e-9) / s_per_iteration
            print(f"GFLOP/s     : {gflops:.4f}")
            print(f"Error       : {cu.linalg.norm(C_true - C) / cu.linalg.norm(C):.4f}")
            print((22 + len(self.kernel_name))*"=")

        return C.get()