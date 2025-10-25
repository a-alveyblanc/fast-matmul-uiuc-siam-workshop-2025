import numpy as np
import numpy.linalg as la

n = 128        # global problem size
bn = bm = 32   # threadblock size 

a = np.random.randn(n, n)
b = np.random.randn(n, n)

# flatten for linearized indexing in loops to match CUDA kernels
c_true = (a @ b).flatten()
a = a.flatten()
b = b.flatten()
c = np.zeros_like(a).flatten()

for io in range(n // bm):           # parallel over threadblocks
    for jo in range(n // bn):       # parallel over threadblocks

        for ii in range(bm):          # parallel over threads
            for jj in range(bn):      # parallel over threads
                i = io * bm + ii
                j = jo * bn + jj

                acc = 0.0
                for k in range(n):  # sequential within threads
                    acc += a[i*n + k] * b[k*n + j]

                c[i*n + j] = acc

# sanity check
rel_err = la.norm(c - c_true) / la.norm(c_true)
print(f"Relative error = {rel_err:.4}")
assert rel_err < 1e-12
