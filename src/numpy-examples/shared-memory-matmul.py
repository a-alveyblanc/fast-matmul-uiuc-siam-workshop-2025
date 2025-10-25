import numpy as np
import numpy.linalg as la

n = 64  # n small because python loops are slow and there are a lot of them 
bn = bm = bk = 32

a = np.random.randn(n, n)
b = np.random.randn(n, n)

c_true = (a @ b).flatten()
a = a.flatten()
b = b.flatten()
c = np.zeros_like(a)

a_s = np.zeros((bm, bk)).flatten()
b_s = np.zeros((bk, bn)).flatten()

for io in range(0, n, bm):                  # parallel over threadblocks 
    for jo in range(0, n, bn):              # parallel over threadblocks

        for ii in range(bm):                # parallel over threads
            for jj in range(bn):            # parallel over threads

                acc = 0.0
                for ko in range(0, n, bk):  # sequential within threads 

                    for ii_t in range(bm):     # parallel over threads
                        for jj_t in range(bk): # parallel over threads
                            a_s[ii_t*bk + jj_t] = a[(io + ii_t)*n + (ko + jj_t)]

                    for ii_t in range(bk):     # parallel over threads
                        for jj_t in range(bn): # parallel over threads
                            b_s[ii_t*bn + jj_t] = b[(ko + ii_t)*n + (jo + jj_t)]

                    for ki in range(bk):    # sequential within threads
                        acc += a_s[ii*bk + ki] * b_s[ki*bn + jj]

                c[(io + ii)*n + (jo + jj)] = acc

# sanity check
rel_err = la.norm(c - c_true) / la.norm(c_true)
print(f"Relative error = {rel_err:.4}")
assert rel_err < 1e-12, f"Relative error = {rel_err:.4}"
