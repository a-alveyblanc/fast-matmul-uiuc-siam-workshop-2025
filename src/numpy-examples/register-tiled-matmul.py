import numpy as np
import numpy.linalg as la

n = 64             # global problem size
bn = bm = bk = 32  # output tile size
tm = tn = 4        # register tile size

tbp_y = bm // tm
tbp_x = bn // tn

a = np.random.randn(n, n)
b = np.random.randn(n, n)

c_true = (a @ b).flatten()

# global arrays
a = a.flatten()
b = b.flatten()
c = np.zeros_like(a)

# shared memory
a_s = np.zeros((bm, bk)).flatten()
b_s = np.zeros((bk, bn)).flatten()

# register tiles
a_row = np.zeros(tm)
b_col = np.zeros(tn)
c_reg = np.zeros((tm, tn)).flatten()
