The three kernels we will cover are:
1. Naive matrix multiplication with a focus on index expressions and coalesced
   global accesses
2. Shared memory accesses with a focus on storing to/loading from shared memory
   and bank conflicts
3. Register tiled matrix multiplication

At each level, we'll look at loop structure first and then determine the proper
storage instructions to insert to exploit the memory hierarchy of the GPU.

Bounds checking is explicitly omitted to reduce visual noise. For these kernels
to be correct, sizes should be selected so they are compatible without bounds
checking.
