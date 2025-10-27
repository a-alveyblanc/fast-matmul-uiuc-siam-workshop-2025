# Optimization techniques for GPUs (WIP)
SIAM @ UIUC workshop on optimization techniques for GPUs. Learn about GPU
architecture by implementing matrix multiplication from naive to register tiled.
Lots of details to cover in each example, but goal is to assist in gaining
familiarity with how to think about optimizing algorithms on a GPU quickly. Goal
is to fill ~1-1.5 hours.

## Sketch of topics covered:
- Quick GPU architecture overview/review
    - SIMT, SMs, "threads", memory hierarchy
- Why matrix multiplication
- Math vs efficient computation
- Naive GPU implementation
- Shared memory implementation
    - Note on bank conflicts
- Register tiled implementation
    - Notes on register pressure, occupancy, etc.

Bounds checking in kernels is explicitly omitted to reduce visual noise. The
assumption is that sizes are carefully picked so that nothing goes out of
bounds.
