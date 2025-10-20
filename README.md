# Optimization techniques for GPUs (WIP)
SIAM @ UIUC workshop on optimization techniques for GPUs. Learn about GPU
architecture by implementing matrix multiplication from naive to register tiled.
Lots of material, but goal is to assist in gaining familiarity with how to think
about optimizing algorithms on a GPU quickly. Goal is to fill ~1-1.5 hours.

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
