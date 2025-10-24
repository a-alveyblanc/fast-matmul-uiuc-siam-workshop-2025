template <class FP_T>
__global__ void naive_matmul(const FP_T *__restrict__ A,
                             const FP_T *__restrict__ B, FP_T *__restrict__ C,
                             const int n) {
  // NOTE: one possible solution (if we want to access as A[i,j]) is to swap the
  // .x and .y. This is because from the hardware's perspective, x is the
  // "fastest moving" axis and j is the "fastest moving" axis of the array in
  // memory, leading to "coalesced accesses". Otherwise, we can swap the roles
  // of i and j in the index expressions below
  int i = blockIdx.y * blockDim.y + threadIdx.y; 
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  FP_T acc = (FP_T) 0.0;
  for (int k = 0; k < n; ++k) {
    // NOTE: solution given the index switch above
    acc += A[i*n + k] * B[k*n + j];
  }

  // NOTE: solution given the index switch above
  C[i*n + j] = acc;
}
