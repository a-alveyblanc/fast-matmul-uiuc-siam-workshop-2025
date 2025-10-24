template <class FP_T>
__global__ void naive_matmul(const FP_T *__restrict__ A,
                             const FP_T *__restrict__ B, FP_T *__restrict__ C,
                             const int n) {
  // FIXME: given what we know about linearized memory, does it make more sense
  // from a performance perspective to access a row-major array as 
  //   1. A[i, j]
  //   2. A[j, i]
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  FP_T acc = (FP_T) 0.0;
  for (int k = 0; k < n; ++k) {
    // FIXME: write the index expressions that correctly access A and B 
    // acc += A[] * B[];
  }

  // FIXME: write the index expression that correctly accesses C
  // C[] = acc;
}
