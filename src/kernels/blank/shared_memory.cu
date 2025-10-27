template <class FP_T, int BM, int BN, int BK>
__global__ void shared_memory_matmul(const FP_T *__restrict__ A,
                                     const FP_T *__restrict__ B,
                                     FP_T *__restrict__ C,
                                     const int n) {
  // selects the (i, j) tile of our output
  int i = blockIdx.y * blockDim.y;
  int j = blockIdx.x * blockDim.x;

  // FIXME: what should this be? does it matter here?
  // int ii = threadIdx.???;
  // int jj = threadIdx.???;

  // FIXME: can you fix the bank conflicts?
  __shared__ FP_T smem[BM*BK + BK*BN];

  // FIXME: set up a variable for the A shared memory buffer
  // FP_T *As = ...;
  // int lda = ...;

  // FIXME: set up a variable for the B shared memory buffer
  // HINT: requires pointer arithmetic
  // FP_T *Bs = ...;
  // int ldb = ...;

  // NOTE: ldX stands for "leading dimension X", if you're familiar with
  // numpy-ish terminology this is the "stride" of an axis of an array. put
  // loosely, it's the number of elements we need to jump over to move to the
  // next entry in that axis.
  // as a concrete example, in our row-major matrix A, we have to jump over lda
  // number of elements to get from one row to the next

  FP_T acc = (FP_T) 0.0;
  for (int ko = 0; ko < n; ko += BK) { // outer loop over k-blocks
    // FIXME: insert the correct index expressions for shared and global memory
    // As[] = A[];
    // Bs[] = B[];
    __syncthreads(); // ensure all necessary data is loaded 

    for (int ki = 0; ki < BK; ++ki) { // reduction loop within k-blocks 
      // FIXME: insert the instruction that accumulates the result
      // acc += ...;
    }
    __syncthreads(); // ensure all data has been used
  }

  // FIXME: insert the correct index expression for the store into C
  // C[] = acc;
}