template <class FP_T, int BM, int BN, int BK>
__global__ void shared_memory_matmul(const FP_T *__restrict__ A,
                                     const FP_T *__restrict__ B,
                                     FP_T *__restrict__ C,
                                     const int n) {
  int i = blockIdx.y * blockDim.y;
  int j = blockIdx.x * blockDim.x;

  // NOTE: it does matter given the equations above, namely the added threadIdx
  // at the end
  int ii = threadIdx.y;
  int jj = threadIdx.x;

  // NOTE: prone to bank conflicts, pad one of the arrays to eliminate them 
  __shared__ FP_T smem[BM*BK + BK*BN];
  // __shared__ FP_T smem[BM*(BK + 1) + BK*BN];

  // NOTE: for a row-major array, bank conflicts can be expected to arise when
  // we are iterating down a particular column with threads. this is because
  // threads in a warp have a high likelyhood of accessing different addresses
  // in the same bank (given the correct sizes of BM, BK with BK mattering more
  // here and BM "accidentally" helping us). why is this the case?

  // NOTE: for those curious, the bank id of some array entry can be computed 
  // by (addr / 4) % 32. for those who understand bit manipulation this
  // is the same as taking the lower 5 bits of the address: (addr >> 2) & 0x1F

  FP_T *As = smem;
  int lda = BK;
  // int lda = BK + 1;  // uncomment if using padded array

  FP_T *Bs = As + BM*BK;
  // FP_T *Bs = As + BM*(BK + 1);  // uncomment if using padded array
  int ldb = BN;

  // NOTE: ldX stands for "leading dimension X", if you're familiar with
  // numpy-ish terminology this is the "stride" of an axis of an array. put
  // loosely, it's the number of elements we need to jump over to move to the
  // next entry in that axis.
  // as a concrete example, in our row-major matrix A, we have to jump over lda
  // number of elements to get from one row to the next

  FP_T acc = (FP_T) 0.0;
  for (int ko = 0; ko < n; ko += BK) { // outer loop over k-blocks
    As[ii*lda + jj] = A[(i + ii)*n + (ko + jj)]; // i + ii selects the row
                                                 // ko + jj selects the column
    Bs[ii*ldb + jj] = B[(i + ko)*n + (j + jj)];  // i + ko selects the row
                                                 // j + jj selects the column
    __syncthreads(); // ensure all necessary data is loaded 

    for (int ki = 0; ki < BK; ++ki) { // reduction loop within k-blocks 
      acc += As[ii*lda + ki] * Bs[ki*ldb + jj];
    }
    __syncthreads(); // ensure all data has been used
  }

  C[(i + ii)*n + (j + jj)] = acc;
}
