template <class FP_T, int BM, int BN, int BK, int TM, int TN>
__global__ void register_tiled_matmul(const FP_T *__restrict__ A,
                                      const FP_T *__restrict__ B,
                                      FP_T *__restrict__ C,
                                      const int N) {
  // choose upper left corner of (i, j) output tile
  int i = blockIdx.y * BM;  // Q: why BM instead of blockDim.y?
  int j = blockIdx.x * BN;  // Q: why BN instead of blockDim.x?

  int By = BM / TM;
  int Bx = BN / TN;

  int ii = threadIdx.y;
  int jj = threadIdx.x;

  int iiC = i + ii*TM;
  int jjC = j + jj*TN;

  // NOTE: given row-major and non-transposed in memory, which array should we
  // pad to reduce bank conflicts?
  __shared__ FP_T As[BM][BK];
  __shared__ FP_T Bs[BK][BN];

  FP_T acc[TM][TN] = {0.0};

  for (int ko = 0; ko < N; ko += BK) {

    // we now have less threads per block than output entries, therefore each
    // thread needs to load a tile from global memory to shared memory
    // Q: is this method of loading from global to shared efficient? why or why
    // not?
    for (int row_load = ii; row_load < BM; row_load += By)
      for (int col_load = jj; col_load < BK; col_load += Bx)
        As[row_load][col_load] = A[(i + row_load)*N + (ko + col_load)];

    // collaboratively load tiles into shared memory for B
    for (int row_load = ii; row_load < BK; row_load += By)
      for (int col_load = jj; col_load < BN; col_load += Bx)
        Bs[row_load][col_load] = B[(ko + row_load)*N + (j + col_load)];

    __syncthreads();

    // microkernel performing the matmul
    for (int ki = 0; ki < BK; ++ki) {
      FP_T a_row[TM];
      FP_T b_col[TN];

      // shared -> register loading
      for (int a = 0; a < TM; ++a)
        a_row[a] = As[ii*TM + a][ki];

      // shared -> register loading
      for (int b = 0; b < TN; ++b)
        b_col[b] = Bs[ki][jj*TN + b];

      // perform the reduction
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          acc[m][n] += a_row[m] * b_col[n];
    }
    __syncthreads();
  }

  // store into C from register accumulator
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n)
      C[(iiC + m)*N + jjC + n] = acc[m][n];
}

