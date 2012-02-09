const char* dgemm_desc = "Tiled dgemm.";

#if !defined(BLOCK_SIZE_1)
#define BLOCK_SIZE_1 41
#endif
#if !defined(BLOCK_SIZE_2)
#define BLOCK_SIZE_2 128
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
    {
        /* For each column j of B */ 
        for (int j = 0; j < N; ++j) 
        {
            /* Compute C(i,j) */
            double cij = C[i+j*lda];
            for (int k = 0; k < K; ++k)
            {
                cij += A[i+k*lda] * B[k+j*lda];
            }
            C[i+j*lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    /* For each block-row of A */
    for (int i1 = 0; i1 < lda; i1 += BLOCK_SIZE_2)
    {
        /* For each block-column of B */
        for (int j1 = 0; j1 < lda; j1 += BLOCK_SIZE_2)
        {
            /* Accumulate block dgemms into block of C */
            for (int k1 = 0; k1 < lda; k1 += BLOCK_SIZE_2)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M1 = min (BLOCK_SIZE_2, lda-i1);
                int N1 = min (BLOCK_SIZE_2, lda-j1);
                int K1 = min (BLOCK_SIZE_2, lda-k1);
                
                /* For each block-row of A */
                for (int i2 = 0; i2 < M1; i2 += BLOCK_SIZE_1)
                {
                    /* For each block-column of B */
                    for (int j2 = 0; j2 < N1; j2 += BLOCK_SIZE_1)
                    {
                        /* Accumulate block dgemms into block of C */
                        for (int k2 = 0; k2 < K1; k2 += BLOCK_SIZE_1)
                        {
                            /* Correct block dimensions if block "goes off edge of" the matrix */
                            int M2 = min (BLOCK_SIZE_1, M1-i2);
                            int N2 = min (BLOCK_SIZE_1, N1-j2);
                            int K2 = min (BLOCK_SIZE_1, K1-k2);
                            /* Perform individual block dgemm */
                            do_block(lda, M2, N2, K2, A + i1 + k1*lda + i2 + k2*lda, B + k1 + j1*lda + k2 + j2*lda, C + i1 + j1*lda + i2 + j2*lda);
                        }
                    }
                }
            }
        }
    }
}
