# include <stdlib.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

void basic_dgemm_1x1x1 (int lda, int M, int N, int K, double *A, double *B, double *C);
void basic_dgemm_4x8x1 (int lda, int M, int N, int K, double *A, double *B, double *C);

extern double *AA;
static double A_temp[1000*1000];

void col_maj_2_row_maj(const int M,
		       const double *A, double *AA){

  double *col, *col_s;
  double* AA_s;
  int ii,jj;
  col_s = A;
  AA_s = AA;
  
  for(jj=0; jj<M; jj++){
    col = col_s;
    AA = AA_s;
    for(ii=0; ii<M; ii++){
      *AA = *col;
      col++;
      AA += M;
    }
    col_s += M;
    AA_s ++;
  }
  AA = AA_s - M;
}

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
                cij += A[i+k*lda] * B[k+j*lda];
            C[i+j*lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
    double *AA;
      if(lda<=1000)
    AA = A_temp;
  else {
    AA = (double*)malloc(sizeof(double)*lda*lda);
  }
  col_maj_2_row_maj(lda,A,AA);
    /* For each block-row of A */ 
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE, lda-i);
                int N = min (BLOCK_SIZE, lda-j);
                int K = min (BLOCK_SIZE, lda-k);

                /* Perform individual block dgemm */
                basic_dgemm_4x8x1(lda, M, N, K, AA + k + i*lda, B + k + j*lda, C + i + j*lda);
            }
        }
    }
}

void basic_dgemm_1x1x1 (int lda, int M, int N, int K, double *A, double *B, double *C) {
int i, j, k;
int bi, bj, bk;
int num_m, num_n, num_k;
int r = 1;
int c = 1;
num_m = M/r;
num_n = N/c;
num_k = K;
for (bi=0; bi<num_m; bi++) {
	 i = bi*r;
	 const register double *Ai_ = A + i*lda;
	for (bj=0; bj<num_n; bj++) {
		 j = bj*c;
		 const register double *B_j = B + j*lda;
		 register double cij_1 = C[((j+0)*lda)+i+0];
			for (k=0; k<K; ++k) {
			 cij_1 += Ai_[k+0*lda]*B_j[k+0*lda];
}
		 C[((j+0)*lda)+i+0] = cij_1 ;
}
}
  if(M%r !=0 || N%c !=0) {
  basic_dgemm_1x1x1(lda, M, N-num_n*c, K,A, B+num_n*c*lda, C+num_n*c*lda);
  basic_dgemm_1x1x1(lda, M-num_m*r, num_n*c, K, A+num_m*r*lda, B, C+num_m*r);
}
}

void basic_dgemm_4x8x1 (int lda, int M, int N, int K, double *A, double *B, double *C) {
int i, j, k;
int bi, bj, bk;
int num_m, num_n, num_k;
int r = 4;
int c = 8;
num_m = M/r;
num_n = N/c;
num_k = K;
for (bi=0; bi<num_m; bi++) {
	 i = bi*r;
	 const register double *Ai_ = A + i*lda;
	for (bj=0; bj<num_n; bj++) {
		 j = bj*c;
		 const register double *B_j = B + j*lda;
		 register double cij_1 = C[((j+0)*lda)+i+0];
		 register double cij_2 = C[((j+0)*lda)+i+1];
		 register double cij_3 = C[((j+0)*lda)+i+2];
		 register double cij_4 = C[((j+0)*lda)+i+3];
		 register double cij_5 = C[((j+1)*lda)+i+0];
		 register double cij_6 = C[((j+1)*lda)+i+1];
		 register double cij_7 = C[((j+1)*lda)+i+2];
		 register double cij_8 = C[((j+1)*lda)+i+3];
		 register double cij_9 = C[((j+2)*lda)+i+0];
		 register double cij_10 = C[((j+2)*lda)+i+1];
		 register double cij_11 = C[((j+2)*lda)+i+2];
		 register double cij_12 = C[((j+2)*lda)+i+3];
		 register double cij_13 = C[((j+3)*lda)+i+0];
		 register double cij_14 = C[((j+3)*lda)+i+1];
		 register double cij_15 = C[((j+3)*lda)+i+2];
		 register double cij_16 = C[((j+3)*lda)+i+3];
		 register double cij_17 = C[((j+4)*lda)+i+0];
		 register double cij_18 = C[((j+4)*lda)+i+1];
		 register double cij_19 = C[((j+4)*lda)+i+2];
		 register double cij_20 = C[((j+4)*lda)+i+3];
		 register double cij_21 = C[((j+5)*lda)+i+0];
		 register double cij_22 = C[((j+5)*lda)+i+1];
		 register double cij_23 = C[((j+5)*lda)+i+2];
		 register double cij_24 = C[((j+5)*lda)+i+3];
		 register double cij_25 = C[((j+6)*lda)+i+0];
		 register double cij_26 = C[((j+6)*lda)+i+1];
		 register double cij_27 = C[((j+6)*lda)+i+2];
		 register double cij_28 = C[((j+6)*lda)+i+3];
		 register double cij_29 = C[((j+7)*lda)+i+0];
		 register double cij_30 = C[((j+7)*lda)+i+1];
		 register double cij_31 = C[((j+7)*lda)+i+2];
		 register double cij_32 = C[((j+7)*lda)+i+3];
			for (k=0; k<K; ++k) {
			 cij_1 += Ai_[k+0*lda]*B_j[k+0*lda];
			 cij_2 += Ai_[k+1*lda]*B_j[k+0*lda];
			 cij_3 += Ai_[k+2*lda]*B_j[k+0*lda];
			 cij_4 += Ai_[k+3*lda]*B_j[k+0*lda];
			 cij_5 += Ai_[k+0*lda]*B_j[k+1*lda];
			 cij_6 += Ai_[k+1*lda]*B_j[k+1*lda];
			 cij_7 += Ai_[k+2*lda]*B_j[k+1*lda];
			 cij_8 += Ai_[k+3*lda]*B_j[k+1*lda];
			 cij_9 += Ai_[k+0*lda]*B_j[k+2*lda];
			 cij_10 += Ai_[k+1*lda]*B_j[k+2*lda];
			 cij_11 += Ai_[k+2*lda]*B_j[k+2*lda];
			 cij_12 += Ai_[k+3*lda]*B_j[k+2*lda];
			 cij_13 += Ai_[k+0*lda]*B_j[k+3*lda];
			 cij_14 += Ai_[k+1*lda]*B_j[k+3*lda];
			 cij_15 += Ai_[k+2*lda]*B_j[k+3*lda];
			 cij_16 += Ai_[k+3*lda]*B_j[k+3*lda];
			 cij_17 += Ai_[k+0*lda]*B_j[k+4*lda];
			 cij_18 += Ai_[k+1*lda]*B_j[k+4*lda];
			 cij_19 += Ai_[k+2*lda]*B_j[k+4*lda];
			 cij_20 += Ai_[k+3*lda]*B_j[k+4*lda];
			 cij_21 += Ai_[k+0*lda]*B_j[k+5*lda];
			 cij_22 += Ai_[k+1*lda]*B_j[k+5*lda];
			 cij_23 += Ai_[k+2*lda]*B_j[k+5*lda];
			 cij_24 += Ai_[k+3*lda]*B_j[k+5*lda];
			 cij_25 += Ai_[k+0*lda]*B_j[k+6*lda];
			 cij_26 += Ai_[k+1*lda]*B_j[k+6*lda];
			 cij_27 += Ai_[k+2*lda]*B_j[k+6*lda];
			 cij_28 += Ai_[k+3*lda]*B_j[k+6*lda];
			 cij_29 += Ai_[k+0*lda]*B_j[k+7*lda];
			 cij_30 += Ai_[k+1*lda]*B_j[k+7*lda];
			 cij_31 += Ai_[k+2*lda]*B_j[k+7*lda];
			 cij_32 += Ai_[k+3*lda]*B_j[k+7*lda];
}
		 C[((j+0)*lda)+i+0] = cij_1 ;
		 C[((j+0)*lda)+i+1] = cij_2 ;
		 C[((j+0)*lda)+i+2] = cij_3 ;
		 C[((j+0)*lda)+i+3] = cij_4 ;
		 C[((j+1)*lda)+i+0] = cij_5 ;
		 C[((j+1)*lda)+i+1] = cij_6 ;
		 C[((j+1)*lda)+i+2] = cij_7 ;
		 C[((j+1)*lda)+i+3] = cij_8 ;
		 C[((j+2)*lda)+i+0] = cij_9 ;
		 C[((j+2)*lda)+i+1] = cij_10 ;
		 C[((j+2)*lda)+i+2] = cij_11 ;
		 C[((j+2)*lda)+i+3] = cij_12 ;
		 C[((j+3)*lda)+i+0] = cij_13 ;
		 C[((j+3)*lda)+i+1] = cij_14 ;
		 C[((j+3)*lda)+i+2] = cij_15 ;
		 C[((j+3)*lda)+i+3] = cij_16 ;
		 C[((j+4)*lda)+i+0] = cij_17 ;
		 C[((j+4)*lda)+i+1] = cij_18 ;
		 C[((j+4)*lda)+i+2] = cij_19 ;
		 C[((j+4)*lda)+i+3] = cij_20 ;
		 C[((j+5)*lda)+i+0] = cij_21 ;
		 C[((j+5)*lda)+i+1] = cij_22 ;
		 C[((j+5)*lda)+i+2] = cij_23 ;
		 C[((j+5)*lda)+i+3] = cij_24 ;
		 C[((j+6)*lda)+i+0] = cij_25 ;
		 C[((j+6)*lda)+i+1] = cij_26 ;
		 C[((j+6)*lda)+i+2] = cij_27 ;
		 C[((j+6)*lda)+i+3] = cij_28 ;
		 C[((j+7)*lda)+i+0] = cij_29 ;
		 C[((j+7)*lda)+i+1] = cij_30 ;
		 C[((j+7)*lda)+i+2] = cij_31 ;
		 C[((j+7)*lda)+i+3] = cij_32 ;
}
}
  if(M%r !=0 || N%c !=0) {
  basic_dgemm_1x1x1(lda, M, N-num_n*c, K,A, B+num_n*c*lda, C+num_n*c*lda);
  basic_dgemm_1x1x1(lda, M-num_m*r, num_n*c, K, A+num_m*r*lda, B, C+num_m*r);
}
}