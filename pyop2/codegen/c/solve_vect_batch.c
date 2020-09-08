#include <petscsys.h>
#include <mkl.h>

static void solve(double4* __restrict__ Aout, const double4* __restrict__ A, const double4* __restrict__ B, PetscBLASInt N)
{
    MKL_INT nmat = 4;
    MKL_COMPACT_PACK format = MKL_COMPACT_AVX;
    MKL_INT mkl_N = N;
    MKL_LAYOUT layout = MKL_COL_MAJOR;
    MKL_INT info;
    MKL_TRANSPOSE T = MKL_TRANS;
    MKL_TRANSPOSE TN = MKL_NOTRANS;
    double one = 1.0;

    double* A_compact = (double *)mkl_malloc(sizeof(double)*N*N*nmat, 64);
    double* Awork = (double *)mkl_malloc(sizeof(double)*N*N*nmat, 64);
    memcpy(A_compact, A, N*N*sizeof(double)*nmat);
    double* B_compact = (double *)mkl_malloc(sizeof(double)*N*nmat, 64);
    memcpy(B_compact, B, N*sizeof(double)*nmat);
    double* C_compact = (double *)mkl_malloc(sizeof(double)*N*nmat, 64);

    int n, k;
    for (n = 0; n < N; n++) {
        for (k = 0; k < nmat; k++){
            C_compact[n*nmat+k] = 0.0;
        }
    }
    
    mkl_dgetrfnp_compact(layout, mkl_N, mkl_N, A_compact, mkl_N, &info, format, nmat);
    if(info == 0){
        mkl_dgetrinp_compact(layout, mkl_N, A_compact, mkl_N, Awork, N*N, &info, format, nmat);
    }else{
        fprintf(stderr, "Getrf throws nonzero info.");
        abort();
    }
    mkl_dgemm_compact(layout, TN, TN, mkl_N, one, mkl_N, one, A_compact, mkl_N, B_compact , mkl_N, one, C_compact, mkl_N, format, nmat);
    
    for (n = 0; n < N; n++) {
        for (k = 0; k < nmat; k++){
            Aout[n][k] = C_compact[n*nmat+k];
        }
    }
    mkl_free(A_compact);
    mkl_free(B_compact);
    mkl_free(C_compact);
    mkl_free(Awork);
}