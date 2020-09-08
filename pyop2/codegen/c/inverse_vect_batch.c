#include <petscsys.h>
#include <mkl.h>

static void inverse(double4* Aout, const double4*  A, PetscBLASInt N)
{
    MKL_INT nmat = 4;
    MKL_COMPACT_PACK format = MKL_COMPACT_AVX;
    MKL_INT mkl_N = N;
    MKL_LAYOUT layout = MKL_ROW_MAJOR;
    MKL_INT info;

    double* A_compact = (double *)mkl_malloc(sizeof(double)*N*N*nmat, 64);
    double* Awork = (double *)mkl_malloc(sizeof(double)*N*N*nmat, 64);
    memcpy(A_compact, A, N*N*sizeof(double4));
    
    mkl_dgetrfnp_compact(layout, mkl_N, mkl_N, A_compact, mkl_N, &info, format, nmat);
    if(info == 0){
        mkl_dgetrinp_compact(layout, mkl_N, A_compact, mkl_N, Awork, N*N, &info, format, nmat);
    }else{
        fprintf(stderr, "mkl_dgetrinp_compact throws nonzero info.");
        abort();
    }
    memcpy(Aout, A_compact, N*N*sizeof(double4));
}
