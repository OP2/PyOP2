#include <petscsys.h>
#include <petscblaslapack.h>

#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
static void inverse(double4* __restrict__ Aout, const double4* __restrict__ A, PetscBLASInt N)
{
    PetscBLASInt nmat = 4;
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(PetscBLASInt));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(PetscScalar));
    PetscScalar *Abuf = malloc(N*N*sizeof(PetscScalar));
    PetscBLASInt k, n, m;
    for (k = 0; k < nmat; k++){
        for (n = 0; n < N; n++) {
            for (m = 0; m < N; m++) {
                Abuf[n*N+m] = A[n*N+m][k];
            }
        }
        LAPACKgetrf_(&N, &N, Abuf, &N, ipiv, &info);
        if(info == 0){
            LAPACKgetri_(&N, Abuf, &N, ipiv, Awork, &N, &info);
        }
        if(info != 0){
            fprintf(stderr, "Getri throws nonzero info.");
            abort();
        }
        for (n = 0; n < N; n++) {
            for (m = 0; m < N; m++) {
                Aout[n*N+m][k] = Abuf[n*N+m];
            }
        }
    }
    if ( N > BUF_SIZE ) {
        free(Awork);
        free(ipiv);
    }
}