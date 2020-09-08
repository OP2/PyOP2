#include <petscsys.h>
#include <petscblaslapack.h>

#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
static void solve(double4* __restrict__ out, const double4* __restrict__ A, const double4* __restrict__ B, PetscBLASInt N)
{
    PetscBLASInt nmat = 4;
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(PetscBLASInt));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(PetscScalar));
    PetscScalar *Abuf = malloc(N*N*sizeof(PetscScalar));
    PetscScalar *Bbuf = malloc(N*sizeof(PetscScalar));
    memcpy(out,B,N*sizeof(PetscScalar));
    PetscBLASInt NRHS = 1;
    const char T = 'T';
    PetscBLASInt k, n, m;
    for (k = 0; k < nmat; k++){
        for (n = 0; n < N; n++) {
            for (m = 0; m < N; m++) {
                Abuf[n*N+m] = A[n*N+m][k];
            }
            Bbuf[n] = B[n][k];
        }
        LAPACKgetrf_(&N, &N, Abuf, &N, ipiv, &info);
        if(info == 0){
            LAPACKgetrs_(&T, &N, &NRHS, Abuf, &N, ipiv, Bbuf, &N, &info);
        }
        if(info != 0){
            fprintf(stderr, "Getrs throws nonzero info.");
            abort();
        }
        for (n = 0; n < N; n++) {
            out[n][k] = Bbuf[n];
        }
    }
    if ( N > BUF_SIZE ) {
        free(Awork);
        free(ipiv);
    }
}