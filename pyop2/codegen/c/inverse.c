#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
static PetscScalar Aout_proxy_buffer[BUF_SIZE*BUF_SIZE];
#endif

static void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N,
                    PetscBLASInt incA, PetscBLASInt incAout)
{
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));

    PetscInt N_sq = N * N;
    PetscInt one = 1;

    // Aout_proxy: 'Aout', but stored contiguously
    PetscScalar *Aout_proxy;
    if (incAout == 1)
      Aout_proxy = Aout;
    else
    {
      // TODO: Must see if allocating has a significant performance impact
      Aout_proxy = N_sq <= BUF_SIZE ? Aout_proxy_buffer : malloc(N*N*sizeof(*Aout));
    }

    BLAScopy_(&N_sq, A, &incA, Aout_proxy, &one);

    LAPACKgetrf_(&N, &N, Aout_proxy, &N, ipiv, &info);
    if(info == 0){
        LAPACKgetri_(&N, Aout_proxy, &N, ipiv, Awork, &N, &info);

        // Copy Aout_proxy back to Aout
        if (Aout != Aout_proxy)
          BLAScopy_(&N_sq, Aout_proxy, &one, Aout, &incAout);
    }
    if(info != 0){
        fprintf(stderr, "Getri throws nonzero info.");
        abort();
    }
    if ( N > BUF_SIZE ) {
        free(Awork);
        free(ipiv);
    }
}
