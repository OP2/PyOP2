#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

#ifndef PYOP2_INV_LOG_EVENTS
#define PYOP2_INV_LOG_EVENTS
static PetscLogEvent USER_EVENT_inv_memcpy;
static PetscLogEvent USER_EVENT_inv_getrf;
static PetscLogEvent USER_EVENT_inv_getri;
#endif

static void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N)
{
    PetscLogEventRegister("PyOP2InverseCallable_memcpy",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_memcpy);
    PetscLogEventRegister("PyOP2InverseCallable_getrf",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_getrf);
    PetscLogEventRegister("PyOP2InverseCallable_getri",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_getri);

    PetscLogEventBegin(USER_EVENT_inv_memcpy,0,0,0,0);
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Aout, A, N*N*sizeof(PetscScalar));
    PetscLogEventEnd(USER_EVENT_inv_memcpy,0,0,0,0);

    PetscLogEventBegin(USER_EVENT_inv_getrf,0,0,0,0);
    LAPACKgetrf_(&N, &N, Aout, &N, ipiv, &info);
    PetscLogEventEnd(USER_EVENT_inv_getrf,0,0,0,0);

    if(info == 0){
        PetscLogEventBegin(USER_EVENT_inv_getri,0,0,0,0);
        LAPACKgetri_(&N, Aout, &N, ipiv, Awork, &N, &info);
        PetscLogEventEnd(USER_EVENT_inv_getri,0,0,0,0);
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
