#include <petscsys.h>
#include <petscblaslapack.h>
#include <petsclog.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

static void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N)
{
    PetscLogEvent  USER_EVENT_memcpy;
    PetscLogEvent  USER_EVENT_getrf;
    PetscLogEvent  USER_EVENT_getri;
    PetscClassId   classid;

    PetscClassIdRegister("PyOP2InverseCallable",&classid);
    PetscLogEventRegister("PyOP2InverseCallable_memcpy",classid,&USER_EVENT_memcpy);
    PetscLogEventRegister("PyOP2InverseCallable_getrf",classid,&USER_EVENT_getrf);
    PetscLogEventRegister("PyOP2InverseCallable_getri",classid,&USER_EVENT_getri);

    PetscLogEventBegin(USER_EVENT_memcpy,0,0,0,0);
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Aout, A, N*N*sizeof(PetscScalar));
    PetscLogEventEnd(USER_EVENT_memcpy,0,0,0,0);

    PetscLogEventBegin(USER_EVENT_getrf,0,0,0,0);
    LAPACKgetrf_(&N, &N, Aout, &N, ipiv, &info);
    PetscLogEventEnd(USER_EVENT_getrf,0,0,0,0);

    if(info == 0){
        PetscLogEventBegin(USER_EVENT_getri,0,0,0,0);
        LAPACKgetri_(&N, Aout, &N, ipiv, Awork, &N, &info);
        PetscLogEventEnd(USER_EVENT_getri,0,0,0,0);
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
