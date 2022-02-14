#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

#ifndef PYOP2_SOLVE_LOG_EVENTS
#define PYOP2_SOLVE_LOG_EVENTS
static PetscLogEvent USER_EVENT_solve_memcpy;
static PetscLogEvent USER_EVENT_solve_getrf;
static PetscLogEvent USER_EVENT_solve_getrs;
#endif

static void solve(PetscScalar* __restrict__ out, const PetscScalar* __restrict__ A, const PetscScalar* __restrict__ B, PetscBLASInt N)
{
    PetscLogEventRegister("PyOP2SolveCallable_memcpy",PETSC_OBJECT_CLASSID,&USER_EVENT_solve_memcpy);
    PetscLogEventRegister("PyOP2SolveCallable_getrf",PETSC_OBJECT_CLASSID,&USER_EVENT_solve_getrf);
    PetscLogEventRegister("PyOP2SolveCallable_getrs",PETSC_OBJECT_CLASSID,&USER_EVENT_solve_getrs);

    PetscLogEventBegin(USER_EVENT_solve_memcpy,0,0,0,0);
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    memcpy(out,B,N*sizeof(PetscScalar));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Awork,A,N*N*sizeof(PetscScalar));
    PetscLogEventEnd(USER_EVENT_solve_memcpy,0,0,0,0);

    PetscBLASInt NRHS = 1;
    const char T = 'T';
    PetscLogEventBegin(USER_EVENT_solve_getrf,0,0,0,0);
    LAPACKgetrf_(&N, &N, Awork, &N, ipiv, &info);
    PetscLogEventEnd(USER_EVENT_solve_getrf,0,0,0,0);

    if(info == 0){
        PetscLogEventBegin(USER_EVENT_solve_getrs,0,0,0,0);
        LAPACKgetrs_(&T, &N, &NRHS, Awork, &N, ipiv, out, &N, &info);
        PetscLogEventEnd(USER_EVENT_solve_getrs,0,0,0,0);
    }

    if(info != 0){
        fprintf(stderr, "Gesv throws nonzero info.");
        abort();
    }

    if ( N > BUF_SIZE ) {
        free(ipiv);
        free(Awork);
    }
}
