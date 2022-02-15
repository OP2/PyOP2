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
PetscLogEvent USER_EVENT_solve_memcpy = -1;
PetscLogEvent USER_EVENT_solve_getrf = -1;
PetscLogEvent USER_EVENT_solve_getrs = -1;
#endif

#ifndef BEGIN_LOG
#define BEGIN_LOG
static void beginLog(PetscLogEvent eventId){
    #ifdef PYOP2_PROFILING_ENABLED
    PetscLogEventBegin(eventId,0,0,0,0);
    #endif
}
#endif

#ifndef END_LOG
#define END_LOG
static void endLog(PetscLogEvent eventId){
    #ifdef PYOP2_PROFILING_ENABLED
    PetscLogEventEnd(eventId,0,0,0,0);
    #endif
}
#endif

void solve(PetscScalar* __restrict__ out, const PetscScalar* __restrict__ A, const PetscScalar* __restrict__ B, PetscBLASInt N)
{
    beginLog(USER_EVENT_solve_memcpy);
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    memcpy(out,B,N*sizeof(PetscScalar));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Awork,A,N*N*sizeof(PetscScalar));
    endLog(USER_EVENT_solve_memcpy);

    PetscBLASInt NRHS = 1;
    const char T = 'T';
    beginLog(USER_EVENT_solve_getrf);
    LAPACKgetrf_(&N, &N, Awork, &N, ipiv, &info);
    endLog(USER_EVENT_solve_getrf);

    if(info == 0){
        beginLog(USER_EVENT_solve_getrs);
        LAPACKgetrs_(&T, &N, &NRHS, Awork, &N, ipiv, out, &N, &info);
        endLog(USER_EVENT_solve_getrs);
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
