#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

static PetscScalar out_proxy_buffer[BUF_SIZE];

#ifndef PYOP2_SOLVE_LOG_EVENTS
#define PYOP2_SOLVE_LOG_EVENTS
PetscLogEvent ID_solve_memcpy = -1;
PetscLogEvent ID_solve_getrf = -1;
PetscLogEvent ID_solve_getrs = -1;
static PetscBool log_active_solve = 0;
#endif


/*
 * @param[incA]: Stride value while accessing elements of 'A'.
 * @param[incB]: Stride value while accessing elements of 'B'.
 * @param[incOut]: Stride value while accessing elements of 'out'.
 */
void solve(PetscScalar* __restrict__ out, const PetscScalar* __restrict__ A, const PetscScalar* __restrict__ B, PetscBLASInt N,
           PetscBLASInt incA, PetscBLASInt incB, PetscBLASInt incOut)
{
    PetscScalar* out_proxy;  /// output laid-out with unit stride, expected by LAPACK
    PetscInt N_sq = N*N;
    PetscInt one = 1;
    PetscLogIsActive(&log_active_solve);
    if (log_active_solve){PetscLogEventBegin(ID_solve_memcpy,0,0,0,0);}
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));

    if (incOut == 1)
      out_proxy = out;
    else
      out_proxy = (N <= BUF_SIZE) ? out_proxy_buffer : malloc(N*sizeof(*out));

    BLAScopy_(&N, B, &incB, out_proxy, &one);

    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N_sq*sizeof(*Awork));
    BLAScopy_(&N_sq, A, &incA, Awork, &one);
    if (log_active_solve){PetscLogEventEnd(ID_solve_memcpy,0,0,0,0);}

    PetscBLASInt NRHS = 1;
    const char T = 'T';
    if (log_active_solve){PetscLogEventBegin(ID_solve_getrf,0,0,0,0);}
    LAPACKgetrf_(&N, &N, Awork, &N, ipiv, &info);
    if (log_active_solve){PetscLogEventEnd(ID_solve_getrf,0,0,0,0);}

    if(info == 0){
        if (log_active_solve){PetscLogEventBegin(ID_solve_getrs,0,0,0,0);}
        LAPACKgetrs_(&T, &N, &NRHS, Awork, &N, ipiv, out_proxy, &N, &info);

        if (out != out_proxy)
            BLAScopy_(&N, out_proxy, &one, out, &incOut);

        if (log_active_solve){PetscLogEventEnd(ID_solve_getrs,0,0,0,0);}
    }

    if(info != 0){
        fprintf(stderr, "Gesv throws nonzero info.");
        abort();
    }

    if (ipiv != ipiv_buffer)
      free(ipiv);

    if (Awork != work_buffer)
      free(Awork);

    if ((out_proxy != out) && (out_proxy != out_proxy_buffer))
      free(out_proxy);
}
