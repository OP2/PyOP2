#ifndef _SPARSITY_UTILS_H
#define _SPARSITY_UTILS_H

#include "op_lib_core.h"

#ifdef __cplusplus
extern "C" {
#endif

void build_sparsity_pattern ( int rmult, int cmult, int nrows, int nmaps,
                              op_map * rowmaps, op_map * colmaps,
                              int ** row_lgmap, int ** col_lgmap,
                              int ** nnz, int ** o_nnz,
                              int ** rowptr, int ** colidx,
                              int ** o_rowptr, int ** o_colidx );

#ifdef __cplusplus
}
#endif

#endif // _SPARSITY_UTILS_H
