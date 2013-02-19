#include <vector>
#include <set>
#include "sparsity_utils.h"

void build_sparsity_pattern ( int rmult, int cmult, int nrows, int nmaps,
                              op_map * rowmaps, op_map * colmaps,
                              int ** _nnz, int ** _o_nnz,
                              int ** _rowptr, int ** _colidx,
                              int ** _o_rowptr, int ** _o_colidx )
{
  // Create and populate auxiliary data structure: for each element of
  // the from set, for each row pointed to by the row map, add all
  // columns pointed to by the col map
  int lsize = nrows*rmult;
  std::vector< std::set< int > > s_diag(lsize);
  std::vector< std::set< int > > s_odiag(lsize);

  for ( int m = 0; m < nmaps; m++ ) {
    op_map rowmap = rowmaps[m];
    op_map colmap = colmaps[m];
    int rsize = rowmap->from->size + rowmap->from->exec_size;
    for ( int e = 0; e < rsize; ++e ) {
      for ( int i = 0; i < rowmap->dim; ++i ) {
        for ( int r = 0; r < rmult; r++ ) {
          int row = rmult * rowmap->map[i + e*rowmap->dim] + r;
          // NOTE: this hides errors due to invalid map entries
          if ( row < lsize ) { // ignore values inside the MPI halo region
            for ( int c = 0; c < cmult; c++ ) {
              for ( int d = 0; d < colmap->dim; d++ ) {
                int entry = cmult * colmap->map[d + e * colmap->dim] + c;
                if ( entry < lsize ) {
                  s_diag[row].insert(entry);
                } else {
                  s_odiag[row].insert(entry);
                }
              }
            }
          }
        }
      }
    }
  }

  // Create final sparsity structure
  int * nnz = (int*)malloc(lsize * sizeof(int));
  int * o_nnz = (int *)malloc(lsize * sizeof(int));
  int * rowptr = (int*)malloc((lsize+1) * sizeof(int));
  int * o_rowptr = (int*)malloc((lsize+1) * sizeof(int));
  rowptr[0] = 0;
  o_rowptr[0] = 0;
  for ( int row = 0; row < lsize; ++row ) {
    nnz[row] = s_diag[row].size();
    o_nnz[row] = s_odiag[row].size();
    rowptr[row+1] = rowptr[row] + nnz[row];
    o_rowptr[row+1] = o_rowptr[row] + o_nnz[row];
  }
  int * colidx = (int*)malloc(rowptr[lsize] * sizeof(int));
  int * o_colidx = (int*)malloc(o_rowptr[lsize] * sizeof(int));
  // Note: elements in a set are always sorted, so no need to sort colidx
  for ( int row = 0; row < lsize; ++row ) {
    std::copy(s_diag[row].begin(), s_diag[row].end(), colidx + rowptr[row]);
    std::copy(s_odiag[row].begin(), s_odiag[row].end(), o_colidx + o_rowptr[row]);
  }
  *_nnz = nnz;
  *_o_nnz = o_nnz;
  *_rowptr = rowptr;
  *_o_rowptr = o_rowptr;
  *_colidx = colidx;
  *_o_colidx = o_colidx;
}
