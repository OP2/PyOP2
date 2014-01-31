static inline void rhs_cell_integral_0_otherwise (double A[3] , double **vertex_coordinates , double **w0 )
{
  double J[4];
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];;
  double K[4];
  double detJ;
  detJ = J[0]*J[3] - J[1]*J[2]; K[0] = J[3] / detJ; K[1] = -J[1] / detJ; K[2] = -J[2] / detJ; K[3] = J[0] / detJ;;
  const double det = fabs(detJ);
  static const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
  static const double FE0[3][3] = {{0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.166666666666667, 0.666666666666667, 0.166666666666667}};
  for (int ip = 0; ip<3; ip++)
  {
    double F0 = 0.0;
    for (int r = 0; r<3; r++)
    {
      F0 += (w0[r][0]*FE0[ip][r]);
    }
#pragma pyop2 itspace
    for (int j = 0; j<3; j++)
    {
      A[j] += (det*W3[ip]*FE0[ip][j]*F0);
    }
  }
}