static inline void laplacian_cell_integral_0_otherwise (double A[3][3] , double **vertex_coordinates )
{
  double J[4];
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];;
  double K[4];
  double detJ;
  detJ = J[0]*J[3] - J[1]*J[2]; K[0] = J[3] / detJ; K[1] = -J[1] / detJ; K[2] = -J[2] / detJ; K[3] = J[0] / detJ;;
  const double det = fabs(detJ);
  static const double W1 = 0.5;
  static const double FE0_D10[1][3] = {{-1.0, 1.0, 0.0}};
  static const double FE0_D01[1][3] = {{-1.0, 0.0, 1.0}};
#pragma pyop2 itspace
  for (int j = 0; j<3; j++)
  {
#pragma pyop2 itspace
    for (int k = 0; k<3; k++)
    {
      A[j][k] += (((((K[1]*FE0_D10[0][k])+(K[3]*FE0_D01[0][k]))*((K[1]*FE0_D10[0][j])+(K[3]*FE0_D01[0][j])))+(((K[0]*FE0_D10[0][k])+(K[2]*FE0_D01[0][k]))*((K[0]*FE0_D10[0][j])+(K[2]*FE0_D01[0][j]))))*det*W1);
    }
  }
}