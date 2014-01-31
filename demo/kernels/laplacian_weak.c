static inline void rhs_exterior_facet_integral_0_2 (double A[3] , double **vertex_coordinates , double **w0 , double **w1 , unsigned int *facet_p )
{
  unsigned int facet = *facet_p;
  double J[4];
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];;
  double K[4];
  double detJ;
  detJ = J[0]*J[3] - J[1]*J[2]; K[0] = J[3] / detJ; K[1] = -J[1] / detJ; K[2] = -J[2] / detJ; K[3] = J[0] / detJ;;
  unsigned int edge_vertices[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  const unsigned int v0 = edge_vertices[facet][0];
  const unsigned int v1 = edge_vertices[facet][1];
  const double dx0 = vertex_coordinates[v1 + 0][0] - vertex_coordinates[v0 + 0][0];
  const double dx1 = vertex_coordinates[v1 + 3][0] - vertex_coordinates[v0 + 3][0];
  const double det = sqrt(dx0*dx0 + dx1*dx1);
  static const double W2[2] = {0.5, 0.5};
  static const double FE0_f0[2][3] = {{0.0, 0.788675134594813, 0.211324865405187},
  {0.0, 0.211324865405187, 0.788675134594813}};
  static const double FE0_f1[2][3] = {{0.788675134594813, 0.0, 0.211324865405187},
  {0.211324865405187, 0.0, 0.788675134594813}};
  static const double FE0_f2[2][3] = {{0.788675134594813, 0.211324865405187, 0.0},
  {0.211324865405187, 0.788675134594813, 0.0}};
  switch (facet)
  {
  case 0:
    {
    for (int ip = 0; ip<2; ip++)
    {
      double F0 = 0.0;
      for (int r = 0; r<3; r++)
      {
        F0 += (w1[r][0]*FE0_f0[ip][r]);
      }
#pragma pyop2 itspace
      for (int j = 0; j<3; j++)
      {
        A[j] += (det*W2[ip]*FE0_f0[ip][j]*F0);
      }
    }
      break;
    }
  case 1:
    {
    for (int ip = 0; ip<2; ip++)
    {
      double F0 = 0.0;
      for (int r = 0; r<3; r++)
      {
        F0 += (w1[r][0]*FE0_f1[ip][r]);
      }
#pragma pyop2 itspace
      for (int j = 0; j<3; j++)
      {
        A[j] += (det*W2[ip]*FE0_f1[ip][j]*F0);
      }
    }
      break;
    }
  case 2:
    {
    for (int ip = 0; ip<2; ip++)
    {
      double F0 = 0.0;
      for (int r = 0; r<3; r++)
      {
        F0 += (w1[r][0]*FE0_f2[ip][r]);
      }
#pragma pyop2 itspace
      for (int j = 0; j<3; j++)
      {
        A[j] += (det*W2[ip]*FE0_f2[ip][j]*F0);
      }
    }
      break;
    }
  }
}