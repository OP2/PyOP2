



static inline void mass_cell_integral_0_otherwise (double A[3][3] , double **vertex_coordinates )
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
#pragma pyop2 itspace
    for (int j = 0; j<3; j++)
    {
#pragma pyop2 itspace
      for (int k = 0; k<3; k++)
      {
        A[j][k] += (det*W3[ip]*FE0[ip][k]*FE0[ip][j]);

      }

    }

  }


}
