//
// Created by sauron on 08.04.26.
//

#ifndef PMCONV_BSPLINE1D_H
#define PMCONV_BSPLINE1D_H
#include "TridiagonalMatrix.h"


class Bspline1D
{
    private:
        unsigned int degree;
        unsigned int num_points;

        // solution
        float *coeficients;
        // tridiagonal matrix
        TridiagonalMatrix *tridiagonal_matrix;

        // spline
        float inline basis_function(float x);
    public:
        // contructor
        Bspline1D(float *values, unsigned int num_points);
        // destruction
        ~Bspline1D();

        float interpolate(float x);

};



#endif //PMCONV_BSPLINE1D_H
