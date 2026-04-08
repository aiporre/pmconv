//
// Created by sauron on 08.04.26.
//

#include "Bspline1D.h"
#include <math.h>

float inline Bspline1D::basis_function(float x) {
    float u = fabsf(x);
    float value = 0;
    if (u<1) {
        value = (4 - 6*u*u + 3*u*u*u) / 6;
    } else if  (u<2) {
        value = (2-u)*(2-u)*(2-u) / 6;
    } else {
        value = 0;
    }
    return value;
}

// constructor
Bspline1D::Bspline1D(float *values, unsigned int num_points) {
    this->degree = 3; // cubic B-spline
    this->num_points = num_points;
    const float lower_diag = (float) 1/6;
    const float upper_diag = (float) 1/6;
    const float center_diag = (float) 4/6;
    this->tridiagonal_matrix = new TridiagonalMatrix(num_points, lower_diag, center_diag, upper_diag);
    this->coeficients = this->tridiagonal_matrix->system(values);
}

// destructor
Bspline1D::~Bspline1D() {
    delete tridiagonal_matrix;
}

float Bspline1D::interpolate(float x) {

    int basis_center_domain = (int) fmin(fmax(roundf(x), 2), this->num_points-3);

    float h = 0;

    for (int j=basis_center_domain-2; j<=basis_center_domain+2; j++) {
        h += this->coeficients[j] * basis_function(x - j);
    }

    return h;
}
