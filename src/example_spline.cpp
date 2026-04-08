//
// Created by sauron on 08.04.26.
//
#include "Bspline1D.h"
#include <iostream>
#include <math.h>


float single_sin(float x) {
    return sinf(0.5*2.1416*x);
}

int main(void) {
    unsigned int num_points = 10;
    // create the values array
    float *random_values = new float[num_points];



    for (unsigned int i=0; i<num_points; i++) {
        random_values[i] = (float) rand() / RAND_MAX; // random values between 0 and 1
    }

    Bspline1D *interpolator = new Bspline1D(random_values, num_points);

    // test values
    float x = 4.5;

    float u = interpolator->interpolate(x);



    // print resutls
    std::cout << "input values:" << std::endl;
    std::cout << x << std::endl;
    std::cout << "new values:" << u << std::endl;

}