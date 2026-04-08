//
// Created by sauron on 08.04.26.
//
#include "Bspline1D.h"
#include "matplotlibcpp.h"
#include <iostream>
#include <math.h>
#include <vector>

namespace plt = matplotlibcpp;


float single_sin(float x) {
    return sinf(0.5*2.1416*x);
}

int main(void) {
    unsigned int num_points = 10;
    // create the values array
    //float *random_values = new float[num_points];
    float *values_single_sin = new float[num_points];


    const float delta = 1.0f;
    for (unsigned int i=0; i<num_points; i++) {
        //random_values[i] = (float) rand() / RAND_MAX; // random values between 0 and 1
        values_single_sin[i] = single_sin(i*delta);
    }

    // Bspline1D *interpolator = new Bspline1D(random_values, num_points);

    Bspline1D *interpolator = new Bspline1D(values_single_sin, num_points);
    // test values
    float x = 4.5;

    float u = interpolator->interpolate(x);



    // print resutls
    std::cout << "input values:" << std::endl;
    std::cout << x << std::endl;
    std::cout << "new values:" << u << std::endl;

    std::vector<double> values_y(num_points);
    std::vector<double> new_values_y(1);
    std::vector<double> values_x(num_points);
    std::vector<double> new_values_x(1);

    for (unsigned int i=0; i<num_points; i++) {
        values_x[i] = i*delta;
    }


    values_y.assign(values_single_sin, values_single_sin + num_points);
    new_values_y[0] = u;
    new_values_x[0] = x;

    plt::plot(values_x, values_y, "ro-");
    plt::plot(new_values_x, new_values_y, "go");
    // plt::plot({1,2,3,4});
    // plt::plot({0,1,2,3,4,5,6,7,8,9}, values_single_sin, "ro-");
    // plt::plot({x}, {u}, "go");
    plt::show();

}