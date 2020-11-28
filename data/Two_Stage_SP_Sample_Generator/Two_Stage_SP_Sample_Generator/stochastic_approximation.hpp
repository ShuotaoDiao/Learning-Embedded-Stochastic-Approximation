//
//  stochastic_approximation.hpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 11/7/20.
//

#ifndef stochastic_approximation_hpp
#define stochastic_approximation_hpp

#include <stdio.h>
#include <ilcplex/ilocplex.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#include "sample_generator.hpp"
#include "utils.hpp"
// compute the subgradient of the second stage value function
std::vector<double> subgradient_baa99(const std::vector<double>& x, const std::vector<double>& omega);

// stochastic approximation main body
// stochastic approximation for solving baa99
std::vector<double> sa_baa99(const std::vector<double>& x_init, double initial_stepsize, double num_iteration);

// projection
std::vector<double> projection_baa99(const std::vector<double>& x);

// sample_loader
std::vector<double> sample_loader_baa99();
std::vector<double> sample_loader_landS();
std::vector<double> sample_loader_landS2();
std::vector<double> sample_loader_landS3();
std::vector<double> sample_loader_shipment();

// benchmark function to evalutate the performance of SA algorithm
double benchmark_baa99(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set);
double benchmark_landS(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set);
double benchmark_landS2(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set);
double benchmark_shipment(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set);


// evaluate benchmark function value
std::vector<double> benchmark_baa99_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set);
std::vector<double> benchmark_landS_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set);
std::vector<double> benchmark_landS2_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set);
std::vector<double> benchmark_shipment_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set);


// benchmark sample set
std::vector<std::vector<double>> benchmark_sample_baa99(int num_samples);
std::vector<std::vector<double>> benchmark_sample_landS(int num_samples);
std::vector<std::vector<double>> benchmark_sample_landS2(int num_samples);
std::vector<std::vector<double>> benchmark_sample_landS3(int num_samples);
std::vector<std::vector<double>> benchmark_sample_shipment(int num_samples);

#endif /* stochastic_approximation_hpp */
