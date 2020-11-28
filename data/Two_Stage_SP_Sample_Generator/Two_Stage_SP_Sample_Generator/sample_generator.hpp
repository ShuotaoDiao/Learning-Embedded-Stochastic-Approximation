//
//  sample_generator.hpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 10/30/20.
//

#ifndef sample_generator_hpp
#define sample_generator_hpp

#include <stdio.h>
#include <ilcplex/ilocplex.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#include "utils.hpp"

// data structure for function value
struct h_function{
    double value = 0;
    IloBool flag_solvable;
};


// output the values of second stage problem
double baa99_2ndStageValue(const std::vector<double>& x, const std::vector<double>& d, bool flag_verbose);
h_function landS_2ndStageValue(const std::vector<double>& x, const std::vector<double>& omega, bool flag_verbose);
h_function landS2_2ndStageValue(const std::vector<double>& x, const std::vector<double>& omega, bool flag_verbose);
h_function cep_2ndStageValue(const std::vector<double>& xM, const std::vector<double>& zM, const std::vector<double>& omega, bool flag_verbose);
h_function shipment_2ndStageValue(const std::vector<double>& x, const std::vector<double>& xi);

// generate samples with labels
void baa99_sample_generator(const std::string& outputPath, int num_samples);
void landS_sample_generator(const std::string& outputPath, int num_samples);
void landS2_sample_generator(const std::string& outputPath, int num_samples);
void landS3_sample_generator(const std::string& outputPath, int num_samples);
void cep_sample_generator(const std::string& outputPath, int num_samples);
void shipment_sample_generator(const std::string& outputPath, int num_samples);


// projection
std::vector<double> landS_projection(const std::vector<double>& x);
std::vector<double> cep_projection(const std::vector<double>& xM, const std::vector<double>& zM);
std::vector<double> shipment_projection(const std::vector<double>& x); // not implemented since it is trivial

// test functions
void test_baa99();
void test_landS();
void test_cep();
#endif /* sample_generator_hpp */
