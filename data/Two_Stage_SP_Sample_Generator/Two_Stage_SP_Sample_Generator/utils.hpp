//
//  utils.hpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 11/17/20.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "stochastic_approximation.hpp"

// struct for discrete random variables
struct discrete_prob {
    double value;
    double prob;
};

// read solution record form text file
std::vector<std::vector<double>> read_solution(const std::string& path);

// read distribution
std::vector<discrete_prob> read_discrete_distribution(const std::string& path);

// write benchmark values
void write_benchmark_value(const std::vector<double>& benchmark_values, const std::string& path);
#endif /* utils_hpp */
