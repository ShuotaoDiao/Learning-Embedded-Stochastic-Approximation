//
//  utils.cpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 11/17/20.
//

#include "utils.hpp"
// read solution record form text file
std::vector<std::vector<double>> read_solution(const std::string& path) {
    std::vector<std::vector<double>> solution_record;
    // create a read file object
    // x1, x2, x3,...
    std::ifstream readFile(path);
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            std::vector<double> solution;
            while (getline(ss1, line1, ',')) {
                std::stringstream ss2(line1);
                double value;
                ss2 >> value;
                solution.push_back(value);
            }
            solution_record.push_back(solution);
        }
    }
    readFile.close();
    std::cout << "Finish reading solutions." << std::endl;
    return solution_record;
}

// read distribution
std::vector<discrete_prob> read_discrete_distribution(const std::string& path) {
    std::vector<discrete_prob> discrete_distribution;
    // create a read file object
    // value,prob
    std::ifstream readFile(path);
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            discrete_prob instance;
            int index = 0;
            while (getline(ss1, line1, ',')) {
                std::stringstream ss2(line1);
                if (index = 0){
                    double value;
                    ss2 >> value;
                    instance.value = value;
                }
                else {
                    double prob;
                    ss2 >> prob;
                    instance.prob = prob;
                }
                index += 1;
            }
            discrete_distribution.push_back(instance);
        }
    }
    return discrete_distribution;
}

// write benchmark values
void write_benchmark_value(const std::vector<double>& benchmark_values, const std::string& path) {
    // output the sample set
    std::fstream writeFile;
    writeFile.open(path, std::fstream::app);
    //
    int num_values = benchmark_values.size();
    for (int index = 0; index < num_values; ++index) {
        writeFile << benchmark_values[index] << std::endl;
    }
    std::cout << "Finish writing benchmark values." << std::endl;
    writeFile.close();
}
