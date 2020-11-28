//
//  main.cpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 10/30/20.
//

#include <iostream>
#include <ilcplex/ilocplex.h>
#include <vector>
#include <string>

#include "sample_generator.hpp"
#include "stochastic_approximation.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    int num_samples;
    std::string outputPath;
    // baa99
    //test_baa99();
    /*
    num_samples = 10000;
    outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/baa99_sample4.csv";
    baa99_sample_generator(outputPath, num_samples);
     */
    // landS
    //test_landS();
    // cep
    //test_cep();
    //num_samples = 10000;
    //outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/cep_sample4.csv";
    //cep_sample_generator(outputPath, num_samples);
    // landS
    /*
    num_samples = 10000;
    outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/landS_sample5.csv";
    landS_sample_generator(outputPath, num_samples);
     */
    // landS2
    /*
    num_samples = 10000;
    outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/landS2_sample4.csv";
    landS2_sample_generator(outputPath, num_samples);
     */
    // landS3
    /*
    num_samples = 10000;
    outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/landS3_sample4.csv";
    landS3_sample_generator(outputPath, num_samples);
     */
    // shipment
    /*
    num_samples = 10000;
    outputPath = "/Users/sonny/Documents/Courses/CSCI566/experiment/training_set/shipment_sample4.csv";
    shipment_sample_generator(outputPath, num_samples);
     */
    // test on SA
    /*
    std::vector<double> x;
    x.push_back(50);
    x.push_back(50);
    int num_iteration = 100;
    double initial_stepsize = 10.0;
    std::vector<double> x_est = sa_baa99(x, initial_stepsize, num_iteration);
    */
    // evaluate solution quality
    // baa99
    /*
    int num_benchmark_samples = 2000;
    // generate benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_baa99(num_benchmark_samples);
    // seed = 123
    std::cout << "seed = 123" << std::endl;
    std::string lesa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_baa99.txt";
    std::string sa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_sa_baa99.txt";
    std::string lesa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_baa99.txt";
    std::string sa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_sa_baa99.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record1 = read_solution(lesa_path1);
    std::vector<std::vector<double>> sa_solution_record1 = read_solution(sa_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa1 = benchmark_baa99_eval(lesa_solution_record1, benchmark_set);
    std::vector<double> benchmark_values_sa1 = benchmark_baa99_eval(sa_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa1, lesa_write_path1);
    write_benchmark_value(benchmark_values_sa1, sa_write_path1);
    
    // seed = 5
    std::cout << "seed = 5" << std::endl;
    std::string lesa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_baa99.txt";
    std::string sa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_sa_baa99.txt";
    std::string lesa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_baa99.txt";
    std::string sa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_sa_baa99.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record2 = read_solution(lesa_path2);
    std::vector<std::vector<double>> sa_solution_record2 = read_solution(sa_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa2 = benchmark_baa99_eval(lesa_solution_record2, benchmark_set);
    std::vector<double> benchmark_values_sa2 = benchmark_baa99_eval(sa_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa2, lesa_write_path2);
    write_benchmark_value(benchmark_values_sa2, sa_write_path2);
    
    // seed = 37
    std::cout << "seed = 37" << std::endl;
    std::string lesa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_baa99.txt";
    std::string sa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_sa_baa99.txt";
    std::string lesa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_baa99.txt";
    std::string sa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_sa_baa99.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record3 = read_solution(lesa_path3);
    std::vector<std::vector<double>> sa_solution_record3 = read_solution(sa_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa3 = benchmark_baa99_eval(lesa_solution_record3, benchmark_set);
    std::vector<double> benchmark_values_sa3 = benchmark_baa99_eval(sa_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa3, lesa_write_path3);
    write_benchmark_value(benchmark_values_sa3, sa_write_path3);
     */
    // landS
    /*
    int num_benchmark_samples = 2000;
    // generate benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_landS(num_benchmark_samples);
    // seed = 123
    std::cout << "seed = 123" << std::endl;
    std::string lesa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_landS.txt";
    std::string sa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_sa_landS.txt";
    std::string lesa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_landS.txt";
    std::string sa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_sa_landS.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record1 = read_solution(lesa_path1);
    std::vector<std::vector<double>> sa_solution_record1 = read_solution(sa_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa1 = benchmark_landS_eval(lesa_solution_record1, benchmark_set);
    std::vector<double> benchmark_values_sa1 = benchmark_landS_eval(sa_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa1, lesa_write_path1);
    write_benchmark_value(benchmark_values_sa1, sa_write_path1);
    
    // seed = 5
    std::cout << "seed = 5" << std::endl;
    std::string lesa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_landS.txt";
    std::string sa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_sa_landS.txt";
    std::string lesa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_landS.txt";
    std::string sa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_sa_landS.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record2 = read_solution(lesa_path2);
    std::vector<std::vector<double>> sa_solution_record2 = read_solution(sa_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa2 = benchmark_landS_eval(lesa_solution_record2, benchmark_set);
    std::vector<double> benchmark_values_sa2 = benchmark_landS_eval(sa_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa2, lesa_write_path2);
    write_benchmark_value(benchmark_values_sa2, sa_write_path2);
    
    // seed = 37
    std::cout << "seed = 37" << std::endl;
    std::string lesa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_landS.txt";
    std::string sa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_sa_landS.txt";
    std::string lesa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_landS.txt";
    std::string sa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_sa_landS.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record3 = read_solution(lesa_path3);
    std::vector<std::vector<double>> sa_solution_record3 = read_solution(sa_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa3 = benchmark_landS_eval(lesa_solution_record3, benchmark_set);
    std::vector<double> benchmark_values_sa3 = benchmark_landS_eval(sa_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa3, lesa_write_path3);
    write_benchmark_value(benchmark_values_sa3, sa_write_path3);
     */
    // landS2
    /*
    int num_benchmark_samples = 2000;
    // generate benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_landS2(num_benchmark_samples);
    // seed = 123
    std::cout << "seed = 123" << std::endl;
    std::string lesa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_landS2.txt";
    std::string sa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_sa_landS2.txt";
    std::string lesa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_landS2.txt";
    std::string sa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_sa_landS2.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record1 = read_solution(lesa_path1);
    std::vector<std::vector<double>> sa_solution_record1 = read_solution(sa_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa1 = benchmark_landS2_eval(lesa_solution_record1, benchmark_set);
    std::vector<double> benchmark_values_sa1 = benchmark_landS2_eval(sa_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa1, lesa_write_path1);
    write_benchmark_value(benchmark_values_sa1, sa_write_path1);
    // seed = 5
    std::cout << "seed = 5" << std::endl;
    std::string lesa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_landS2.txt";
    std::string sa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_sa_landS2.txt";
    std::string lesa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_landS2.txt";
    std::string sa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_sa_landS2.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record2 = read_solution(lesa_path2);
    std::vector<std::vector<double>> sa_solution_record2 = read_solution(sa_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa2 = benchmark_landS2_eval(lesa_solution_record2, benchmark_set);
    std::vector<double> benchmark_values_sa2 = benchmark_landS2_eval(sa_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa2, lesa_write_path2);
    write_benchmark_value(benchmark_values_sa2, sa_write_path2);
    // seed = 37
    std::cout << "seed = 37" << std::endl;
    std::string lesa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_landS2.txt";
    std::string sa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_sa_landS2.txt";
    std::string lesa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_landS2.txt";
    std::string sa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_sa_landS2.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record3 = read_solution(lesa_path3);
    std::vector<std::vector<double>> sa_solution_record3 = read_solution(sa_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa3 = benchmark_landS2_eval(lesa_solution_record3, benchmark_set);
    std::vector<double> benchmark_values_sa3 = benchmark_landS2_eval(sa_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa3, lesa_write_path3);
    write_benchmark_value(benchmark_values_sa3, sa_write_path3);
     */
    // landS3
    int num_benchmark_samples = 2000;
    // generate benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_landS3(num_benchmark_samples);
    // seed = 123
    std::cout << "seed = 123" << std::endl;
    std::string lesa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_landS3.txt";
    std::string sa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_sa_landS3.txt";
    std::string lesa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_landS3.txt";
    std::string sa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_sa_landS3.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record1 = read_solution(lesa_path1);
    std::vector<std::vector<double>> sa_solution_record1 = read_solution(sa_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa1 = benchmark_landS2_eval(lesa_solution_record1, benchmark_set);
    std::vector<double> benchmark_values_sa1 = benchmark_landS2_eval(sa_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa1, lesa_write_path1);
    write_benchmark_value(benchmark_values_sa1, sa_write_path1);
    // seed = 5
    std::cout << "seed = 5" << std::endl;
    std::string lesa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_landS3.txt";
    std::string sa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_sa_landS3.txt";
    std::string lesa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_landS3.txt";
    std::string sa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_sa_landS3.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record2 = read_solution(lesa_path2);
    std::vector<std::vector<double>> sa_solution_record2 = read_solution(sa_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa2 = benchmark_landS2_eval(lesa_solution_record2, benchmark_set);
    std::vector<double> benchmark_values_sa2 = benchmark_landS2_eval(sa_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa2, lesa_write_path2);
    write_benchmark_value(benchmark_values_sa2, sa_write_path2);
    // seed = 37
    std::cout << "seed = 37" << std::endl;
    std::string lesa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_landS3.txt";
    std::string sa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_sa_landS3.txt";
    std::string lesa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_landS3.txt";
    std::string sa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_sa_landS3.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record3 = read_solution(lesa_path3);
    std::vector<std::vector<double>> sa_solution_record3 = read_solution(sa_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa3 = benchmark_landS2_eval(lesa_solution_record3, benchmark_set);
    std::vector<double> benchmark_values_sa3 = benchmark_landS2_eval(sa_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa3, lesa_write_path3);
    write_benchmark_value(benchmark_values_sa3, sa_write_path3);
    // landS3 MAML
    //int num_benchmark_samples = 2000;
    // generate benchmark set
    //std::vector<std::vector<double>> benchmark_set = benchmark_sample_landS3(num_benchmark_samples);
    // seed = 123
    std::cout << "MAML" << std::endl;
    std::cout << "seed = 123" << std::endl;
    std::string lesa_maml_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_landS3_maml.txt";
    std::string lesa_maml_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_landS3_maml.txt";
    std::vector<std::vector<double>> lesa_maml_solution_record1 = read_solution(lesa_maml_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_maml_values_lesa1 = benchmark_landS2_eval(lesa_maml_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_maml_values_lesa1, lesa_maml_write_path1);
    
    //
    std::cout << "seed = 5" << std::endl;
    std::string lesa_maml_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_landS3_maml.txt";
    std::string lesa_maml_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_landS3_maml.txt";
    std::vector<std::vector<double>> lesa_maml_solution_record2 = read_solution(lesa_maml_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_maml_values_lesa2 = benchmark_landS2_eval(lesa_maml_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_maml_values_lesa2, lesa_maml_write_path2);
    
    //
    std::cout << "seed = 37" << std::endl;
    std::string lesa_maml_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_landS3_maml.txt";
    std::string lesa_maml_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_landS3_maml.txt";
    std::vector<std::vector<double>> lesa_maml_solution_record3 = read_solution(lesa_maml_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_maml_values_lesa3 = benchmark_landS2_eval(lesa_maml_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_maml_values_lesa3, lesa_maml_write_path3);
    // shipment
    /*
    int num_benchmark_samples = 2000;
    // generate benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_shipment(num_benchmark_samples);
    // seed = 123
    std::cout << "seed = 123" << std::endl;
    std::string lesa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_lesa_shipment.txt";
    std::string sa_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/x_record_sa_shipment.txt";
    std::string lesa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_lesa_shipment.txt";
    std::string sa_write_path1 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed123/evaluation/benchmark_sa_shipment.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record1 = read_solution(lesa_path1);
    std::vector<std::vector<double>> sa_solution_record1 = read_solution(sa_path1);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa1 = benchmark_shipment_eval(lesa_solution_record1, benchmark_set);
    std::vector<double> benchmark_values_sa1 = benchmark_shipment_eval(sa_solution_record1, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa1, lesa_write_path1);
    write_benchmark_value(benchmark_values_sa1, sa_write_path1);
    
    // seed = 5
    std::cout << "seed = 5" << std::endl;
    std::string lesa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_lesa_shipment.txt";
    std::string sa_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/x_record_sa_shipment.txt";
    std::string lesa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_lesa_shipment.txt";
    std::string sa_write_path2 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed5/evaluation/benchmark_sa_shipment.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record2 = read_solution(lesa_path2);
    std::vector<std::vector<double>> sa_solution_record2 = read_solution(sa_path2);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa2 = benchmark_shipment_eval(lesa_solution_record2, benchmark_set);
    std::vector<double> benchmark_values_sa2 = benchmark_shipment_eval(sa_solution_record2, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa2, lesa_write_path2);
    write_benchmark_value(benchmark_values_sa2, sa_write_path2);
    
    // seed = 37
    std::cout << "seed = 37" << std::endl;
    std::string lesa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_lesa_shipment.txt";
    std::string sa_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/x_record_sa_shipment.txt";
    std::string lesa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_lesa_shipment.txt";
    std::string sa_write_path3 = "/Users/sonny/Documents/Courses/CSCI566/experiment2/seed37/evaluation/benchmark_sa_shipment.txt";
    // read solution
    std::vector<std::vector<double>> lesa_solution_record3 = read_solution(lesa_path3);
    std::vector<std::vector<double>> sa_solution_record3 = read_solution(sa_path3);
    // evaluate benchmark values
    std::vector<double> benchmark_values_lesa3 = benchmark_shipment_eval(lesa_solution_record3, benchmark_set);
    std::vector<double> benchmark_values_sa3 = benchmark_shipment_eval(sa_solution_record3, benchmark_set);
    // write record
    write_benchmark_value(benchmark_values_lesa3, lesa_write_path3);
    write_benchmark_value(benchmark_values_sa3, sa_write_path3);
     */
    return 0;
}
