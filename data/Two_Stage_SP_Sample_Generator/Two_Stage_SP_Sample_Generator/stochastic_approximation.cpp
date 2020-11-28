//
//  stochastic_approximation.cpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 11/7/20.
//

#include "stochastic_approximation.hpp"

// compute the subgradient of the second stage value function
std::vector<double> subgradient_baa99(const std::vector<double>& x, const std::vector<double>& omega) {
    int num_eq_cons = 4;
    // initialize CPLEX model environment
    IloEnv env;
    IloModel mod(env);
    // decision variable
    IloNumVar w11(env,0,IloInfinity,ILOFLOAT);
    IloNumVar w12(env,0,IloInfinity,ILOFLOAT);
    IloNumVar w22(env,0,IloInfinity,ILOFLOAT);
    IloNumVar v1(env,0,IloInfinity,ILOFLOAT);
    IloNumVar v2(env,0,IloInfinity,ILOFLOAT);
    IloNumVar u1(env,0,IloInfinity,ILOFLOAT);
    IloNumVar u2(env,0,IloInfinity,ILOFLOAT);
    mod.add(w11);
    mod.add(w12);
    mod.add(w22);
    mod.add(v1);
    mod.add(v2);
    mod.add(u1);
    mod.add(u2);
    // objective function
    IloExpr expr_obj(env);
    expr_obj = -8 * w11 - 4 * w12 - 4 * w22 + 0.2 * v1 + 0.2 * v2 + 10 * u1 + 10 * u2;
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    IloRangeArray constraintsEquality(env);
    IloExpr con_d1 = w11 + u1;
    constraintsEquality.add(con_d1 == omega[0]);
    IloExpr con_d2 = w12 + u2;
    constraintsEquality.add(con_d2 == omega[1]);
    IloExpr con_s1 = -x[0] + w11 + w12 + v1;
    constraintsEquality.add(con_s1 == 0);
    IloExpr con_s2 = -x[1] + w22 + v2;
    constraintsEquality.add(con_s2 == 0);
    mod.add(constraintsEquality);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool solvable_flag = cplex.solve();
    IloNumArray dual_equality(env);
    std::vector<double> dual_eq_mutiplier;
    if (solvable_flag == IloTrue) {
        cplex.getDuals(dual_equality,constraintsEquality);
        for (int index = 0; index < num_eq_cons; ++index) {
            double temp = - dual_equality[index];
            dual_eq_mutiplier.push_back(temp);
        }
    }
    // subgradient
    std::vector<double> subgradient;
    subgradient.push_back(-dual_eq_mutiplier[2]); // x1
    subgradient.push_back(-dual_eq_mutiplier[3]); // x2
    env.end();
    return subgradient;
}


// stochastic approximation for solving baa99
std::vector<double> sa_baa99(const std::vector<double>& x_init, double initial_stepsize, double num_iteration) {
    // set up random seed
    srand(123);
    int num_samples = 1000;
    // benchmark set
    std::vector<std::vector<double>> benchmark_set = benchmark_sample_baa99(num_samples);
    std::vector<double> x_cur;
    x_cur = x_init;
    std::vector<double> x_new;
    x_new.push_back(0);
    x_new.push_back(0);
    std::vector<double> record_benchmark_value;
    for (int it = 0; it < num_iteration; ++it) {
        // current step size
        double cur_stepsize = initial_stepsize / (it + 1.0);
        // draw a sample
        std::vector<double> omega = sample_loader_baa99();
        // compute the subgradient of the second stage value function
        std::vector<double> subgradient_cur = subgradient_baa99(x_cur, omega);
        // add up the subgradient in the first stage
        subgradient_cur[0] = subgradient_cur[0] + 4;
        subgradient_cur[1] = subgradient_cur[1] + 2;
        // gradient step
        for (int index = 0; index < 2; ++index) {
            x_new[index] = x_cur[index] - cur_stepsize * subgradient_cur[index];
            std::cout << x_new[index] << std::endl;
        }
        // projection
        std::vector<double> x_proj = projection_baa99(x_new);
        // update
        for (int index = 0; index < 2; ++index) {
            x_cur[index] = x_proj[index];
        }
        // output results
        std::cout << "====================================\n";
        std::cout << "it " << it << std::endl;
        for (int index = 0; index < 2; ++index) {
            std::cout << "x[" << index + 1<< "]: " << x_cur[index] << std::endl;
        }
        double temp_benchmark_value = benchmark_baa99(x_cur, benchmark_set);
        record_benchmark_value.push_back(temp_benchmark_value);
        std::cout << "Benchmark Value: " << temp_benchmark_value << std::endl;
        std::cout << "====================================\n";
    }
    // print out the benchmark values
    for (int index = 0; index < num_iteration; ++index) {
        std::cout << record_benchmark_value[index] << ", ";
    }
    std::cout << std::endl;
    return x_cur;
}

// projection
std::vector<double> projection_baa99(const std::vector<double>& x) {
    // initialize CPLEX model environment
    IloEnv env;
    IloModel mod(env);
    IloNumVar x1(env, 0, 217, ILOFLOAT);
    IloNumVar x2(env, 0, 217, ILOFLOAT);
    mod.add(x1);
    mod.add(x2);
    // objective function
    IloExpr expr_obj(env);
    expr_obj += 0.5 * (x1 - x[0]) * (x1 - x[0]) + 0.5 * (x2 - x[1]) * (x2 - x[1]);
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool solvable_flag = cplex.solve();
    std::vector<double> x_proj;
    if (solvable_flag == IloTrue) {
        x_proj.push_back(cplex.getValue(x1));
        x_proj.push_back(cplex.getValue(x2));
    }
    else {
        std::cout << "Warning: Problem is infeasible!" << std::endl;
    }
    return x_proj;
}


// sample_loader
std::vector<double> sample_loader_baa99() {
    // P(d1)
    double p_d1 = ((double) rand()) / RAND_MAX;
    // d1
    double d1 = 0;
    if (p_d1 < 0.04) {
        d1 = 17.75731865;
    }
    else if (p_d1 < 0.08) {
        d1 = 32.96224832;
    }
    else if (p_d1 < 0.12) {
        d1 = 43.68044355;
    }
    else if (p_d1 < 0.16) {
        d1 = 52.29173734;
    }
    else if (p_d1 < 0.20) {
        d1 = 59.67893765;
    }
    else if (p_d1 < 0.24) {
        d1 = 66.27551249;
    }
    else if (p_d1 < 0.28) {
        d1 = 72.33076402;
    }
    else if (p_d1 < 0.32) {
        d1 = 78.00434172;
    }
    else if (p_d1 < 0.36) {
        d1 = 83.40733268;
    }
    else if (p_d1 < 0.40) {
        d1 = 88.62275117;
    }
    else if (p_d1 < 0.44) {
        d1 = 93.71693266;
    }
    else if (p_d1 < 0.48) {
        d1 = 98.74655459;
    }
    else if (p_d1 < 0.52) {
        d1 = 103.7634931;
    }
    else if (p_d1 < 0.56) {
        d1 = 108.8187082;
    }
    else if (p_d1 < 0.60) {
        d1 = 113.9659517;
    }
    else if (p_d1 < 0.64) {
        d1 = 119.2660233;
    }
    else if (p_d1 < 0.68) {
        d1 = 124.7925174;
    }
    else if (p_d1 < 0.72) {
        d1 = 130.6406496;
    }
    else if (p_d1 < 0.76) {
        d1 = 136.9423425;
    }
    else if (p_d1 < 0.80) {
        d1 = 143.8948148;
    }
    else if (p_d1 < 0.84) {
        d1 = 151.8216695;
    }
    else if (p_d1 < 0.88) {
        d1 = 161.326406;
    }
    else if (p_d1 < 0.92) {
        d1 = 173.7895514;
    }
    else if (p_d1 < 0.96) {
        d1 = 194.0396804;
    }
    else {
        d1 = 216.3173937;
    }
    // P(d2)
    double p_d2 = ((double) rand()) / RAND_MAX;
    // d2
    double d2 = 0;
    if (p_d2 < 0.04) {
        d2 = 5.960319592;
    }
    else if (p_d2 < 0.08) {
        d2 = 26.21044859;
    }
    else if (p_d2 < 0.12) {
        d2 = 38.673594;
    }
    else if (p_d2 < 0.16) {
        d2 = 48.17833053;
    }
    else if (p_d2 < 0.20) {
        d2 = 56.10518525;
    }
    else if (p_d2 < 0.24) {
        d2 = 63.05765754;
    }
    else if (p_d2 < 0.28) {
        d2 = 69.35935045;
    }
    else if (p_d2 < 0.32) {
        d2 = 75.20748263;
    }
    else if (p_d2 < 0.36) {
        d2 = 80.73397668;
    }
    else if (p_d2 < 0.40) {
        d2 = 86.03404828;
    }
    else if (p_d2 < 0.44) {
        d2 = 91.18129176;
    }
    else if (p_d2 < 0.48) {
        d2 = 96.2365069;
    }
    else if (p_d2 < 0.52) {
        d2 = 101.2534454;
    }
    else if (p_d2 < 0.56) {
        d2 = 106.2830673;
    }
    else if (p_d2 < 0.60) {
        d2 = 111.3772488;
    }
    else if (p_d2 < 0.64) {
        d2 = 116.5926673;
    }
    else if (p_d2 < 0.68) {
        d2 = 121.9956583;
    }
    else if (p_d2 < 0.72) {
        d2 = 127.669236;
    }
    else if (p_d2 < 0.76) {
        d2 = 133.7244875;
    }
    else if (p_d2 < 0.80) {
        d2 = 140.3210624;
    }
    else if (p_d2 < 0.84) {
        d2 = 147.7082627;
    }
    else if (p_d2 < 0.88) {
        d2 = 156.3195565;
    }
    else if (p_d2 < 0.92) {
        d2 = 167.0377517;
    }
    else if (p_d2 < 0.96) {
        d2 = 182.2426813;
    }
    else {
        d2 = 216.3173937;
    }
    // generate d vector
    std::vector<double> cur_d;
    cur_d.push_back(d1);
    cur_d.push_back(d2);
    return cur_d;
}

std::vector<double> sample_loader_landS() {
    // generate omega
    double p_omega = ((double) rand()) / RAND_MAX;
    std::vector<double> cur_omega;
    if (p_omega < 0.3) {
        cur_omega.push_back(3.0);
    }
    else if (p_omega < 0.7) {
        cur_omega.push_back(5.0);
    }
    else {
        cur_omega.push_back(7.0);
    }
    return cur_omega;
}

std::vector<double> sample_loader_landS2() {
    // generate omega
    // omega 1
    double p_omega1 = ((double) rand()) / RAND_MAX;
    std::vector<double> cur_omega;
    if (p_omega1 < 0.25) {
        cur_omega.push_back(0.0);
    }
    else if (p_omega1 < 0.50) {
        cur_omega.push_back(0.96);
    }
    else if (p_omega1 < 0.75){
        cur_omega.push_back(2.96);
    }
    else {
        cur_omega.push_back(3.96);
    }
    // omega 2
    double p_omega2 = ((double) rand()) / RAND_MAX;
    if (p_omega2 < 0.25) {
        cur_omega.push_back(0.0);
    }
    else if (p_omega2 < 0.50) {
        cur_omega.push_back(0.96);
    }
    else if (p_omega2 < 0.75){
        cur_omega.push_back(2.96);
    }
    else {
        cur_omega.push_back(3.96);
    }
    // omega 3
    double p_omega3 = ((double) rand()) / RAND_MAX;
    if (p_omega3 < 0.25) {
        cur_omega.push_back(0.0);
    }
    else if (p_omega3 < 0.50) {
        cur_omega.push_back(0.96);
    }
    else if (p_omega3 < 0.75){
        cur_omega.push_back(2.96);
    }
    else {
        cur_omega.push_back(3.96);
    }
    return cur_omega;
}

std::vector<double> sample_loader_landS3() {
    // generate omega
    // omega 1
    double p_omega1 = ((double) rand()) / RAND_MAX;
    std::vector<double> cur_omega;
    for (int index = 1; index < 101; ++index) {
        if (p_omega1 < 0.01) {
            cur_omega.push_back(0.0);
            break;
        }
        else {
            if (p_omega1 >= 0.01 * ((double) (index - 1)) && p_omega1 <= 0.01 * ((double) index)) {
                double value = 0.04 * ((double) (index - 1));
                cur_omega.push_back(value);
                break;
            }
            else if (index == 100){ // extreme case
                cur_omega.push_back(3.9600);
                break;
            }
        }
    }
    
    // omega 2
    double p_omega2 = ((double) rand()) / RAND_MAX;
    for (int index = 1; index < 101; ++index) {
        if (p_omega2 < 0.01) {
            cur_omega.push_back(0.0);
            break;
        }
        else {
            if (p_omega2 >= 0.01 * ((double) (index - 1)) && p_omega2 <= 0.01 * ((double) index)) {
                double value = 0.04 * ((double) (index - 1));
                cur_omega.push_back(value);
                break;
            }
            else if (index == 100){ // extreme case
                cur_omega.push_back(3.9600);
                break;
            }
        }
    }
    // omega 3
    double p_omega3 = ((double) rand()) / RAND_MAX;
    for (int index = 1; index < 101; ++index) {
        if (p_omega3 < 0.01) {
            cur_omega.push_back(0.0);
            break;
        }
        else {
            if (p_omega3 >= 0.01 * ((double) (index - 1)) && p_omega3 < 0.01 * ((double) index)) {
                double value = 0.04 * ((double) (index - 1));
                cur_omega.push_back(value);
                break;
            }
            else if (index == 100){ // extreme case
                cur_omega.push_back(3.9600);
                break;
            }
        }
    }
    return cur_omega;
}

std::vector<double> sample_loader_shipment() {
    // generate omega
    std::vector<double> cur_omega;
    for (int index = 0; index < 12; ++index) {
        double p_omega = ((double) rand()) / RAND_MAX;
        double omega_instance = 0;
        for (int i = 0; i < 10; ++i) {
            if (p_omega >= 0.1 * i && p_omega < 0.1 * (i+1)) {
                omega_instance = 2.0 * i;
                cur_omega.push_back(omega_instance);
                break;
            }
        }
    }
    return cur_omega;
}


// benchmark function to evalutate the performance of SA algorithm
double benchmark_baa99(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set){
    int num_samples = sample_set.size();
    double second_stage_value = 0;
    for (int index = 0; index < num_samples; ++index) {
        double temp_second_stage_value = baa99_2ndStageValue(x,sample_set[index],false);
        second_stage_value += temp_second_stage_value / ((double) num_samples);
    }
    double total_value = second_stage_value;
    total_value += 4.0 * x[0] + 2.0 * x[1];
    return total_value;
}

double benchmark_landS(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set) {
    int num_samples = sample_set.size();
    double second_stage_value = 0;
    for (int index = 0; index < num_samples; ++index) {
        h_function cur_h = landS_2ndStageValue(x,sample_set[index],false);
        second_stage_value += cur_h.value / ((double) num_samples);
    }
    double total_value = second_stage_value;
    total_value += 10.0 * x[0] + 7.0 * x[1] + 16.0 * x[2] + 6.0 * x[3];
    return total_value;
}

double benchmark_landS2(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set) {
        int num_samples = sample_set.size();
        double second_stage_value = 0;
        for (int index = 0; index < num_samples; ++index) {
            h_function cur_h = landS2_2ndStageValue(x,sample_set[index],false);
            second_stage_value += cur_h.value / ((double) num_samples);
        }
        double total_value = second_stage_value;
        total_value += 10.0 * x[0] + 7.0 * x[1] + 16.0 * x[2] + 6.0 * x[3];
        return total_value;
}

double benchmark_shipment(const std::vector<double>& x, const std::vector<std::vector<double>>& sample_set) {
    int num_samples = sample_set.size();
    double second_stage_value = 0;
    for (int index = 0; index < num_samples; ++index) {
        h_function cur_h = shipment_2ndStageValue(x,sample_set[index]);
        second_stage_value += cur_h.value / ((double) num_samples);
    }
    double total_value = second_stage_value;
    total_value += 5.0 * x[0] + 5.0 * x[1] + 5.0 * x[2] + 5.0 * x[3];
    return total_value;
}


// evaluate benchmark function value
std::vector<double> benchmark_baa99_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set) {
    std::vector<double> benchmark_values;
    int num_solutions = x.size();
    for (int index = 0; index < num_solutions; ++index) {
        double cur_value = benchmark_baa99(x[index], sample_set);
        benchmark_values.push_back(cur_value);
    }
    return benchmark_values;
}

std::vector<double> benchmark_landS_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set) {
    std::vector<double> benchmark_values;
    int num_solutions = x.size();
    for (int index = 0; index < num_solutions; ++index) {
        double cur_value = benchmark_landS(x[index], sample_set);
        benchmark_values.push_back(cur_value);
    }
    return benchmark_values;
}

std::vector<double> benchmark_landS2_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set) {
    std::vector<double> benchmark_values;
    int num_solutions = x.size();
    for (int index = 0; index < num_solutions; ++index) {
        double cur_value = benchmark_landS2(x[index], sample_set);
        benchmark_values.push_back(cur_value);
    }
    return benchmark_values;
}

std::vector<double> benchmark_shipment_eval(const std::vector<std::vector<double>> x, const std::vector<std::vector<double>>& sample_set) {
    std::vector<double> benchmark_values;
    int num_solutions = x.size();
    for (int index = 0; index < num_solutions; ++index) {
        double cur_value = benchmark_shipment(x[index], sample_set);
        benchmark_values.push_back(cur_value);
    }
    return benchmark_values;
}


// benchmark sample set
std::vector<std::vector<double>> benchmark_sample_baa99(int num_samples) {
    std::vector<std::vector<double>> sample_set;
    for (int index = 0; index < num_samples; ++index) {
        std::vector<double> cur_sample = sample_loader_baa99();
        sample_set.push_back(cur_sample);
    }
    return sample_set;
}

std::vector<std::vector<double>> benchmark_sample_landS(int num_samples) {
    std::vector<std::vector<double>> sample_set;
    for (int index = 0; index < num_samples; ++index) {
        std::vector<double> cur_sample = sample_loader_landS();
        sample_set.push_back(cur_sample);
    }
    return sample_set;
}

std::vector<std::vector<double>> benchmark_sample_landS2(int num_samples) {
    std::vector<std::vector<double>> sample_set;
    for (int index = 0; index < num_samples; ++index) {
        std::vector<double> cur_sample = sample_loader_landS2();
        sample_set.push_back(cur_sample);
    }
    return sample_set;
}


std::vector<std::vector<double>> benchmark_sample_landS3(int num_samples) {
    std::vector<std::vector<double>> sample_set;
    for (int index = 0; index < num_samples; ++index) {
        std::vector<double> cur_sample = sample_loader_landS3();
        sample_set.push_back(cur_sample);
    }
    return sample_set;
}


std::vector<std::vector<double>> benchmark_sample_shipment(int num_samples) {
    std::vector<std::vector<double>> sample_set;
    for (int index = 0; index < num_samples; ++index) {
        std::vector<double> cur_sample = sample_loader_shipment();
        sample_set.push_back(cur_sample);
    }
    return sample_set;
}
