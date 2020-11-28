//
//  sample_generator.cpp
//  Two_Stage_SP_Sample_Generator
//
//  Created by Shuotao Diao on 10/30/20.
//

#include "sample_generator.hpp"

// output the values of second stage problem
double baa99_2ndStageValue(const std::vector<double>& x, const std::vector<double>& d, bool flag_verbose){
    if (x.size() != 2) {
        std::cout << "Error! Dimension of vector x is not 2!\n";
        return -1;
    }
    if (d.size() != 2) {
        std::cout << "Error! Dimension of vector d is not 2!\n";
        return -1;
    }
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
    IloExpr con_d1 = w11 + u1;
    mod.add(con_d1 == d[0]);
    IloExpr con_d2 = w12 + u2;
    mod.add(con_d2 == d[1]);
    IloExpr con_s1 = -x[0] + w11 + w12 + v1;
    mod.add(con_s1 == 0);
    IloExpr con_s2 = -x[1] + w22 + v2;
    mod.add(con_s2 == 0);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    if (flag_verbose == false) {
        cplex.setOut(env.getNullStream());
    }
    cplex.solve();
    // get the optimal value
    double opt_value = cplex.getObjValue();
    if (flag_verbose == true) {
        std::cout << "w11: " << cplex.getValue(w11) << std::endl;
        std::cout << "w12: " << cplex.getValue(w12) << std::endl;
        std::cout << "w22: " << cplex.getValue(w22) << std::endl;
        std::cout << "v1: " << cplex.getValue(v1) << std::endl;
        std::cout << "v2: " << cplex.getValue(v2) << std::endl;
        std::cout << "u1: " << cplex.getValue(u1) << std::endl;
        std::cout << "u2: " << cplex.getValue(u2) << std::endl;
    }
    env.end();
    return opt_value;
}

// landS
h_function landS_2ndStageValue(const std::vector<double>& x, const std::vector<double>& omega, bool flag_verbose) {
    if (x.size() != 4) {
        std::cout << "Error! Dimension of vector x is not 4!\n";
        h_function cur_h;
        cur_h.flag_solvable = IloFalse;
        return cur_h;
    }
    if (omega.size() != 1) {
        std::cout << "Error! Dimension of vector omega is not 1!\n";
        h_function cur_h;
        cur_h.flag_solvable = IloFalse;
        return cur_h;
    }
    // initialize CPLEX model environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray y1(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y2(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y3(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y4(env, 3, 0, IloInfinity, ILOFLOAT);
    mod.add(y1);
    mod.add(y2);
    mod.add(y3);
    mod.add(y4);
    // objective expression
    IloExpr expr_obj(env);
    // constraint expression
    // S2C1
    IloExpr expr_S2C1(env);
    // S2C2
    IloExpr expr_S2C2(env);
    // S2C3
    IloExpr expr_S2C3(env);
    // S2C4
    IloExpr expr_S2C4(env);
    // S2C5
    IloExpr expr_S2C5(env);
    // S2C6
    IloExpr expr_S2C6(env);
    // S2C7
    IloExpr expr_S2C7(env);
    // x1
    expr_S2C1 += -x[0];
    // x2
    expr_S2C2 += -x[1];
    // x3
    expr_S2C3 += -x[2];
    // x4
    expr_S2C4 += -x[3];
    // y11
    expr_obj += 40.0 * y1[0];
    expr_S2C1 += y1[0];
    expr_S2C5 += y1[0];
    // y21
    expr_obj += 45.0 * y2[0];
    expr_S2C2 += y2[0];
    expr_S2C5 += y2[0];
    // y31
    expr_obj += 32.0 * y3[0];
    expr_S2C3 += y3[0];
    expr_S2C5 += y3[0];
    // y41
    expr_obj += 55.0 * y4[0];
    expr_S2C4 += y4[0];
    expr_S2C5 += y4[0];
    // y12
    expr_obj += 24.0 * y1[1];
    expr_S2C1 += y1[1];
    expr_S2C6 += y1[1];
    // y22
    expr_obj += 27.0 * y2[1];
    expr_S2C2 += y2[1];
    expr_S2C6 += y2[1];
    // y32
    expr_obj += 19.2 * y3[1];
    expr_S2C3 += y3[1];
    expr_S2C6 += y3[1];
    // y42
    expr_obj += 33.0 * y4[1];
    expr_S2C4 += y4[1];
    expr_S2C6 += y4[1];
    // y13
    expr_obj += 4.0 * y1[2];
    expr_S2C1 += y1[2];
    expr_S2C7 += y1[2];
    // y23
    expr_obj += 4.5 * y2[2];
    expr_S2C2 += y2[2];
    expr_S2C7 += y2[2];
    // y33
    expr_obj += 3.2 * y3[2];
    expr_S2C3 += y3[2];
    expr_S2C7 += y3[2];
    // y43
    expr_obj += 5.5 * y4[2];
    expr_S2C4 += y4[2];
    expr_S2C7 += y4[2];
    // objective function
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    IloRangeArray constraints(env);
    constraints.add(expr_S2C1 <= 0.0);
    constraints.add(expr_S2C2 <= 0.0);
    constraints.add(expr_S2C3 <= 0.0);
    constraints.add(expr_S2C4 <= 0.0);
    constraints.add(expr_S2C5 >= omega[0]);
    constraints.add(expr_S2C6 >= 3.0);
    constraints.add(expr_S2C7 >= 2.0);
    mod.add(constraints);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    if (flag_verbose == false) {
        cplex.setOut(env.getNullStream());
    }
    IloBool solvable_flag = cplex.solve();
    // get the optimal value
    h_function cur_h;
    if (solvable_flag == IloTrue) {
        cur_h.value = cplex.getObjValue();
        cur_h.flag_solvable = solvable_flag;
        if (flag_verbose == true) {
            for (int index = 0; index < 3; ++index) {
                std::cout << "y1" << index+1 << ": " << cplex.getValue(y1[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y2" << index+1 << ": " << cplex.getValue(y2[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y3" << index+1 << ": " << cplex.getValue(y3[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y4" << index+1 << ": " << cplex.getValue(y4[index]) << std::endl;
            }
        }
    }
    else {
        cur_h.flag_solvable = solvable_flag;
        std::cout << "Warning: Second stage is infeasible!\n";
    }
    env.end();
    return cur_h;
}

// landS2
h_function landS2_2ndStageValue(const std::vector<double>& x, const std::vector<double>& omega, bool flag_verbose) {
    if (x.size() != 4) {
        std::cout << "Error! Dimension of vector x is not 4!\n";
        h_function cur_h;
        cur_h.flag_solvable = IloFalse;
        return cur_h;
    }
    if (omega.size() != 3) {
        std::cout << "Error! Dimension of vector omega is not 3!\n";
        h_function cur_h;
        cur_h.flag_solvable = IloFalse;
        return cur_h;
    }
    // initialize CPLEX model environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray y1(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y2(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y3(env, 3, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray y4(env, 3, 0, IloInfinity, ILOFLOAT);
    mod.add(y1);
    mod.add(y2);
    mod.add(y3);
    mod.add(y4);
    // objective expression
    IloExpr expr_obj(env);
    // constraint expression
    // S2C1
    IloExpr expr_S2C1(env);
    // S2C2
    IloExpr expr_S2C2(env);
    // S2C3
    IloExpr expr_S2C3(env);
    // S2C4
    IloExpr expr_S2C4(env);
    // S2C5
    IloExpr expr_S2C5(env);
    // S2C6
    IloExpr expr_S2C6(env);
    // S2C7
    IloExpr expr_S2C7(env);
    // x1
    expr_S2C1 += -x[0];
    // x2
    expr_S2C2 += -x[1];
    // x3
    expr_S2C3 += -x[2];
    // x4
    expr_S2C4 += -x[3];
    // y11
    expr_obj += 40.0 * y1[0];
    expr_S2C1 += y1[0];
    expr_S2C5 += y1[0];
    // y21
    expr_obj += 45.0 * y2[0];
    expr_S2C2 += y2[0];
    expr_S2C5 += y2[0];
    // y31
    expr_obj += 32.0 * y3[0];
    expr_S2C3 += y3[0];
    expr_S2C5 += y3[0];
    // y41
    expr_obj += 55.0 * y4[0];
    expr_S2C4 += y4[0];
    expr_S2C5 += y4[0];
    // y12
    expr_obj += 24.0 * y1[1];
    expr_S2C1 += y1[1];
    expr_S2C6 += y1[1];
    // y22
    expr_obj += 27.0 * y2[1];
    expr_S2C2 += y2[1];
    expr_S2C6 += y2[1];
    // y32
    expr_obj += 19.2 * y3[1];
    expr_S2C3 += y3[1];
    expr_S2C6 += y3[1];
    // y42
    expr_obj += 33.0 * y4[1];
    expr_S2C4 += y4[1];
    expr_S2C6 += y4[1];
    // y13
    expr_obj += 4.0 * y1[2];
    expr_S2C1 += y1[2];
    expr_S2C7 += y1[2];
    // y23
    expr_obj += 4.5 * y2[2];
    expr_S2C2 += y2[2];
    expr_S2C7 += y2[2];
    // y33
    expr_obj += 3.2 * y3[2];
    expr_S2C3 += y3[2];
    expr_S2C7 += y3[2];
    // y43
    expr_obj += 5.5 * y4[2];
    expr_S2C4 += y4[2];
    expr_S2C7 += y4[2];
    // objective function
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    IloRangeArray constraints(env);
    constraints.add(expr_S2C1 <= 0.0);
    constraints.add(expr_S2C2 <= 0.0);
    constraints.add(expr_S2C3 <= 0.0);
    constraints.add(expr_S2C4 <= 0.0);
    constraints.add(expr_S2C5 >= omega[0]);
    constraints.add(expr_S2C6 >= omega[1]);
    constraints.add(expr_S2C7 >= omega[2]);
    mod.add(constraints);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    if (flag_verbose == false) {
        cplex.setOut(env.getNullStream());
    }
    IloBool solvable_flag = cplex.solve();
    // get the optimal value
    h_function cur_h;
    if (solvable_flag == IloTrue) {
        cur_h.value = cplex.getObjValue();
        cur_h.flag_solvable = solvable_flag;
        if (flag_verbose == true) {
            for (int index = 0; index < 3; ++index) {
                std::cout << "y1" << index+1 << ": " << cplex.getValue(y1[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y2" << index+1 << ": " << cplex.getValue(y2[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y3" << index+1 << ": " << cplex.getValue(y3[index]) << std::endl;
            }
            for (int index = 0; index < 3; ++index) {
                std::cout << "y4" << index+1 << ": " << cplex.getValue(y4[index]) << std::endl;
            }
        }
    }
    else {
        cur_h.flag_solvable = solvable_flag;
        std::cout << "Warning: Second stage is infeasible!\n";
    }
    env.end();
    return cur_h;
}


// CEP
h_function cep_2ndStageValue(const std::vector<double>& xM, const std::vector<double>& zM, const std::vector<double>& omega, bool flag_verbose) {
    h_function cur_h;
    if (xM.size() != 4) {
        cur_h.flag_solvable = IloFalse;
        std::cout << "Warning: Dimension of vector xM is not 4!\n";
    }
    if (zM.size() != 4) {
        cur_h.flag_solvable = IloFalse;
        std::cout << "Warning: Dimension of vector zM is not 4!\n";
    }
    if (omega.size() != 3) {
        cur_h.flag_solvable = IloFalse;
        std::cout << "Warning: Dimension of vector omgea is not 3!\n";
    }
    // set up cplex environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray yP1M(env, 4, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray yP2M(env, 4, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray yP3M(env, 4, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray sP(env, 3, 0, IloInfinity, ILOFLOAT);
    mod.add(yP1M);
    mod.add(yP2M);
    mod.add(yP3M);
    mod.add(sP);
    // expressions
    IloExpr expr_obj(env);
    IloExpr expr_CAPM1(env);
    IloExpr expr_CAPM2(env);
    IloExpr expr_CAPM3(env);
    IloExpr expr_CAPM4(env);
    IloExpr expr_DEMP1(env);
    IloExpr expr_DEMP2(env);
    IloExpr expr_DEMP3(env);
    //
    expr_CAPM1 += zM[0];
    
    expr_CAPM2 += zM[1];
    
    expr_CAPM3 += zM[2];
    
    expr_CAPM4 += zM[3];
    
    expr_obj += 2.6 * yP1M[0];
    expr_CAPM1 += -yP1M[0];
    expr_DEMP1 += 0.6 * yP1M[0];
    
    expr_obj += 3.4 * yP1M[1];
    expr_CAPM2 += -yP1M[1];
    expr_DEMP1 += 0.6 * yP1M[1];
    
    expr_obj += 3.4 * yP1M[2];
    expr_CAPM3 += -yP1M[2];
    expr_DEMP1 += 0.9 * yP1M[2];
    
    expr_obj += 2.5 * yP1M[3];
    expr_CAPM4 += -yP1M[3];
    expr_DEMP1 += 0.8 * yP1M[3];
    
    expr_obj += 1.5 * yP2M[0];
    expr_CAPM1 += -yP2M[0];
    expr_DEMP2 += 0.1 * yP2M[0];
    
    expr_obj += 2.4 * yP2M[1];
    expr_CAPM2 += -yP2M[1];
    expr_DEMP2 += 0.9 * yP2M[1];
    
    expr_obj += 2 * yP2M[2];
    expr_CAPM3 += -yP2M[2];
    expr_DEMP2 += 0.6 * yP2M[2];
    
    expr_obj += 3.6 * yP2M[3];
    expr_CAPM4 += -yP2M[3];
    expr_DEMP2 += 0.8 * yP2M[3];
    
    expr_obj += 4 * yP3M[0];
    expr_CAPM1 += -yP3M[0];
    expr_DEMP3 += 0.05 * yP3M[0];
    
    expr_obj += 3.8 * yP3M[1];
    expr_CAPM2 += -yP3M[1];
    expr_DEMP3 += 0.2 * yP3M[1];
    
    expr_obj += 3.5 * yP3M[2];
    expr_CAPM3 += -yP3M[2];
    expr_DEMP3 += 0.5 * yP3M[2];
    
    expr_obj += 3.2 * yP3M[3];
    expr_CAPM4 += -yP3M[3];
    expr_DEMP3 += 0.8 * yP3M[3];
    
    expr_obj += 400 * sP[0];
    expr_DEMP1 += sP[0];
    
    expr_obj += 400 * sP[1];
    expr_DEMP2 += sP[1];
    
    expr_obj += 400 * sP[2];
    expr_DEMP3 += sP[2];
    
    // objective function
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    IloRangeArray constraintsG(env);
    constraintsG.add(expr_CAPM1 >= 0);
    constraintsG.add(expr_CAPM2 >= 0);
    constraintsG.add(expr_CAPM3 >= 0);
    constraintsG.add(expr_CAPM4 >= 0);
    constraintsG.add(expr_DEMP1 >= omega[0]);
    constraintsG.add(expr_DEMP2 >= omega[1]);
    constraintsG.add(expr_DEMP3 >= omega[2]);
    mod.add(constraintsG);
    // set up CPLEX solver
    IloCplex cplex(env);
    cplex.extract(mod);
    if (flag_verbose == false) {
        cplex.setOut(env.getNullStream());
    }
    cur_h.flag_solvable = cplex.solve();
    // obtain optimal value
    cur_h.value = cplex.getObjValue();
    if (flag_verbose == true) {
        for (int index = 0; index < 4; ++index) {
            std::cout << "yP1M" << index + 1 << " = " << cplex.getValue(yP1M[index]) << std::endl;
        }
        for (int index = 0; index < 4; ++index) {
            std::cout << "yP2M" << index + 1 << " = " << cplex.getValue(yP2M[index]) << std::endl;
        }
        for (int index = 0; index < 4; ++index) {
            std::cout << "yP3M" << index + 1 << " = " << cplex.getValue(yP3M[index]) << std::endl;
        }
        for (int index = 0; index < 3; ++index) {
            std::cout << "sP" << index + 1 << " = " << cplex.getValue(sP[index]) << std::endl;
        }
    }
    env.end();
    return cur_h;
}

// shipment planning
h_function shipment_2ndStageValue(const std::vector<double>& x, const std::vector<double>& xi) {
    // parameters
    int dx = 4;
    int dxi = 12;
    double p2 = 100;
    std::vector<std::vector<double>> c;
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        std::vector<double> c_row(dxi,0);
        c.push_back(c_row);
    }
    // shipping cost
    c[0][0] = 1.5000;
    c[0][1] = 5.0026;
    c[0][2] = 9.3408;
    c[0][3] = 13.1240;
    c[0][4] = 16.0390;
    c[0][5] = 17.8740;
    c[0][6] = 18.5000;
    c[0][7] = 17.8740;
    c[0][8] = 16.0390;
    c[0][9] = 13.1240;
    c[0][10] = 9.3408;
    c[0][11] = 5.0026;
    
    c[1][0] = 13.1240;
    c[1][1] = 9.3408;
    c[1][2] = 5.0026;
    c[1][3] = 1.5000;
    c[1][4] = 5.0026;
    c[1][5] = 9.3408;
    c[1][6] = 13.1240;
    c[1][7] = 16.0390;
    c[1][8] = 17.8740;
    c[1][9] = 18.5000;
    c[1][10] = 17.8740;
    c[1][11] = 16.0390;
    
    c[2][0] = 18.5000;
    c[2][1] = 17.8740;
    c[2][2] = 16.0390;
    c[2][3] = 13.1240;
    c[2][4] = 9.3408;
    c[2][5] = 5.0026;
    c[2][6] = 1.5000;
    c[2][7] = 5.0026;
    c[2][8] = 9.3408;
    c[2][9] = 13.1240;
    c[2][10] = 16.0390;
    c[2][11] = 17.8740;
    
    c[3][0] = 13.1240;
    c[3][1] = 16.0390;
    c[3][2] = 17.8740;
    c[3][3] = 18.5000;
    c[3][4] = 17.8740;
    c[3][5] = 16.0390;
    c[3][6] = 13.1240;
    c[3][7] = 9.3408;
    c[3][8] = 5.0026;
    c[3][9] = 1.5000;
    c[3][10] = 5.0026;
    c[3][11] = 9.3408;
    
    std::vector<double> gx2(dx,0);
    // construct cplex environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray y(env,dx,0,IloInfinity,ILOFLOAT); // last minute production
    mod.add(y);
    y.setNames("y");
    std::vector<IloNumVarArray> z; // shipment
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        IloNumVarArray z_temp(env,dxi,0,IloInfinity,ILOFLOAT);
        z.push_back(z_temp);
    }
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        std::string name = "z" + std::to_string(index_warehouse);
        const char* name_eligible = name.c_str();
        z[index_warehouse].setNames(name_eligible);
        mod.add(z[index_warehouse]);
    }
    // objective function
    IloExpr expr1(env);
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        expr1 += p2 * y[index_warehouse];
        for (int index_location = 0; index_location < dxi; index_location++) {
            expr1 += c[index_warehouse][index_location] * z[index_warehouse][index_location];
        }
    }
    IloObjective obj = IloMinimize(env,expr1);
    mod.add(obj);
    // constraints
    for (int index_location = 0; index_location < dxi; index_location++) {
        IloExpr expr2(env);
        for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
            expr2 += -z[index_warehouse][index_location];
        }
        mod.add(expr2 <= -xi[index_location]);
    }
    IloRangeArray constraints2(env);
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        IloExpr expr3(env);
        expr3 = -x[index_warehouse] - y[index_warehouse];
        for (int index_location = 0; index_location < dxi; index_location++) {
            expr3 += z[index_warehouse][index_location];
        }
        constraints2.add(expr3 <= 0);
    }
    mod.add(constraints2);
    // solve
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool flag_solvable = cplex.solve();
//    cplex.exportModel("/Users/sonny/Documents/output/tss_subgradient_second_stage.lp");
    h_function cur_h;
    cur_h.flag_solvable = flag_solvable;
    if (flag_solvable == IloTrue){
        //std::cout << "Subproblem is feasible." << std::endl;
        cur_h.value = cplex.getObjValue();
    }
    else {
        std::cout << "Subproblem is infeasible." << std::endl;
    } // end else
    
    // dual multipliers
    /*
    IloNumArray duals(env);
    cplex.getDuals(duals, constraints2);
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        gx2[index_warehouse] = -duals[index_warehouse];
        std::cout << "gx2[" << index_warehouse << "] = " << gx2[index_warehouse] << std::endl;;
    }
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        std::cout << "y[" << index_warehouse << "] = " << cplex.getValue(y[index_warehouse]) << std::endl;
    }
    for (int index_warehouse = 0; index_warehouse < dx; index_warehouse++) {
        for (int index_location = 0; index_location < dxi; index_location++) {
            std::cout << "z[" << index_warehouse << "][" << index_location << "] = " << cplex.getValue(z[index_warehouse][index_location]) << std::endl;;
        }
        std::cout << "*********************" << std::endl;
    }
     */
    env.end(); // release the memory of cplex
    return cur_h;
}

// generate samples with labels
void baa99_sample_generator(const std::string& outputPath, int num_samples){
    // generate demands
    std::vector<std::vector<double>> d;
    // sample first stage decision variable
    std::vector<std::vector<double>> x;
    // second stage problem function value
    std::vector<double> h;
    //
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
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
        d.push_back(cur_d);
        // generate x vector
        double x1 = ((double) rand()) / RAND_MAX * 217.0;
        double x2 = ((double) rand()) / RAND_MAX * 217.0;
        std::vector<double> cur_x;
        cur_x.push_back(x1);
        cur_x.push_back(x2);
        x.push_back(cur_x);
        // generate second stage problem function value
        double cur_value = baa99_2ndStageValue(cur_x, cur_d, false);
        h.push_back(cur_value);
    }
    // output samples with labels
    // write sample
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    /*
    // first line item category
    writeTrainingSet << "Index,Name,Value\n";
    // samples
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        writeTrainingSet << index_sample << ",x1," << x[index_sample][0] << std::endl;
        writeTrainingSet << index_sample << ",x2," << x[index_sample][1] << std::endl;
        writeTrainingSet << index_sample << ",omega1," << d[index_sample][0] << std::endl;
        writeTrainingSet << index_sample << ",omega2," << d[index_sample][1] << std::endl;
        writeTrainingSet << index_sample << ",h," << h[index_sample] << std::endl;
    }
     */
    writeTrainingSet << "x1,x2,omega1,omega2,h\n";
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        writeTrainingSet << x[index_sample][0] << "," << x[index_sample][1] << ",";
        writeTrainingSet << d[index_sample][0] << "," << d[index_sample][1] << ",";
        writeTrainingSet << h[index_sample] << std::endl;
    }
    //
    writeTrainingSet.close();
    std::cout << "Finish generating samples from baa99.\n";
}

void landS_sample_generator(const std::string& outputPath, int num_samples) {
    // random variables
    std::vector<std::vector<double>> omega;
    // x
    std::vector<std::vector<double>> x;
    // h function values
    std::vector<h_function> h;
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
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
        omega.push_back(cur_omega);
        // generate x
        std::vector<double> cur_x;
        for (int index = 0; index < 4; ++index) {
            double x_tmp = ((double) rand()) / RAND_MAX * 20.0;
            cur_x.push_back(x_tmp);
        }
        // project cur_x on the feasible region
        std::vector<double> x_proj = landS_projection(cur_x);
        x.push_back(x_proj);
        // obtain h value
        h_function cur_h = landS_2ndStageValue(x_proj, cur_omega, false);
        h.push_back(cur_h);
    }
    // output samples with labels
    // write sample
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    /*
    // first line item category
    writeTrainingSet << "Index,Name,Value\n";
    // samples
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << index_sample << ",x" << index + 1 << "," << x[index_sample][index] << std::endl;
        }
        writeTrainingSet << index_sample << ",omega1," << omega[index_sample][0] << std::endl;
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << index_sample << ",h," << h[index_sample].value << std::endl;
        }
        else {
            writeTrainingSet << index_sample << ",h,infeasible" << std::endl;
        }
    }
     */
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "x" << index + 1 << ",";
    }
    writeTrainingSet << "omega1,h\n";
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << x[index_sample][index] << ",";
        }
        writeTrainingSet << omega[index_sample][0] << ",";
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << h[index_sample].value << std::endl;
        }
        else {
            writeTrainingSet << "infeasible\n";
        }
    }
    //
    writeTrainingSet.close();
    std::cout << "Finish generating samples from landS.\n";
}

void landS2_sample_generator(const std::string& outputPath, int num_samples) {
    // random variables
    std::vector<std::vector<double>> omega;
    // x
    std::vector<std::vector<double>> x;
    // h function values
    std::vector<h_function> h;
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
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
        omega.push_back(cur_omega);
        // generate x
        std::vector<double> cur_x;
        for (int index = 0; index < 4; ++index) {
            double x_tmp = ((double) rand()) / RAND_MAX * 20.0;
            cur_x.push_back(x_tmp);
        }
        // project cur_x on the feasible region
        std::vector<double> x_proj = landS_projection(cur_x);
        x.push_back(x_proj);
        // obtain h value
        h_function cur_h = landS2_2ndStageValue(x_proj, cur_omega, false);
        h.push_back(cur_h);
    }
    // output samples with labels
    // write sample
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "x" << index + 1 << ",";
    }
    for (int index = 0; index < 3; ++index) {
        writeTrainingSet << "omega" << index + 1 << ",";
    }
    writeTrainingSet << "h\n";
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << x[index_sample][index] << ",";
        }
        for (int index = 0; index < 3; ++index) {
            writeTrainingSet << omega[index_sample][index] << ",";
        }
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << h[index_sample].value << std::endl;
        }
        else {
            writeTrainingSet << "infeasible\n";
        }
    }
    //
    writeTrainingSet.close();
    std::cout << "Finish generating samples from landS2.\n";
}

// landS2 and landS3 share the same second stage function
// the only difference is the distribution of the random variables
void landS3_sample_generator(const std::string& outputPath, int num_samples) {
    // random variables
    std::vector<std::vector<double>> omega;
    // x
    std::vector<std::vector<double>> x;
    // h function values
    std::vector<h_function> h;
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
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
        omega.push_back(cur_omega);
        // generate x
        std::vector<double> cur_x;
        for (int index = 0; index < 4; ++index) {
            double x_tmp = ((double) rand()) / RAND_MAX * 20.0;
            cur_x.push_back(x_tmp);
        }
        // project cur_x on the feasible region
        std::vector<double> x_proj = landS_projection(cur_x);
        x.push_back(x_proj);
        // obtain h value
        h_function cur_h = landS2_2ndStageValue(x_proj, cur_omega, false);
        h.push_back(cur_h);
    }
    // output samples with labels
    // write sample
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "x" << index + 1 << ",";
    }
    for (int index = 0; index < 3; ++index) {
        writeTrainingSet << "omega" << index + 1 << ",";
    }
    writeTrainingSet << "h\n";
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << x[index_sample][index] << ",";
        }
        for (int index = 0; index < 3; ++index) {
            writeTrainingSet << omega[index_sample][index] << ",";
        }
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << h[index_sample].value << std::endl;
        }
        else {
            writeTrainingSet << "infeasible\n";
        }
    }
    //
    writeTrainingSet.close();
    std::cout << "Finish generating samples from landS3.\n";
}

void cep_sample_generator(const std::string& outputPath, int num_samples) {
    std::vector<std::vector<double>> xM; // 0 - 3000
    std::vector<std::vector<double>> zM; // 0 - 3000
    std::vector<std::vector<double>> omega;
    // h function values
    std::vector<h_function> h;
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        // generate omega
        double omega1;
        double omega2;
        double omega3;
        double p_omega1 = ((double) rand()) / RAND_MAX;
        double p_omega2 = ((double) rand()) / RAND_MAX;
        double p_omega3 = ((double) rand()) / RAND_MAX;
        if (p_omega1 < 0.166667) {
            omega1 = 0;
        }
        else if (p_omega1 < 0.166667 * 2.0) {
            omega1 = 600;
        }
        else if (p_omega1 < 0.166667 * 3.0) {
            omega1 = 1200;
        }
        else if (p_omega1 < 0.166667 * 4.0) {
            omega1 = 1800;
        }
        else if (p_omega1 < 0.166667 * 5.0) {
            omega1 = 2400;
        }
        else {
            omega1 = 3000;
        }
        
        if (p_omega2 < 0.166667) {
            omega2 = 0;
        }
        else if (p_omega2 < 0.166667 * 2.0) {
            omega2 = 600;
        }
        else if (p_omega2 < 0.166667 * 3.0) {
            omega2 = 1200;
        }
        else if (p_omega2 < 0.166667 * 4.0) {
            omega2 = 1800;
        }
        else if (p_omega2 < 0.166667 * 5.0) {
            omega2 = 2400;
        }
        else {
            omega2 = 3000;
        }
        
        if (p_omega3 < 0.166667) {
            omega3 = 0;
        }
        else if (p_omega3 < 0.166667 * 2.0) {
            omega3 = 600;
        }
        else if (p_omega3 < 0.166667 * 3.0) {
            omega3 = 1200;
        }
        else if (p_omega3 < 0.166667 * 4.0) {
            omega3 = 1800;
        }
        else if (p_omega3 < 0.166667 * 5.0) {
            omega3 = 2400;
        }
        else {
            omega3 = 3000;
        }
        std::vector<double> cur_omega;
        cur_omega.push_back(omega1);
        cur_omega.push_back(omega2);
        cur_omega.push_back(omega3);
        omega.push_back(cur_omega);
        // generate xM and zM
        std::vector<double> xM_cur;
        std::vector<double> zM_cur;
        for (int index = 0; index < 4; ++index) {
            xM_cur.push_back(((double) rand()) / RAND_MAX * 3000);
            zM_cur.push_back(((double) rand()) / RAND_MAX * 3000);
        }
        std::vector<double> x_proj = cep_projection(xM_cur, zM_cur);
        std::vector<double> xM_proj;
        std::vector<double> zM_proj;
        for (int index = 0; index < 4; ++index) {
            xM_proj.push_back(x_proj[index]);
        }
        for (int index = 0; index < 4; ++index) {
            zM_proj.push_back(x_proj[4 + index]);
        }
        xM.push_back(xM_proj);
        zM.push_back(zM_proj);
        // get the h value
        h_function cur_h = cep_2ndStageValue(xM_proj, zM_proj, cur_omega, false);
        h.push_back(cur_h);
    }
    
    // output the sample set
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    // first line
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "xM" << index + 1 << ",";
    }
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "zM" << index + 1 << ",";
    }
    for (int index = 0; index < 3; ++index) {
        writeTrainingSet << "omega" << index + 1 << ",";
    }
    writeTrainingSet << "h\n";
    // rest lines
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << xM[index_sample][index] << ",";
        }
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << zM[index_sample][index] << ",";
        }
        for (int index = 0; index < 3; ++index) {
            writeTrainingSet << omega[index_sample][index] << ",";
        }
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << h[index_sample].value << std::endl;
        }
        else {
            std::cout << "Warning: Second stage problem in infeasible!\n";
            writeTrainingSet << "infeasible" << std::endl;
        }
    }
    writeTrainingSet.close();
    std::cout << "Finish generating training set\n";
}


void shipment_sample_generator(const std::string& outputPath, int num_samples) {
    // random variables
    std::vector<std::vector<double>> omega;
    // x
    std::vector<std::vector<double>> x;
    // h function values
    std::vector<h_function> h;
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
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
        //
        std::vector<double> cur_x;
        for (int index = 0; index < 4; ++index) {
            double x_instance = ((double) rand()) / RAND_MAX * 60;
            cur_x.push_back(x_instance);
        }
        omega.push_back(cur_omega);
        x.push_back(cur_x);
        // get the current h value
        h_function cur_h = shipment_2ndStageValue(cur_x, cur_omega);
        h.push_back(cur_h);
    }
    
    // output samples with labels
    // write sample
    std::fstream writeTrainingSet;
    writeTrainingSet.open(outputPath, std::fstream::app);
    for (int index = 0; index < 4; ++index) {
        writeTrainingSet << "x" << index + 1 << ",";
    }
    for (int index = 0; index < 12; ++index) {
        writeTrainingSet << "omega" << index + 1 << ",";
    }
    writeTrainingSet << "h\n";
    for (int index_sample = 0; index_sample < num_samples; ++index_sample) {
        for (int index = 0; index < 4; ++index) {
            writeTrainingSet << x[index_sample][index] << ",";
        }
        for (int index = 0; index < 12; ++index) {
            writeTrainingSet << omega[index_sample][index] << ",";
        }
        if (h[index_sample].flag_solvable == IloTrue) {
            writeTrainingSet << h[index_sample].value << std::endl;
        }
        else {
            writeTrainingSet << "infeasible\n";
        }
    }
    //
    writeTrainingSet.close();
    std::cout << "Finish generating samples from shipment.\n";
    
}


// projection
std::vector<double> landS_projection(const std::vector<double>& x) {
    // set up cplex environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray x_proj(env,4,0,IloInfinity,ILOFLOAT);
    mod.add(x_proj);
    // objective function
    IloExpr expr_obj(env);
    for (int index = 0; index < 4; ++index) {
        expr_obj += 0.5 * (x_proj[index] - x[index]) * (x_proj[index] - x[index]);
    }
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    IloExpr expr_con1(env);
    expr_con1 += x_proj[0] + x_proj[1] + x_proj[2] + x_proj[3];
    mod.add(expr_con1 >= 12);
    IloExpr expr_con2(env);
    expr_con2 += 10 * x_proj[0] + 7 * x_proj[1] + 16 * x_proj[2] + 6 * x_proj[3];
    mod.add(expr_con2 <= 120);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    cplex.solve();
    std::vector<double> x_proj2;
    for (int index = 0; index < 4; ++index) {
        x_proj2.push_back(cplex.getValue(x_proj[index]));
    }
    env.end();
    return x_proj2;
}

std::vector<double> cep_projection(const std::vector<double>& xM, const std::vector<double>& zM) {
    // set up cplex environment
    IloEnv env;
    IloModel mod(env);
    // decision variables
    IloNumVarArray x(env,4,0,IloInfinity,ILOFLOAT);
    IloNumVarArray z(env,4,0,IloInfinity,ILOFLOAT);
    //
    IloExpr expr_obj(env);
    for (int index = 0; index < 4; ++index) {
        expr_obj += 0.5 * (x[index] - xM[index]) * (x[index] - xM[index]);
        expr_obj += 0.5 * (z[index] - zM[index]) * (z[index] - zM[index]);
    }
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constraints
    mod.add(z[0] <= 2000);
    mod.add(z[1] <= 2000);
    mod.add(z[2] <= 3000);
    mod.add(z[3] <= 3000);
    for (int index = 0; index < 4; ++index) {
        IloExpr expr_con(env);
        expr_con += -x[index] + z[index];
        mod.add(expr_con <= 500);
    }
    mod.add(0.8 * z[0] + 0.04 * z[1] + 0.03 * z[2] + 0.01 * z[3] <= 100);
    // set up the cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    cplex.solve();
    std::vector<double> x_proj;
    for (int index = 0; index < 4; ++index) {
        x_proj.push_back(cplex.getValue(x[index]));
    }
    for (int index = 0; index < 4; ++index) {
        x_proj.push_back(cplex.getValue(z[index]));
    }
    return x_proj;
}

// test functions
void test_baa99(){
    std::vector<double> x;
    x.push_back(120.0);
    x.push_back(30.0);
    std::vector<double> d;
    d.push_back(103.7634931);
    d.push_back(26.21044859);
    double value = baa99_2ndStageValue(x, d, true);
    std::cout << "functin value: " << value << std::endl;
}

void test_landS(){
    std::vector<double> x;
    x.push_back(12);
    x.push_back(0);
    x.push_back(0);
    x.push_back(0);
    std::vector<double> omega;
    omega.push_back(3.0);
    h_function cur_h = landS_2ndStageValue(x, omega, true);
    if (cur_h.flag_solvable == IloTrue) {
        std::cout << "Function value: " << cur_h.value << std::endl;
    }
}


void test_cep(){
    std::vector<double> xM;
    std::vector<double> zM;
    std::vector<double> omega;
    xM.push_back(10);
    xM.push_back(10);
    xM.push_back(10);
    xM.push_back(10);
    zM.push_back(20);
    zM.push_back(20);
    zM.push_back(20);
    zM.push_back(20);
    omega.push_back(3000);
    omega.push_back(3000);
    omega.push_back(3000);
    h_function cur_h = cep_2ndStageValue(xM, zM, omega, true);
    if (cur_h.flag_solvable == IloTrue) {
        std::cout << "Function value: " << cur_h.value << std::endl;
    }
}
