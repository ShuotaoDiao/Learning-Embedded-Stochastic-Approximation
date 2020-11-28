# Stochastic Approximation for landS2 and landS3
# Shuotao Diao, sdiao@usc.edu

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import random
import time

# import the Model class from docplex.mp
from docplex.mp.model import Model


# landS2_3
def subgradient_landS2_3(x, omega):
    m = Model(name='landS_second_stage')
    y1 = m.continuous_var_list(3, name='y1')
    y2 = m.continuous_var_list(3, name='y2')
    y3 = m.continuous_var_list(3, name='y3')
    y4 = m.continuous_var_list(3, name='y4')
    # objective
    m.minimize(40.0 * y1[0] + 45.0 * y2[0] + 32.0 * y3[0] + 55.0 * y4[0] + 24.0 * y1[1] + 27.0 * y2[1]
               + 19.2 * y3[1] + 33.0 * y4[1] + 4.0 * y1[2] + 4.5 * y2[2] + 3.2 * y3[2] + 5.5 * y4[2])
    # constraints
    con_s2c1 = m.add_constraint(-x[0] + y1[0] + y1[1] + y1[2] <= 0)
    con_s2c2 = m.add_constraint(-x[1] + y2[0] + y2[1] + y2[2] <= 0)
    con_s2c3 = m.add_constraint(-x[2] + y3[0] + y3[1] + y3[2] <= 0)
    con_s2c4 = m.add_constraint(-x[3] + y4[0] + y4[1] + y4[2] <= 0)

    con_s2c5 = m.add_constraint(y1[0] + y2[0] + y3[0] + y4[0] >= omega[0])
    con_s2c6 = m.add_constraint(y1[1] + y2[1] + y3[1] + y4[1] >= omega[1])
    con_s2s7 = m.add_constraint(y1[2] + y2[2] + y3[2] + y4[2] >= omega[2])
    # solve the problem
    s = m.solve()
    #
    obj = s.get_objective_value()
    # dual
    dual_con_s2c1 = - con_s2c1.dual_value
    dual_con_s2c2 = - con_s2c2.dual_value
    dual_con_s2c3 = - con_s2c3.dual_value
    dual_con_s2c4 = - con_s2c4.dual_value
    # subgradient
    subgradient = np.zeros(4)
    subgradient[0] = -dual_con_s2c1
    subgradient[1] = -dual_con_s2c2
    subgradient[2] = -dual_con_s2c3
    subgradient[3] = -dual_con_s2c4
    # optimal value
    obj = s.get_objective_value()
    # end the model
    m.end()
    return subgradient, obj


# sample generator
def sample_generator_landS2():
    p_omega1 = random.random()
    omega = []
    if p_omega1 < 0.25:
        omega.append(0.0)
    elif p_omega1 < 0.5:
        omega.append(0.96)
    elif p_omega1 < 0.75:
        omega.append(2.96)
    else:
        omega.append(3.96)

    # omega2
    p_omega2 = random.random()
    if p_omega2 < 0.25:
        omega.append(0.0)
    elif p_omega2 < 0.5:
        omega.append(0.96)
    elif p_omega2 < 0.75:
        omega.append(2.96)
    else:
        omega.append(3.96)

    # omega3
    p_omega3 = random.random()
    if p_omega3 < 0.25:
        omega.append(0.0)
    elif p_omega3 < 0.5:
        omega.append(0.96)
    elif p_omega3 < 0.75:
        omega.append(2.96)
    else:
        omega.append(3.96)

    return omega


# sample generator
def sample_generator_landS3():
    p_omega1 = random.random()
    omega = []
    for index in range(1, 101, 1):
        if p_omega1 < 0.01:
            omega.append(0.0)
            break
        else:
            if p_omega1 >= 0.01 * (index - 1) and p_omega1 < 0.01 * index:
                omega.append(0.04 * (index - 1))
                break
            elif index == 100:
                omega.append(3.96)
                break
    # omega2
    p_omega2 = random.random()
    for index in range(1, 101, 1):
        if p_omega2 < 0.01:
            omega.append(0.0)
            break
        else:
            if p_omega2 >= 0.01 * (index - 1) and p_omega2 < 0.01 * index:
                omega.append(0.04 * (index - 1))
                break
            elif index == 100:
                omega.append(3.96)
                break
    # omega3
    p_omega3 = random.random()
    for index in range(1, 101, 1):
        if p_omega3 < 0.01:
            omega.append(0.0)
            break
        else:
            if p_omega3 >= 0.01 * (index - 1) and p_omega3 < 0.01 * index:
                omega.append(0.04 * (index - 1))
                break
            elif index == 100:
                omega.append(3.96)
                break
    return omega


# benchmark sample set
def benchmark_sample_landS2(num_samples):
    sample_set = []
    for index in range(num_samples):
        cur_sample = sample_generator_landS2()
        sample_set.append(cur_sample)
    return sample_set

def benchmark_sample_landS3(num_samples):
    sample_set = []
    for index in range(num_samples):
        cur_sample = sample_generator_landS3()
        sample_set.append(cur_sample)
    return sample_set

# benchmark function
def benchmark_landS2_3(x, sample_set):
    num_samples = len(sample_set)
    second_stage_value = 0
    for index in range(num_samples):
        subgradient, temp_second_stage_value = subgradient_landS2_3(x, sample_set[index])
        second_stage_value += temp_second_stage_value / num_samples
    total_value = second_stage_value + 10.0 * x[0] + 7.0 * x[1] + 16.0 * x[2] + 6.0 * x[3]
    return total_value


# projection
def projection_landS(x):
    m = Model(name='landS_projection')
    x_proj = m.continuous_var_list(4, name='x')
    # objective
    m.minimize(0.5 * (x[0] - x_proj[0]) * (x[0] - x_proj[0]) + 0.5 * (x[1] - x_proj[1]) * (x[1] - x_proj[1])
               + 0.5 * (x[2] - x_proj[2]) * (x[2] - x_proj[2]) + 0.5 * (x[3] - x_proj[3]) * (x[3] - x_proj[3]))
    # constraints
    m.add_constraint(x_proj[0] + x_proj[1] + x_proj[2] + x_proj[3] >= 12)
    m.add_constraint(10 * x_proj[0] + 7 * x_proj[1] + 16 * x_proj[2] + 6 * x_proj[3] <= 120)
    #
    s = m.solve()
    # extract optimal solution (projected point)
    x_proj2 = np.zeros(4)
    for index in range(4):
        x_proj2[index] = s.get_value(x_proj[index])
    # end the model
    m.end()
    return x_proj2


# vanilla sa
def sa_landS2(x_init, stepsize_init=1.0, num_iteration=100, _seed=123):
    print('************************************')
    print('Vanilla SA LandS2')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    for it in range(num_iteration):
        # current step size
        stepsize_cur = stepsize_init / (it + 1.0)
        # draw a sample
        omega = sample_generator_landS2()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record


def sa_landS3(x_init, stepsize_init=1.0, num_iteration=100, _seed=123):
    print('************************************')
    print('Vanilla SA LandS3')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    for it in range(num_iteration):
        # current step size
        stepsize_cur = stepsize_init / (it + 1.0)
        # draw a sample
        omega = sample_generator_landS3()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record

# LESA part
def benchmark_sample2tensor_landS2(num_samples):
    sample_set = torch.zeros((num_samples,3))
    for index in range(num_samples):
        cur_sample = sample_generator_landS2()
        sample_set[index] = torch.tensor(cur_sample, dtype=torch.float)
    return sample_set


def benchmark_sample2tensor_landS3(num_samples):
    sample_set = torch.zeros((num_samples,3))
    for index in range(num_samples):
        cur_sample = sample_generator_landS3()
        sample_set[index] = torch.tensor(cur_sample, dtype=torch.float)
    return sample_set


# estimate the orignial objective function
def benchmark_est_landS2_3(x, dataset, model, n_x=5, n_omega=5):
    sample_size = dataset.size()[0]
    input_dataset = torch.zeros((sample_size, n_x + n_omega), dtype=torch.float)
    input_dataset[:,0] = torch.ones(sample_size, dtype=torch.float) * x[0]
    input_dataset[:,1] = torch.ones(sample_size, dtype=torch.float) * x[1]
    input_dataset[:,2] = torch.ones(sample_size, dtype=torch.float) * x[2]
    input_dataset[:,3] = torch.ones(sample_size, dtype=torch.float) * x[3]
    input_dataset[:,n_x:(n_x+3)] = dataset
    est_h = model.forward(input_dataset)
    est_f = torch.mean(est_h) + 10.0 * x[0] + 7.0 * x[1] + 16.0 * x[2] + 6.0 * x[3]
    return est_f


# directional directive
def directional_directive_landS2_3(x, d, dataset, model, h = 0.001):  # default h = 0.01 (may be too large)
    f1 = benchmark_est_landS2_3(x + h * d, dataset, model)
    f2 = benchmark_est_landS2_3(x, dataset, model)
    directional_directive = (f1 - f2) / h
    return directional_directive


# approximate Armijo rule
def backtrack_line_search_landS2_3(x, d, dataset, model, _alpha = 1.0, rho = 0.8, c=0.001, max_it = 20, debug_mode=True):
    alpha = _alpha
    directional_directive = directional_directive_landS2_3(x, d, dataset, model)
    if debug_mode == True:
        print('directional_directive: {}'.format(directional_directive))
    for it in range(max_it):
        x_proj = projection_landS(x + alpha * d)
        f1 = benchmark_est_landS2_3(x_proj, dataset, model)
        f2 = benchmark_est_landS2_3(x, dataset, model)
        if f1 <= f2 + c * alpha * directional_directive:
            if debug_mode == True:
                print('Debug Mode(backtrack_line_search) Aplha is found!')
                print('alpha: {}'.format(alpha))
                print('Debug Mode(backtrack_line_search) f1: {}'.format(f1))
                print('Debug Mode(backtrack_line_search) f2 + c * alpha * directional_directive: {}'.format(f2 + c * alpha * directional_directive))
            return alpha
        # decrease the condidate stepsize
        alpha = rho * alpha
    print('Debug Mode(backtrack_line_search) Alpha is not found!')
    print('Return the last stepsize in backtrack line search')
    return alpha

# version 2
def backtrack_line_search_v2_landS2_3(x, d, dataset, model, _alpha = 1.0, rho = 0.8, c=0.001, max_it = 20, debug_mode=True):
    alpha = _alpha
    directional_directive = directional_directive_landS2_3(x, d, dataset, model)
    if debug_mode == True:
        print('directional_directive: {}'.format(directional_directive))
    for it in range(max_it):
        x_proj = projection_landS(x + alpha * d)
        f1 = benchmark_est_landS2_3(x_proj, dataset, model)
        f2 = benchmark_est_landS2_3(x, dataset, model)
        if f1 <= f2 + min(0, c * alpha * directional_directive):
            if debug_mode == True:
                print('Debug Mode(backtrack_line_search) Aplha is found!')
                print('alpha: {}'.format(alpha))
                print('Debug Mode(backtrack_line_search) f1: {}'.format(f1))
                print('Debug Mode(backtrack_line_search) f2 + c * alpha * directional_directive: {}'.format(f2 + c * alpha * directional_directive))
            return alpha
        # decrease the condidate stepsize
        alpha = rho * alpha
    print('Debug Mode(backtrack_line_search) Aplha is not found!')
    print('Return the last stepsize in backtrack line search')
    return alpha


# vanilla LESA landS2
def lesa_landS2(x_init, model, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA LandS2')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_landS2(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_landS2()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_landS2_3(x_cur, -subgradient_cur, dataset, model)
        else:
            print('Not using backtrack line search')
            stepsize_cur = stepsize_init / (it + 1.0)
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record

# version 2
def lesa_v2_landS2(x_init, model, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA Version 2 LandS2')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_landS2(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_landS2()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_v2_landS2_3(x_cur, -subgradient_cur, dataset, model)
        else:
            print('Not using backtrack line search')
            stepsize_cur = stepsize_init / (it + 1.0)
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record

# vanilla LESA landS3
def lesa_landS3(x_init, model, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA LandS3')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_landS3(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_landS3()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_landS2_3(x_cur, -subgradient_cur, dataset, model)
        else:
            print('Not using backtrack line search')
            stepsize_cur = stepsize_init / (it + 1.0)
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record

# version 2
def lesa_v2_landS3(x_init, model, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA Version 2 LandS3')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(4)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_landS3(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_landS3()
        #
        subgradient_cur, obj = subgradient_landS2_3(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 10.0
        subgradient_cur[1] = subgradient_cur[1] + 7.0
        subgradient_cur[2] = subgradient_cur[2] + 16.0
        subgradient_cur[3] = subgradient_cur[3] + 6.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_v2_landS2_3(x_cur, -subgradient_cur, dataset, model)
        else:
            print('Not using backtrack line search')
            stepsize_cur = stepsize_init / (it + 1.0)
        # one gradient step
        x_new = x_cur - stepsize_cur * subgradient_cur
        print('Stepsize: {}'.format(stepsize_cur))
        print('Before projection')
        print(x_new)
        # projection
        print('After projection')
        x_cur = projection_landS(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record