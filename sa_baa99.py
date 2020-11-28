# Stochastic Approximation for baa99
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


# baa99 instance
# second stage value
def baa99_2ndStageValue(x, omega):
    m = Model(name='baa99_second_stage')
    # decision variable
    # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
    w11 = m.continuous_var(name='w11')
    w12 = m.continuous_var(name='w12')
    w22 = m.continuous_var(name='w22')
    v1 = m.continuous_var(name='v1')
    v2 = m.continuous_var(name='v2')
    u1 = m.continuous_var(name='u1')
    u2 = m.continuous_var(name='u2')
    # objective
    m.minimize(-8 * w11 - 4 * w12 - 4 * w22 + 0.2 * v1 + 0.2 * v2 + 10 * u1 + 10 * u2)
    # constraints
    m.add_constraint(w11 + u1 == omega[0])
    m.add_constraint(w12 + u2 == omega[1])
    con_s1 = m.add_constraint(-x[0] + w11 + w12 + v1 == 0)
    con_s2 = m.add_constraint(-x[1] + w22 + v2 == 0)
    #
    s = m.solve()
    #
    obj = s.get_objective_value()
    # end the model
    m.end()
    return obj


# subgradient
def subgradient_baa99(x, omega):
    m = Model(name='baa99_second_stage')
    # decision variable
    # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
    w11 = m.continuous_var(name='w11')
    w12 = m.continuous_var(name='w12')
    w22 = m.continuous_var(name='w22')
    v1 = m.continuous_var(name='v1')
    v2 = m.continuous_var(name='v2')
    u1 = m.continuous_var(name='u1')
    u2 = m.continuous_var(name='u2')
    # objective
    m.minimize(-8 * w11 - 4 * w12 - 4 * w22 + 0.2 * v1 + 0.2 * v2 + 10 * u1 + 10 * u2)
    # constraints
    m.add_constraint(w11 + u1 == omega[0])
    m.add_constraint(w12 + u2 == omega[1])
    con_s1 = m.add_constraint(-x[0] + w11 + w12 + v1 == 0)
    con_s2 = m.add_constraint(-x[1] + w22 + v2 == 0)
    #
    s = m.solve()
    #
    dual_con_s1 = -con_s1.dual_value
    dual_con_s2 = -con_s2.dual_value
    #
    subgradient = np.zeros(2)
    subgradient[0] = -dual_con_s1  # x1
    subgradient[1] = -dual_con_s2  # x2
    # end the model
    m.end()
    return subgradient


# sample generator
def sample_generator_baa99():
    # P(d1)
    p_d1 = random.random()
    # d1
    d1 = 0
    if p_d1 < 0.04:
        d1 = 17.75731865
    elif p_d1 < 0.08:
        d1 = 32.96224832
    elif p_d1 < 0.12:
        d1 = 43.68044355
    elif p_d1 < 0.16:
        d1 = 52.29173734
    elif p_d1 < 0.20:
        d1 = 59.67893765
    elif p_d1 < 0.24:
        d1 = 66.27551249
    elif p_d1 < 0.28:
        d1 = 72.33076402
    elif p_d1 < 0.32:
        d1 = 78.00434172
    elif p_d1 < 0.36:
        d1 = 83.40733268
    elif p_d1 < 0.40:
        d1 = 88.62275117
    elif p_d1 < 0.44:
        d1 = 93.71693266
    elif p_d1 < 0.48:
        d1 = 98.74655459
    elif p_d1 < 0.52:
        d1 = 103.7634931
    elif p_d1 < 0.56:
        d1 = 108.8187082
    elif p_d1 < 0.60:
        d1 = 113.9659517
    elif p_d1 < 0.64:
        d1 = 119.2660233
    elif p_d1 < 0.68:
        d1 = 124.7925174
    elif p_d1 < 0.72:
        d1 = 130.6406496
    elif p_d1 < 0.76:
        d1 = 136.9423425
    elif p_d1 < 0.80:
        d1 = 143.8948148
    elif p_d1 < 0.84:
        d1 = 151.8216695
    elif p_d1 < 0.88:
        d1 = 161.326406
    elif p_d1 < 0.92:
        d1 = 173.7895514
    elif p_d1 < 0.96:
        d1 = 194.0396804
    else:
        d1 = 216.3173937
    # P(d2)
    p_d2 = random.random()
    # d1
    d2 = 0
    if p_d2 < 0.04:
        d2 = 17.75731865
    elif p_d2 < 0.08:
        d2 = 32.96224832
    elif p_d2 < 0.12:
        d2 = 43.68044355
    elif p_d2 < 0.16:
        d2 = 52.29173734
    elif p_d2 < 0.20:
        d2 = 59.67893765
    elif p_d2 < 0.24:
        d2 = 66.27551249
    elif p_d2 < 0.28:
        d2 = 72.33076402
    elif p_d2 < 0.32:
        d2 = 78.00434172
    elif p_d2 < 0.36:
        d2 = 83.40733268
    elif p_d2 < 0.40:
        d2 = 88.62275117
    elif p_d2 < 0.44:
        d2 = 93.71693266
    elif p_d2 < 0.48:
        d2 = 98.74655459
    elif p_d2 < 0.52:
        d2 = 103.7634931
    elif p_d2 < 0.56:
        d2 = 108.8187082
    elif p_d2 < 0.60:
        d2 = 113.9659517
    elif p_d2 < 0.64:
        d2 = 119.2660233
    elif p_d2 < 0.68:
        d2 = 124.7925174
    elif p_d2 < 0.72:
        d2 = 130.6406496
    elif p_d2 < 0.76:
        d2 = 136.9423425
    elif p_d2 < 0.80:
        d2 = 143.8948148
    elif p_d2 < 0.84:
        d2 = 151.8216695
    elif p_d2 < 0.88:
        d2 = 161.326406
    elif p_d2 < 0.92:
        d2 = 173.7895514
    elif p_d2 < 0.96:
        d2 = 194.0396804
    else:
        d2 = 216.3173937
    #
    cur_d = np.zeros(2)
    cur_d[0] = d1
    cur_d[1] = d2
    return cur_d


# benchmark sample set
def benchmark_sample_baa99(num_samples):
    sample_set = []
    for index in range(num_samples):
        cur_sample = sample_generator_baa99()
        sample_set.append(cur_sample)
    return sample_set


# benchmark function
def benchmark_baa99(x, sample_set):
    num_samples = len(sample_set)
    second_stage_value = 0
    for index in range(num_samples):
        temp_second_stage_value = baa99_2ndStageValue(x, sample_set[index])
        second_stage_value += temp_second_stage_value / num_samples
    total_value = second_stage_value + 4.0 * x[0] + 2.0 * x[1]
    return total_value


# projection
def projection_baa99(x):
    m = Model(name='baa99_second_stage')
    x1 = m.continuous_var(name='x1')
    x2 = m.continuous_var(name='x2')
    m.minimize(0.5 * (x1 - x[0]) * (x1 - x[0]) + 0.5 * (x2 - x[1]) * (x2 - x[1]))
    m.add_constraint(x1 <= 217)
    m.add_constraint(x2 <= 217)
    # solve the problem
    s = m.solve()
    # get the projected vector
    x_proj = np.zeros(2)
    x_proj[0] = s.get_value(x1)
    x_proj[1] = s.get_value(x2)
    m.end()
    return x_proj


# vanilla stochastic approximation for baa99
def sa_baa99(x_init, stepsize_init=1.0, num_iteration=100, _seed=123):
    print('************************************')
    print('Vanilla SA BAA99')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(2)
    x_record.append(x_cur)
    for it in range(num_iteration):
        # current step size
        stepsize_cur = stepsize_init / (it + 1.0)
        # draw a sample
        omega = sample_generator_baa99()
        #
        subgradient_cur = subgradient_baa99(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 4.0
        subgradient_cur[1] = subgradient_cur[1] + 2.0
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
        x_cur = projection_baa99(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record


def sa_baa99_val(x_init, stepsize_init=1.0, num_iteration=100, num_benchmark_samples=1000, _seed=123):
    print('************************************')
    print('Vanilla SA BAA99')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    benchmark_set = benchmark_sample_baa99(num_benchmark_samples)
    record_benchmark_value = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(2)
    cur_benchmark_value = benchmark_baa99(x_cur, benchmark_set)
    record_benchmark_value.append(cur_benchmark_value)
    for it in range(num_iteration):
        # current step size
        stepsize_cur = stepsize_init / (it + 1.0)
        # draw a sample
        omega = sample_generator_baa99()
        #
        subgradient_cur = subgradient_baa99(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 4.0
        subgradient_cur[1] = subgradient_cur[1] + 2.0
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
        x_cur = projection_baa99(x_new)
        print(x_cur)
        cur_benchmark_value = benchmark_baa99(x_cur, benchmark_set)
        record_benchmark_value.append(cur_benchmark_value)
        print("=======================================")
    # print the curve of benchmark values
    plt.plot(record_benchmark_value)
    plt.xlabel('Iteration number')
    plt.title('BAA99 Vanilla SA')
    plt.show()
    return x_cur


# evaluate the solution quality by the benchmark value
def baa99_eva(x_record, num_benchmark_samples=1000, _seed=256):
    start = time.time()
    benchmark_set = benchmark_sample_baa99(num_benchmark_samples)
    record_benchmark_value = []
    for x in x_record:
        cur_benchmark_value = benchmark_baa99(x, benchmark_set)
        record_benchmark_value.append(cur_benchmark_value)
    plt.plot(record_benchmark_value)
    end = time.time()
    print("Evalution time (s): {}".format(end - start))
    plt.xlabel('Iteration number')
    plt.title('BAA99 Solution Evalution')
    plt.show()


# LESA part

def benchmark_sample2tensor_baa99(num_samples):
    _dim = 2
    sample_set = torch.zeros((num_samples,_dim))
    for index in range(num_samples):
        cur_sample = sample_generator_baa99()
        sample_set[index,:] = torch.tensor(cur_sample, dtype=torch.float)
    return sample_set

# estimate the orignial objective function
def benchmark_est_baa99(x, dataset, nn_baa99_model_load, n_x=5, n_omega=5):
    sample_size = dataset.size()[0]
    input_dataset = torch.zeros((sample_size, n_x + n_omega), dtype=torch.float)
    input_dataset[:,0] = torch.ones(sample_size, dtype=torch.float) * x[0]
    input_dataset[:,1] = torch.ones(sample_size, dtype=torch.float) * x[1]
    input_dataset[:,n_x:(n_x + 2)] = dataset
    est_h = nn_baa99_model_load.forward(input_dataset)
    est_f = torch.mean(est_h) + 4.0 * x[0] + 2.0 * x[1]
    return est_f

# directional directive
def directional_directive_baa99(x, d, dataset, nn_baa99_model_load, h = 0.001):
    f1 = benchmark_est_baa99(x + h * d, dataset, nn_baa99_model_load)
    f2 = benchmark_est_baa99(x, dataset, nn_baa99_model_load)
    directional_directive = (f1 - f2) / h
    return directional_directive


# approximate Armijo rule
def backtrack_line_search_baa99(x, d, dataset, nn_baa99_model_load, _alpha=1.0, rho=0.8, c=0.001, max_it=20, debug_mode=True):
    alpha = _alpha
    directional_directive = directional_directive_baa99(x, d, dataset, nn_baa99_model_load)
    if debug_mode == True:
        print('directional_directive: {}'.format(directional_directive))
    for it in range(max_it):
        x_proj = projection_baa99(x + alpha * d)
        f1 = benchmark_est_baa99(x_proj, dataset, nn_baa99_model_load)
        f2 = benchmark_est_baa99(x, dataset, nn_baa99_model_load)
        if f1 <= f2 + c * alpha * directional_directive:
            if debug_mode == True:
                print('Debug Mode(backtrack_line_search) Aplha is found!')
                print('alpha: {}'.format(alpha))
                print('Debug Mode(backtrack_line_search) f1: {}'.format(f1))
                print('Debug Mode(backtrack_line_search) f2 + c * alpha * directional_directive: {}'.format(f2 + c * alpha * directional_directive))
            return alpha
        alpha = rho * alpha
    print('(backtrack_line_search) Alpha is not found!')
    print('Return the last stepsize in backtrack line search')
    return alpha


# approximate Armijo rule
def backtrack_line_search_v2_baa99(x, d, dataset, nn_baa99_model_load, _alpha=1.0, rho=0.8, c=0.001, max_it=30, debug_mode=True):
    alpha = _alpha
    directional_directive = directional_directive_baa99(x, d, dataset, nn_baa99_model_load)
    if debug_mode == True:
        print('directional_directive: {}'.format(directional_directive))
    for it in range(max_it):
        x_proj = projection_baa99(x + alpha * d)
        f1 = benchmark_est_baa99(x_proj, dataset, nn_baa99_model_load)
        f2 = benchmark_est_baa99(x, dataset, nn_baa99_model_load)
        if f1 <= f2 + min(0, c * alpha * directional_directive):
            if debug_mode == True:
                print('Debug Mode(backtrack_line_search) Aplha is found!')
                print('alpha: {}'.format(alpha))
                print('Debug Mode(backtrack_line_search) f1: {}'.format(f1))
                print('Debug Mode(backtrack_line_search) f2 + c * alpha * directional_directive: {}'.format(f2 + c * alpha * directional_directive))
            return alpha
        alpha = rho * alpha
    print('(backtrack_line_search) Alpha is not found!')
    print('Return the last stepsize in backtrack line search')
    return alpha


# vanilla LESA
def lesa_baa99(x_init, nn_baa99_model_load, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA BAA99')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(2)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_baa99(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_baa99()
        #
        subgradient_cur = subgradient_baa99(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 4.0
        subgradient_cur[1] = subgradient_cur[1] + 2.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_baa99(x_cur, -subgradient_cur, dataset, nn_baa99_model_load)
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
        x_cur = projection_baa99(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record


# version 2
def lesa_v2_baa99(x_init, nn_baa99_model_load, dataset = None, stepsize_init = 1.0, num_iteration = 100, num_benchmark_samples = 1000, _seed = 123, line_search_it = 5):
    print('************************************')
    print('Vanilla LESA Version 2 BAA99')
    print('************************************')
    # set the random seed
    random.seed(_seed)
    # benchmark set
    x_record = []
    # initialize the estimated solution
    x_cur = copy.deepcopy(x_init)
    x_new = np.zeros(2)
    x_record.append(x_cur)
    # check if benchmark dataset is None
    if dataset is None:
        dataset = benchmark_sample2tensor_baa99(num_benchmark_samples)
    for it in range(num_iteration):
        # draw a sample
        omega = sample_generator_baa99()
        #
        subgradient_cur = subgradient_baa99(x_cur, omega)
        subgradient_cur[0] = subgradient_cur[0] + 4.0
        subgradient_cur[1] = subgradient_cur[1] + 2.0
        # choose current step size
        print("=======================================")
        print('it: {}'.format(it + 1))
        print('Omega:')
        print(omega)
        print('subgradient:')
        print(subgradient_cur)
        if it < line_search_it:
            print('Using backtrack line search')
            stepsize_cur = backtrack_line_search_v2_baa99(x_cur, -subgradient_cur, dataset, nn_baa99_model_load)
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
        x_cur = projection_baa99(x_new)
        print(x_cur)
        x_record.append(x_cur)
        print("=======================================")
    return x_cur, x_record