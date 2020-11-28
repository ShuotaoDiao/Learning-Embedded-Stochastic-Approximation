# Utility functions for LESA
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

import torch.nn.functional as F


def normal_train(args, model, dataloader, validation_set):
    # set the model to be train mode
    model.train()
    # set the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    #
    train_it = 0
    loss_record = []
    validation_loss_record = []
    flag_stop = False
    for ep in range(args['epoch']):
        print("Run Epoch {}".format(ep))
        for data in dataloader:
            # input_value = torch.tensor(data[:,:-1], dtype=torch.float)
            input_value = data[:, :-1].clone().detach().to(dtype=torch.float)
            # h_value = torch.tensor(data[:,-1], dtype=torch.float)
            h_value = data[:, -1].clone().detach().to(dtype=torch.float)
            # zero out gradients
            opt.zero_grad()
            # forward recursion
            est_h_value = model(input_value)
            # loss
            loss = F.mse_loss(est_h_value.view(-1, 1), h_value.view(-1, 1))
            # backward recursion
            loss.backward()
            # update the weights
            opt.step()
            # calculate validation loss
            loss_record.append(loss.detach())
            with torch.no_grad():
                input_value_val = torch.tensor(validation_set[:, :-1], dtype=torch.float)
                h_value_val = torch.tensor(validation_set[:, -1], dtype=torch.float)
                est_h_value_val = model(input_value_val)
                val_loss = F.mse_loss(est_h_value_val.view(-1, 1), h_value_val.view(-1, 1))
                validation_loss_record.append(val_loss)
            if train_it % 50 == 0:
                print("It {}, L2 training loss: {} ".format(train_it, loss.item()))
                print("It {}, L2 validation loss: {} ".format(train_it, val_loss.item()))
            train_it += 1
        if flag_stop == True:
            break
    # print the loss plot
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax1.plot(loss_record)
    ax1.title.set_text('Training Loss (MSE Loss)')
    ax1.set_xlabel('Iteration Number')

    ax2 = plt.subplot(122)
    ax2.plot(validation_loss_record)
    ax2.title.set_text('Validation Loss (MSE Loss)')
    ax2.set_xlabel('Iteration Number')
    plt.show()

    # save the model
    if args['output_model'] is not None:
        with open(args['output_model'], 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
        print('Model has been saved.')


def save_solution(x_record, path):
    file = open(path, 'w')
    for x in x_record:
        num_element = x.size
        for index in range(num_element - 1):
            file.write('{}, '.format(x[index]))
        file.write('{}\n'.format(x[-1]))
    file.close()