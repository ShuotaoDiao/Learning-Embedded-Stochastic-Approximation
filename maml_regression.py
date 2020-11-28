import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torchmeta
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters


class RegressionNeuralNetwork(MetaModule):
    def __init__(self, in_channels, hidden1_size=40, hidden2_size=80):
        super(RegressionNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        self.regressor = MetaSequential(
            MetaLinear(in_channels, hidden1_size),
            nn.ReLU(),
            MetaLinear(hidden1_size, hidden2_size),
            nn.ReLU(),
            MetaLinear(hidden2_size, hidden1_size),
            nn.ReLU(),
            MetaLinear(hidden1_size, 1)
        )

    def forward(self, inputs, params=None):
        values = self.regressor(inputs, params=self.get_subdict(params, 'regressor'))
        # values = values.view(values.size(0),-1)
        return values

class RegressionNeuralNetwork_v2(MetaModule):
    def __init__(self, in_channels, hidden1_size=40, hidden2_size=80):
        super(RegressionNeuralNetwork_v2, self).__init__()
        self.in_channels = in_channels
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        self.regressor = MetaSequential(
            MetaLinear(in_channels, hidden1_size),
            nn.LeakyReLU(),
            MetaLinear(hidden1_size, hidden2_size),
            nn.LeakyReLU(),
            MetaLinear(hidden2_size, hidden1_size),
            nn.LeakyReLU(),
            MetaLinear(hidden1_size, 1)
        )

    def forward(self, inputs, params=None):
        values = self.regressor(inputs, params=self.get_subdict(params, 'regressor'))
        # values = values.view(values.size(0),-1)
        return values

def meta_train(args, metaDataloader):
    model = RegressionNeuralNetwork(args['in_channels'], hidden1_size=args['hidden1_size'],
                                    hidden2_size=args['hidden2_size'])
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args['beta'])
    loss_record = []
    # training loop
    for it_outer in range(args['num_it_outer']):
        model.zero_grad()

        train_dataloader = metaDataloader['train']

        test_dataloader = metaDataloader['test']

        outer_loss = torch.tensor(0., dtype=torch.float)
        for task in train_dataloader:
            iterator = iter(train_dataloader[task])
            train_sample = iterator.next()
            # get true h value
            # h_value = torch.tensor(train_sample[:,-1], dtype=torch.float)
            h_value = train_sample[:, -1].clone().detach().to(dtype=torch.float)
            # get input
            # input_value = torch.tensor(train_sample[:,:-1], dtype=torch.float)
            input_value = train_sample[:, :-1].clone().detach().to(dtype=torch.float)
            #
            train_h_value = model(input_value)
            inner_loss = F.mse_loss(train_h_value.view(-1, 1), h_value.view(-1, 1))

            model.zero_grad()
            # print('It {}, task {}, Start updating parameters'.format(it_outer, task))
            params = gradient_update_parameters(model, inner_loss, step_size=args['alpha'],
                                                first_order=args['first_order'])
            # adaptation
            # get test sample
            test_iterator = iter(test_dataloader[task])
            test_sample = test_iterator.next()
            # h_value2 = torch.tensor(test_sample[:,-1], dtype=torch.float)
            h_value2 = test_sample[:, -1].clone().detach().to(dtype=torch.float)
            # test_input_value = torch.tensor(test_sample[:,:-1], dtype=torch.float)
            test_input_value = test_sample[:, :-1].clone().detach().to(dtype=torch.float)
            test_h_value = model(test_input_value, params=params)

            outer_loss += F.mse_loss(test_h_value.view(-1, 1), h_value2.view(-1, 1))

        outer_loss.div_(args['num_tasks'])

        outer_loss.backward()
        meta_optimizer.step()

        loss_record.append(outer_loss.detach())
        if it_outer % 50 == 0:
            print('It {}, outer traning loss: {}'.format(it_outer, outer_loss))
            # print the loss plot
    plt.plot(loss_record)
    plt.title('Outer Training Loss (MSE Loss) in MAML')
    plt.xlabel('Iteration number')
    plt.show()

    # save model
    if args['output_model'] is not None:
        with open(args['output_model'], 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

def meta_train_v2(args, metaDataloader):
    model = RegressionNeuralNetwork_v2(args['in_channels'], hidden1_size=args['hidden1_size'],
                                    hidden2_size=args['hidden2_size'])
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args['beta'])
    loss_record = []
    # training loop
    for it_outer in range(args['num_it_outer']):
        model.zero_grad()

        train_dataloader = metaDataloader['train']

        test_dataloader = metaDataloader['test']

        outer_loss = torch.tensor(0., dtype=torch.float)
        for task in train_dataloader:
            iterator = iter(train_dataloader[task])
            train_sample = iterator.next()
            # get true h value
            # h_value = torch.tensor(train_sample[:,-1], dtype=torch.float)
            h_value = train_sample[:, -1].clone().detach().to(dtype=torch.float)
            # get input
            # input_value = torch.tensor(train_sample[:,:-1], dtype=torch.float)
            input_value = train_sample[:, :-1].clone().detach().to(dtype=torch.float)
            #
            train_h_value = model(input_value)
            inner_loss = F.mse_loss(train_h_value.view(-1, 1), h_value.view(-1, 1))

            model.zero_grad()
            # print('It {}, task {}, Start updating parameters'.format(it_outer, task))
            params = gradient_update_parameters(model, inner_loss, step_size=args['alpha'],
                                                first_order=args['first_order'])
            # adaptation
            # get test sample
            test_iterator = iter(test_dataloader[task])
            test_sample = test_iterator.next()
            # h_value2 = torch.tensor(test_sample[:,-1], dtype=torch.float)
            h_value2 = test_sample[:, -1].clone().detach().to(dtype=torch.float)
            # test_input_value = torch.tensor(test_sample[:,:-1], dtype=torch.float)
            test_input_value = test_sample[:, :-1].clone().detach().to(dtype=torch.float)
            test_h_value = model(test_input_value, params=params)

            outer_loss += F.mse_loss(test_h_value.view(-1, 1), h_value2.view(-1, 1))

        outer_loss.div_(args['num_tasks'])

        outer_loss.backward()
        meta_optimizer.step()

        loss_record.append(outer_loss.detach())
        if it_outer % 50 == 0:
            print('It {}, outer traning loss: {}'.format(it_outer, outer_loss))
            # print the loss plot
    plt.plot(loss_record)
    plt.title('Outer Training Loss (MSE Loss) in MAML')
    plt.xlabel('Iteration number')
    plt.show()

    # save model
    if args['output_model'] is not None:
        with open(args['output_model'], 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)


def fine_tune(args, model, dataloader, validation_set):
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
            if train_it % 5 == 0:
                print("It {}, L2 training loss: {} ".format(train_it, loss.item()))
                print("It {}, L2 validation loss: {}".format(train_it, val_loss.item()))
            if train_it > args['Max_it']:
                print('Stop fine-tuning')
                flag_stop = True
                break
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
        print('The model is saved.')


