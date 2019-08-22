import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import argparse
import sys
import syft as sy
from syft.workers import WebsocketClientWorker
from syft.workers import VirtualWorker
from syft.frameworks.torch.federated import utils

logger = logging.getLogger(__name__)
LOG_INTERVAL = 25

class Model(nn.Module):
    '''
    Neural Network Example Model
    
    Attributes
    :hidden_layers (nn.ModuleList) - hidden units and dimensions for each layer
    :output (nn.Linear) - final fully-connected layer to handle output for model
    :dropout (nn.Dropout) - handling of layer-wise drop-out parameter
    
    Functions
    :forward - handling of forward pass of datum through the network.
    '''
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(args.in_size,
                                                      args.hidden_layers[0])])

        #create hidden layers
        layer_sizes = zip(args.hidden_layers[:-1], args.hidden_layers[1:]) 
        #gives input/output sizes for each layer
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(args.hidden_layers[-1], args.out_size)
        self.dropout = None if args.drop_p is not None \
                                            else nn.Dropout(p=args.drop_p)
        
    def forward(self, x):
        '''
        Forward pass through each layer of neural network.
        Apply dropout as needed, and use ReLU as as activation function
        Return output using final activation function.
        Should be Softmax for MSE Loss and LogSoftmax for NLLLoss
        '''
        for each in self.hidden_layers:
            x = F.relu(each(x)) #apply relu to each hidden node
            
            if self.dropout is not None:
                x = self.dropout(x) #apply dropout
                
        x = self.output(x) #apply output weights
        return args.activation(x, dim=args.dim) #apply activation log softmax
    
def encrypted_federated_train(model, datasets, optimizer, args):
    '''
    Train federated model on worker and their data. Both should be
    already encrypted, so we are training using Secure Multi-Party Computation
    
    Inputs:
        model (Model) : network to train
        datasets (Datasets) : remote data to train on
        args (Arguments) : specific arguments for training
        
    Outputs:
        model (Model): trained model
        training_loss(list, float): training loss at each iteration
    '''
    print(f'SMPC Training on {len(datasets)} remote workers (dataowners)')
    steps = 0
    model.train()  # training mode

    for e in range(1, args.epochs+1):
        running_loss = 0
        for ii, (data, target) in enumerate(datasets):
            # iterates over pointers to remote data
            steps += 1
            # TODO model.send()?
            # NB the steps below all happen remotely
            # zero out gradients so that one forward pass doesnt pick up
            # previous forward's gradients
            optimizer.zero_grad()
            outputs = model.forward(data)  # make prediction
            # get shape of (1,2) as we need at least two dimension
            outputs = outputs.reshape(1, -1)
            loss = ((outputs - target)**2).sum().refresh()
            loss.backward()
            optimizer.step()

            # get loss from remote worker and unencrypt
            _loss = loss.get().float_precision()
            running_loss += _loss

            print_every = 100
            if steps % print_every == 0:
                print('Train Epoch: {} [{}/{}]  \tLoss: {:.6f}'.format(e,
                                        ii+1, len(datasets), _loss/print_every))

                running_loss = 0