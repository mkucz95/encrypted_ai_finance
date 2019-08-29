from torch import nn
from torch import optim
import torch.nn.functional as F
import syft as sy
import torch as th

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
        self.dropout = None if args.drop_p is None \
                                            else nn.Dropout(p=args.drop_p)
        self.args=args
        
    def forward(self, x):
        x = x.view(-1, self.args.in_size)
        for each in self.hidden_layers:
            x = F.relu(each(x)) #apply relu to each hidden node
            
            if self.dropout is not None:
                x = self.dropout(x) #apply dropout
                
        x = self.output(x) #apply output weights
        
        if self.args.activation is None:
            return x
        
        return self.args.activation(x, dim=self.args.dim) #apply activation log softmax
    
def connect_to_workers(n_workers, hook, secure_worker=False):
    '''
    Connect to remote workers with PySyft
    
    Inputs
        n_workers (int) - how many workers to connect to
        secure_worker (bool) - whether to return a trusted aggregator as well
        
    Outputs
        workers (list, <PySyft.VirtualWorker>)
    '''
    workers = [sy.VirtualWorker(hook, id=f"w_{i}") for i in range(n_workers)]

    if secure_worker:
        return workers, sy.VirtualWorker(hook, id='trusted_aggregator')

    else:
        return workers