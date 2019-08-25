# Capstone Project

This is based on the OpenMined advanced tutorial that can be found [here](https://github.com/OpenMined/PySyft/blob/dev/examples/tutorials/advanced/websockets-example-MNIST-parallel/Asynchronous-federated-learning-on-MNIST.ipynb).

## Inspiration

This project is inspired by providing encrypted and secure modelling for financial data. This type of information is very fragile since it is collected by many industries, for example credit rating or credit providing businesses, but has been often mishandled by the companies that collect this data. Unfortunately, there is currently no great alternative to giving up ownership of one's financial data in return for access to capital. The goal of this project is to prove that a credit rating model is possible to train without the ownership of any data.

There are numerous use cases for such a model including:
- training on data that various credit providers have collected on their customers without ever handling the data directly.
    - the benefits of this include not needing to trust the entity that wants to use the data to create a model
    - being able to include information about creditworthiness across many businesses and industries which would inform a complete model
- training encrypted and federated models on data using individual customer's data directly. This example was developed in the tutorial, exploratory notebooks of this project. The challenge to this is creating trust that encrypted and federated methodologies really do preserve privacy and that the opportunity for misuse is almost entirely eliminated. Realistically, the prior use-case is more implementable, and scalable, both in a computational and business sense.

**Interesting note** is that such use cases, and encrypted deep learning in general, can be paired with *blockchain* to enable previosuly unconcievable solutions for common problems. For instance, user's financial data could be kept (encrypted) on a blockchain, which then tracks any models or other entities that use, access, or attempt to access said data. This could then be useful to track how various data for each customer has changed over time, still encrypted of course, but the models that could access this data would be able to form intuitions on how financial data has changed over time and how models need to react to various changes. It would also enable a broader training dataset for any algorithm which would in theory allow better allocation and access to capital. 

## Setup

This code uses an experimental setup called `TrainConfig` developed by OpenMined members. Generally speaking, this involves:

1. Data ownership by individual workers (to train on)
2. What is called a `scheduler` which is aware of the worker names and dataset ownership (the entity interested in the model computation)

Each worker can be represented by two parts:
1. the remote *websocket server worker* which holds the data and performs computations
2. The part that lives on the `scheduler` - a proxy local to the scheduler called the *websocket client worker*

To understand how some of the training and testing processes get carried out in parallel, it is useful to be acquainted with [TorchScript, a PyTorch implementation of JIT tracing and scripting](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting) where python code is compiled into machine-executable code (and then potentially back into python) for performance and multi-processing reasons.

## How to Use

1. run `python start_websocket_server.py` this creates (by default) three remote workers. This then also allocates the each remote worker's dataset, which in reality would already be contained on each respective worker. This works by calling `python run_websocket_server.py  *args` for each desired remote server worker.
2. run `python run_websocket_client.py` to run the training by instantiating a model, creating a test set on a `testing worker` and then performing federated learning with model averaging.
