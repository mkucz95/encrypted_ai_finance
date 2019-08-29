# Private and Encrypted AI - Credit Approval Application
## Federated & Encrypted Deep Learning

### Project Description
This project is an experiment in differential privacy and federated learning as privacy preserving techniques for deep learning. Privacy has recently been at the forefront of news cycles (with Facebook and other tech giants specifically under scrutiny). 

The idea behind encrypted deep learning is simple: preserve people's privacy while aligorithms enabling machine learning models to be developed on this data.

The *project goal* is to show encrypted deep learning in a realistic use case - credit applications. Some of the most significant data breaches in recent years have originated at the hands of the financial industry. More specifically, the credit sector, especially Equifax, have showed both that they are high profile targets for entities with malicious intent and that their data protection techniques are lacking at best.

It would be an improvement over the current system if a user was able to encrypt their databefore passing it into a model that either approves or rejects the individuals application for a new line of credit. This would also serve to simplify the data protection measures currently necessary for companies providing credit related services. Credit providers could make their encrypted models public, or license them to other firms without fearing a breach of intellectual property.The firms licensing the model would be able to use it without knowing the parameters of said model.

Privacy preservation in machine learning is built upon [Differential Privacy](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/differential-privacy.html), and the  cryptography techniques used in this project includes [Secure Multi-Party Computation (SMPC)](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/secure_multi_party_computation.html).

_Note_ <br>This project was inspired by lectures of [Andrew Trask](https://iamtrask.github.io/) in the [Private AI Scholarship Challenge on Udacity](https://www.udacity.com/facebook-AI-scholarship). Furthermore, segments of the code are inspired by the [PySyft tutorials on GitHub](https://github.com/OpenMined/PySyft/tree/dev/examples/tutorials); an excellent resource for people starting off with Private AI. 

#### Layout
I have put together a series of Jupyter Notebooks that show how I have worked through first federated learning and later encrypted deep learning and put those together into a model that can hand credit scores in an encryted and federated manner. This is accompanied by notebooks explaining [Differential Privacy](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/differential-privacy.html) and [Secure Multi-Party Computation (SMPC)](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/secure_multi_party_computation.html), both for those who are unfamiliar with the subject but also for myself since I learn best by explaining to others.

The second party of the project is code that can be run from the command line and uses websockets on a variety of locally-hosted servers to mimic a real-world federated learning setup.

    .
    ├── data                       # data and model storage
    │   └── ...    
    |
    ├── notebooks               
    │   ├── nn-dev.ipynb                             # initial NN dev, 3 layer network
    │   ├── federated_dl.ipynb                       # training model on remote workers
    │   ├── federated_dl_model_averaging.ipynb       # training numerous models across all workers
    │   ├── encrypted_dl.ipynb                       # fully encrypted NN
    │   ├── secure_multi_party_computation.ipynb     # my explanation of and notes on SMPC
    │   └──  differential-privacy.ipynb              # my understanding of DP
    |
    ├── websocket_project               
    │   ├── run_websocket_client.py          # starts single remote client worker on localhost port
    │   ├── run_websocket_server.py          # create websocket server to manage training
    │   └── start_websocket_servers.py       # create remote client workers by calling run_websocket_client
    │  
    └── ...

<hr>

## Data
I am using a publically available data set for the purpose of this example. The data set contains 15 anonymixed features (for privacy protection) and a binary label. I have created fake names and features to illustrate how privacy protection would work in reality. The current anonymization of the dataset is unlikely to have fully preserved privacy. Without practicing proper differential privacy, it is widely know that most 'anonymized' datasets can be reverse engineered, or at the very least that not all anonymity is preserved. 

The data comes courtesy of [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Credit+Approval).

## Project Steps
### Neural Network Development
[notebook](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/nn-dev.html)<br><br>
Data processing to mimic gathering data across many people. In reality, we would never want data to leave the ownership of any individual. Even better, we would want to make sure that not only does the data stay on the devices of these individuals, but also that computations are performed on encrypted data to preserve the maximum amount of privacy possible. 

Here I develop as simple 3-layer fully connected neural network and train it on a training set to save time for future notebooks. This way it is also simpler to prove that this is a solvable problem, but this step is not private at all.

### Federated Deep Learning
[notebook](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/federated_dl.html)<br><br>
Here we train a model on remote data by sending the model to each of these remote workers. We then recieve back an updated version of this model. This is does not yet protect privacy to the extent that we would like. It is entirely possible to deduce information about a specific individual based on the gradients and updated weights of the model that is returned.

### Federated Deep Learning with Model Averaging
[notebook](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/federated_dl_model_averaging.html)<br><br>
To build on the previous step, here we rely on a '*trusted third-party aggregator*' to add privacy in the training process. Each remote worker has a copy of the original model, train their own versions of the model before sending the weights and biases of each of these models to be averaged (using a simple average). This results in a final model that is trained across all the data, but with weights and biases (and change thereof) which cannot be attributed to a single worker. However, the downside of this approach (even though it is still rather computationally efficient) is that it relies on trusting a third-party which is not always wise.

### Encrypted Deep Learning
[notebook](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/encrypted_dl.html)<br><br>
Encrypted deep learning uses fundementals of Secure Multi-Party Computation to remove the need of any third party trust while retaining high degrees of privacy. This is the most computationally intensive, but also privacy preserving, of the methods I have implemented. The tough thing with encrypted deep learning is that we also have to encode all numbers into fixed precision, which limits us in the loss function and predictive calculations we can make while also decreasing the accuracy of the network.

### Federation with WebsocketServerWorkers
[project](https://github.com/mkucz95/private_ai_finance/tree/master/websocket_project#capstone-project)<br><br>
A simulation of a realistic situation where there are remote client workers running on different `localhost` ports with data split amongst them. The training is performed using websockets across these workers. This is the closest to a real implementation of this code. Check out the write-up and code for this [here](https://github.com/mkucz95/private_ai_finance/tree/master/websocket_project#capstone-project).
*Warning*: This code still has bugs.

<hr>

## Conclusion

Even though the training of an encrypted neural network is more nuanced than an unencrypted one, at a minimum, one can encrypt a network, make predictions using this network on encrypted data, preserving privacy. Furthermore, an extension to this is to apply the softmax activation function AFTER all the other computation is done, at least when testing or getting predictions, which could give a more accurate indication into the performance of the neural network.

Even in the scenario where all data and computation is encrypted it does not prevent an adversarial attack where shares are intentionally corrupted during computation. This is generally considered an open problem in SMPC and encrypted deep learning.

A look at the results of the network on the testing set indicates that while performance did suffer slightly from the unencrypted model (could be largely due to the fixed precision encoding of all gradients and data) the performance is largely in line with that of the non-private model. Even though it does not have 100% accuracy, a case could be made for the fact that the labels from the original data could be wrong in the sense that certain people may have been accepted for credit when they shouldn't have been. This is a problem with the underlying dataset, which would be fixed by labeling the data based on who repaid their credit and who didn't.

#### Differential Privacy for Deep Learning
Differential privacy techniques provide certain guarantees for privacy in the context of deep learning. Instead of encrypting data, we add noise to the data (local DP) or to the output of a query (global DP) such that privacy is preserved to an acceptable degree. To familiarize yourself with Differential Privacy, visit a short guide I have put together [here](https://htmlpreview.github.io/?https://github.com/mkucz95/private_ai_finance/blob/master/notebooks/differential-privacy.html). For the purpose of this example, however, I have not implemented differential privacy since data will be encrypted end-to-end anyway. However, one could have private deep learning employing differential privacy on a local or global level, and then work with unencrypted data, gradients, and models. Generally, one could preserve privacy very succesfully in the federated learning approach by utilizing local differential privacy (adding noise to the data before training each model). This would preserve privacy (although not as well as full encryption) towards both a trusted third party aggregator or directly to those training the network.

#### Future Development
- **adding differential privacy preserving techniques** such as local differential privacy
- training data in fully federated and encrypted mode
- websocket based training as a realistic scenario for federated learning (with raspberry pies as bonus)

<hr>

Finally, Please checkout out the [OpenMined PySyft code](https://github.com/OpenMined/PySyft/)! This code base is built on contributions from amazing people, and always looks to expand!

##### How to Run This Code:
1. clone the repository
2. [create and run a virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
3. install dependencies in this new environment `pip install -r requirements.txt`
4. You're ready to go!