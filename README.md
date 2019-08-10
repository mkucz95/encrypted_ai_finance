# Federated & Encrypted Deep Learning for Credit Approval Classification
## Private AI - Finance

### Project Description
This project is an experiment in differential privacy and federated learning as privacy preserving techniques for deep learning. Privacy has recently been at the forefront of news cycles (with Facebook and other tech giants specifically under scrutiny). 

The idea behind encrypted deep learning is simple: preserve people's privacy while aligorithms enabling machine learning models to be developed on this data.

The *project goal* is to show encrypted deep learning in a realistic use case - credit applications. Some of the most significant data breaches in recent years have originated at the hands of the financial industry. More specifically, the credit sector, especially Equifax, have showed both that they are high profile targets for entities with malicious intent and that their data protection techniques are lacking at best.

It would be an improvement over the current system if a user was able to encrypt their databefore passing it into a model that either approves or rejects the individuals application for a new line of credit. This would also serve to simplify the data protection measures currently necessary for companies providing credit related services. Credit providers could make their encrypted models public, or license them to other firms without fearing a breach of intellectual property.The firms licensing the model would be able to use it without knowing the parameters of said model.

## Data
I am using a publically available data set for the purpose of this example. The data set contains 15 anonymixed features (for privacy protection) and a binary label. I have created fake names and features to illustrate how privacy protection would work in reality. The current anonymization of the dataset is unlikely to have fully preserved privacy. Without practicing proper differential privacy, it is widely know that most 'anonymized' datasets can be reverse engineered, or at the very least that not all anonymity is preserved. 

#TODO: Add info about the features I made up

The data comes courtesy of [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Credit+Approval).   


