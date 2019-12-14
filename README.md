# Kriging Convolutional Networks(KCN)

###Overview

This repo contains the implementation of Kriging Convolutional Networks algorithm:

Gabriel Appleby\*, Linfeng Liu\*, Li-Ping Liu, Kriging Convolutional Networks
Gabriel, To appear on AAAI 2020.



### Requirements

* tensorflow (>=1.13.2)
* keras (>=2.2.4)
* sklearn
* scipy



### Data

Our model takes 

* Data: a numpy saved file (.npz) containing:
  * data['Xtrain']: N_tr by D matrix. Here N_tr is the number of training nodes, and D is the feature dimensionality.
  * data['Xtest']: N_te by D matrix. N_te is the number of testing nodes.
  * data['Ytrain']: N_tr by T matrix. T is the targe dimensionality.
  * data['Ytest']: N_te by T matrix.
* n_neighbors: int. The number of nearest neighbors for each node.
* hidden1: int. The number of units in hidden layer 1.
* hidden2: int. The number of units in hidden layer 2. Assigining -1 means not to use this layer.
* dropout: float. The dropout rate.
* kernel_length: folat. The Kernel length for the Gaussian kernel.



### Model

There are three models you can choose: kcn, kcn_att, and kcn_sage. 



###Run the code

Please refer to the `experiment.py` file for details.

####Acknowledge
For the kcn and kcn-att, we leverage the code from Thomas N. Kipf (https://github.com/tkipf/gcn). For the implementation of kcn-sage, we take advantage of `spektral` graph neural network package (https://github.com/danielegrattarola/spektral). We thanks these authors to make their code publicly available.