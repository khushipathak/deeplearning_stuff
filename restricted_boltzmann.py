# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn #for neural networks
import torch.nn.parallel #parallel computattion
import torch.optim as optim #opimisation
import torch.utils.data #tools we need
from torch.autograd import Variable #for stochastic gradient descent

# Importing the dataset
'''we put the separator because different names arent differentiated by commas'''
'''we put header as none because in the dataset there are no column names'''
'''engine=python to make sure the dataset gets imported correctly'''
'''encoding UTF-8 won't be enough for the special characters in some movie titles so we want latin-1'''
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#in the rating column, column 1 is user id, column 2 is movie id, column 3 is rating, clumn 4 is timestamps


#the file already contains training sets and test sets
'''we have 5 pairs of train&test sets in case we wanted to do five fold cross validation'''
# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
'''max user id out of all user ids in test and training set'''
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
'''same for movies'''
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]  #all the movies rated by a particular user
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors (basically arrays only, but of the Torch type)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1 #didnt rate
training_set[training_set == 1] = 0#didnt like
training_set[training_set == 2] = 0#didnt like
training_set[training_set >= 3] = 1#liked
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    '''you always have to start with the __init__ to defiine the class parameters'''
    '''nv = no. of visible nodes, nh= no. of hidden nodes'''
    def __init__(self, nv, nh):   #to attach parameters to a method you have to put a parameter as self in the start
        self.W = torch.randn(nh, nv) #now we are initialising the parameters of the objects i.e. weights and bias
        self.a = torch.randn(1, nh) #bias for hidden nodes
        self.b = torch.randn(1, nv) #bias for visible nodes
    def sample_h(self, x): #refer to the pdf 
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)  #eg. prolly of a user liking a genre given the ratings he gave
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) #eg. prolly of user liking a movie given his preferences
        return p_v_given_h, torch.bernoulli(p_v_given_h) 
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))