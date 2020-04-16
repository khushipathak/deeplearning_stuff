# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values #we keep customer id only for identification
y = dataset.iloc[:, -1].values #binary values if their applications are approved or not

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)



# Training the SOM
from minisom import MiniSom
''' x and y are dimensions, 15 will be the no. of variables in X which are the 14 variables plus 
customer id
sigma is the radius
learning_rate is the rate at which the weights will be updated
decay parameter may be used to improve the convergence'''
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X) #for the initialisation
som.train_random(data = X, num_iteration = 100) #this will train the model on X using reinforcement
#num_iteration is how many times we want the weights to be updayed



# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) #took a transpose of the Mean Interneuron Distances (MID) matrix
colorbar() #will tell us which colour represents high MID and which one represents low MID
#btw they were scaled values
#since frauds are the outliers, the white nodes (high MID i.e. different from normals) are frauds
markers = ['o', 's'] #circles and squares
colors = ['r', 'g'] #red and green

#red circles = rejected customers
#green squares = accepted customers
for i, x in enumerate(X):   #i is number of rows i.e. each customer and x is vectors of customers at different iterations 
    w = som.winner(x) #we got the winning node for this customer
    #now we want to mark this winning node
    plot(w[0] + 0.5, #centre vertically
         w[1] + 0.5, #centre horizontally
         #we put the marker at the centre of the square
         markers[y[i]], #marked accordingly if approved or not
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)