# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
'''this is different that the scaling we usually did, this time we chose this scaler because
we want to observe changing trends'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
'''VERY IMPORTANT!!!'''
'''Okay, so 60 timesteps means that at each time T,

the RNN is going to look

at the 60 stock prices before time T,

i.e. the stock prices between 60 days before time T

and time T, and based on the trends that

it is capturing during these 60 previous timesteps,

it will try to predict the next output.

So 60 timesteps of the past information

from which our RNN is gonna try to learn

and understand some correlation'''
'''60 was decided experimentally'''
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    '''rnn will memorise the 60 timestamps prices to predict what the next price is, and
learn based on the results of its predictions'''
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# needed because this size will be compatible with all our future functions
# will give us a 3d dataset
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
'''
units = number of lstm cells or memory units that you want in the lstm layer
return_sequences = TRUE if you want to add another lstm layer after this
input_shape = the shape of the input contained in X-train that we just created
only 2 dimensions specified, one for the timestamps, and the other for the indicators
'''

'''we chose 50(large number) units(neurons) to solve this complicated problem'''
'''gives us a model with high dimensionality'''


regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #drop 20% irrelevant neurons

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# return_sequences = True by default
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
#it's a normal layer like those in ANN
#only one unit because only one stock price is predicted
regressor.add(Dense(units = 1))

# Compiling the RNN
# we chose the loss because we have a regression problem and not classification
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# instead of sending all the observations into the NN and then backpropagating to resolve errors
# we will send the observations in groups of 32 at a time so batch_size = 32
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

'''imp'''
# Getting the predicted stock price of 2017
''' we will concatenate the training_set and test_set because we want the prev 60 days 
for all the dates in the test_set, and some of them are in the training_set'''

'''axis=0 is vertical, axis=1 is horizantal'''
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

'''following will give us 60 days before the first financial day of January 2017'''
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

'''reshaping required'''
inputs = inputs.reshape(-1,1)

''' we also need to scale the input based on the scaling on the training_set because
our model was trained on the scaled set'''
inputs = sc.transform(inputs)

'''now to prepare the proper test set with 60 days before'''
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

'''now predict'''
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
