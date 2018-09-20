# Import
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# convert an array of values into a dataset matrix
def create_dataset(dataset, days_in_advance):
  dataX, dataY = [], []
  for i in range(len(dataset)):
    if (i + days_in_advance < len(dataset)):
      dataX.append(dataset[i])
      dataY.append(dataset[i + days_in_advance])
  return np.asarray(dataX), np.asarray(dataY)

# Import data
# data = pd.read_csv('./data/all_eth.csv')
data = pd.read_csv('./data/all_bitcoin.csv')
# data = pd.read_csv('./data/data_stocks.csv')

# Drop variables

## data_stocks:
# data = data.drop(['DATE'], 1)

## all_eth && all_bitcoin
data = data.drop(['Date'], 1)
data = data.drop(['Open'], 1)
data = data.drop(['High'], 1)
data = data.drop(['Low'], 1)
data = data.drop(['Volume'], 1)
data = data.drop(['Market Cap'], 1)

# Make data a np.array
data = data.values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

#prepare the X and Y label
X,y = create_dataset(data, int(sys.argv[1]))

#Take 80% of data as the training sample and 20% as testing sample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# Neurons
n_neurons_1 = 256
n_neurons_2 = 128
n_neurons_3 = 64
n_neurons_4 = 32

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([X_train.shape[1], n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Fit neural net
batch_size = 10
pred = []

# Run
epochs = 5
for e in range(epochs):
  # Minibatch training
  for i in range(0, len(y_train) // batch_size):
    start = i * batch_size
    batch_x = X_train[start:start + batch_size]
    batch_y = y_train[start:start + batch_size]
    # Run optimizer with batch
    net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    # Prediction
    pred = net.run(out, feed_dict={X: X_test})

pred = [ [i] for i in pred[0] ]

#Inverse Transform the predicted and testing data outputs to get accuracy
testPredict = scaler.inverse_transform(pred)
testY = scaler.inverse_transform(y_test)

acc = 0
for index,element in enumerate(testY) :
  acc +=  1 - abs((element - testPredict[index])[0])/element[0]
acc /= len(testY)

print("Prediction: ", testPredict[-1])
print("Actual: ", testY[-1])
print("Accuracy: ", acc * 100)

# plot baseline and predictions
plt.plot(testY, label="Actual Price")
plt.plot(testPredict, label="Predicted Price")
plt.xlabel("Day")
plt.ylabel("Ethereum Price")
plt.legend()
plt.show()
