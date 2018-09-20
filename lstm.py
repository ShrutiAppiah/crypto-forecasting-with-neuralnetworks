# Import
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# convert an array of values into a dataset matrix
def create_dataset(dataset, days_in_advance):
  dataX, dataY = [], []
  for i in range(len(dataset)):
    if (i + days_in_advance < len(dataset)):
      dataX.append(dataset[i])
      dataY.append(dataset[i + days_in_advance])
  return np.asarray(dataX), np.asarray(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
df = read_csv('./data/all_bitcoin.csv')
# df = read_csv('./data/all_eth.csv')
# df = read_csv('./data/data_stocks.csv')
gt = read_csv('./data/GoogleTrends.csv')
df = df.iloc[::-1]
## all_eth && all_bitcoin
df = df.drop(['Date','Open','High','Low','Volume','Market Cap'], axis=1)
## data_stocks
# df = df.drop(['DATE'], axis=1)
dataset = df.values
dataset = dataset.astype('float32')
## all_bitcoin
gt = gt.drop(['Day','ethereum','Cryptocurrency'],axis=1)
## all_eth
# gt = gt.drop(['Day','bitcoin','Cryptocurrency'],axis=1)
gdataset = gt.values
gdataset = gdataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#gdataset = scaler.fit_transform(gdataset)

#prepare the X and Y label
X,y = create_dataset(dataset, int(sys.argv[1]))

#Take 80% of data as the training sample and 20% as testing sample
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.20, shuffle=False)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(256, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5, batch_size=10, verbose=2)

# save model for later use
# model.save('./savedModel')
# load_model
# model = load_model('./bitsavedModel')

# # make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

futurePredict = model.predict(np.asarray([[testPredict[-1]]]))
futurePredict = scaler.inverse_transform(futurePredict)

# # invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

print("Price Prediction for last 5 days: ")
print(testPredict[-5:])
print("Bitcoin price for tomorrow: ", futurePredict)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict):len(dataset)-1, :] = testPredict

print(testPredict)

# calculate accuracies
testAcc = 0
for index,element in enumerate(testY) :
  testAcc +=  1 - abs((element - testPredict[index])[0])/element[0]
testAcc /= len(testY)

trainAcc = 0
for index,element in enumerate(trainY) :
  trainAcc += 1 - abs((element - trainPredict[index])[0])/element[0]
trainAcc /= len(trainY)

print("Prediction: ", testPredict[-1])
print("Actual: ", testY[-1])
print("Training Accuracy: ", trainAcc * 100)
print("Testing Accuracy: ", testAcc * 100)

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset),label= "Actual Price")
plt.plot(trainPredictPlot,label = "Training Price")
plt.plot(testPredictPlot,label="Predicted Price")
plt.legend()
plt.xlabel('Day')
# all_bitcoin
plt.ylabel('Bitcoin Price')
# all_eth
# plt.ylabel('Ethereum Price')
# plot google trends
ax2 = plt.twinx()
ax2.plot(gdataset, color="purple", linestyle="dotted",label="Popularity")
# all_bitcoin
ax2.set_ylabel('Bitcoin Trends Popularity')
# all_eth
# ax2.set_ylabel('Ethereum Trends Popularity')
ax2.legend(loc = "lower right")
plt.show()
