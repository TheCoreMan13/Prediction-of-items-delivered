from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np



df = pd.read_csv('orders_autumn_2020.csv',parse_dates=True, index_col='TIMESTAMP' )

#Here we choose the period we want to analyze
df=df.loc['2020-08-08':'2020-08-09']
orders = len(df)


#Data cleaning

columns_to_keep = ['ITEM_COUNT']
df = df[columns_to_keep]
df['ITEM_COUNT'] = df['ITEM_COUNT'].apply(lambda x: x*1000)
df.index.names = ['TIMESTAMP']
df.sort_index(inplace=True)
df.head()
df.describe()
df.plot()
df.isnull().sum()
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.hist(bins=10)
len(df[df['ITEM_COUNT'] == 0])

#I extract the NumPy array from the dataframe and convert the integer values to floating point values, which are more suitable for modeling with a neural network.
dataset = df.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)



#Create recurent neuronal networks
train_size = int(len(scaled) * 0.70)
test_size = len(scaled - train_size)
train, test = scaled[0:train_size, :], scaled[train_size: len(scaled), :]

#Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataset[i + look_back, 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=2, shuffle=True)

#Make preditions

trainPredict = model.predict(X_train, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(X_test, batch_size=batch_size)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])

# Calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))

#Ploting

trainPredictPlot = np.empty_like(scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaled)-1, :] = testPredict

# Plot baseline and predictions

plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(scaled),'-o', drawstyle='default', label='Scaler',color = 'blue' )
plt.plot(trainPredictPlot, '-o', drawstyle='default', label='Trained Predict Items Delivered',color = 'red')
plt.plot(testPredictPlot, '-o',drawstyle='default', label='Prediction Items Delivered', color = 'green')
plt.title('Prediction Items Delivered')
plt.xlabel('Time')
plt.ylabel('Items Delivered')
nr_orders = mpatches.Patch(label='Numbers of orders in the selected period: {}'.format(orders))
plt.legend(handles=[nr_orders])
plt.show()














