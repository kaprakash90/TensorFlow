import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot as plt

l = [i for i in range(1,500,2)] #generate input data

#generate data and label pair for a supervised modelk
pdl = pd.DataFrame(l)
pdls= pdl.shift()
dfs = [pdl, pdls]
df = pd.concat(dfs,axis=1)
df.fillna(0, inplace=True)

data = np.array(df)

train , test = npa[0:-50], npa[-50:] #train test split

#scale the data, without scaling the model predicts the same value for all the inputs
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

train_y = train_scaled[:,-1]
train_x = train_scaled[:,0:-1]
train_x = train_x.reshape(200,1,1)

#test data
test_y = test_scaled[:,-1]
test_x = test_scaled[:,0:-1]

#model
model = Sequential()
model.add(GRU(15, input_shape=(1,1)))
model.add(Dense(1))
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=500, batch_size=20, shuffle=False)
model.summary()
model.save('my_model_rnn.h5')

#test our model
y_pred = []
y_pred_ar = []
for idx, i in enumerate(test_x):
    x = i.reshape(1,1,1)
    y = model.predict(x)
    y_pred.append(y)
    new_row = list(i) + [y_pred[idx][0][0]]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    y_pred_ar.append(inverted[0,-1])

#plot the output
plt.title('test')
plt.plot(test[:,1],"ro", markersize=15, label="target")
plt.plot(y_pred_ar,"ko", markersize=5, label="pred")
plt.legend()
plt.tight_layout()
plt.show()
