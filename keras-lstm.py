import pandas as pd  
from random import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from pylab import *

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout  
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop


fs1, signal1 = wav.read('sound_files/input/sine/1.wav')
fs2, signal2 = wav.read('sound_files/input/sine/2.wav')
a = np.asmatrix(np.random.rand(2))
s = np.asmatrix([signal1, signal2])
mixed = a * s

s1_spec = specgram(signal1, Fs=fs1)
s2_spec = specgram(signal2, Fs=fs2)
mixed_spec = specgram(mixed, Fs=fs1)


## Triangular wave
# flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  
# pdata = pd.DataFrame(flow)  
# data = pdata * random()  # some noise  

def load_data(n_prev = 1):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

in_neurons = 1
out_neurons = 1

model = Sequential()  
model.add(LSTM(output_dim=500, input_dim=in_neurons, return_sequences=True))
model.add(LSTM(output_dim=800, input_dim=500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(output_dim=300, input_dim=800, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=out_neurons, input_dim=300))
model.add(Activation("linear"))
rmsprop = RMSprop(lr=0.5, rho=0.9, epsilon=1e-06)
model.compile(loss="mean_squared_error", optimizer=rmsprop)  

(X_train, y_train) = load_data()  # retrieve data
model.fit(X_train, y_train, batch_size=450, nb_epoch=1000, validation_split=0.05)  

predicted = model.predict(X_train)
rmse = np.sqrt(((predicted - y_train) ** 2).mean(axis=0))

print 'Error:' + str(rmse)

pd.DataFrame(predicted).to_csv("data/predicted.csv")

plt.plot(predicted[1:100])
plt.show()

out_wave = wave.open('sound_files/output/1.wav', 'w')
out_wave.setparams(in_wave.getparams())
out_wave.writeframes(predicted)
