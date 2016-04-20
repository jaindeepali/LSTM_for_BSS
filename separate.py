import utils.data_utils as data_utils
import utils.model_utils as model_utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print "Loading data ..."
X_test, Y_test = data_utils.load_data()
print "Data loaded."

in_neurons = 1
out_neurons = 1
print "Loading model ..."
model = model_utils.create_network(in_neurons, out_neurons)
fname = 'models/trained_model.hdf5'
model.load_weights(fname)
print "Model loaded."

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))

print 'Error:' + str(rmse)

pd.DataFrame(predicted).to_csv("data/predicted.csv")

plt.plot(predicted[1:1000])
plt.savefig('data/out.png')
