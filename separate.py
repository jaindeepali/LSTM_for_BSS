import utils.data_utils as data_utils
import utils.model_utils as model_utils
import numpy as np
import pandas as pd
import sys
import os

print "Loading data ..."
X_test, Y_test = data_utils.load_data(domain='freq')
print "Data loaded."

in_neurons = X_test.shape[-1]
out_neurons = Y_test.shape[-1]

try:
	iteration = sys.argv[1]
except:
	iteration = 0

print "Loading model ..."
model = model_utils.create_network(in_neurons, out_neurons)
fname = 'models/trained_model_' + str(iteration) + '.hdf5'
model.load_weights(fname)
print "Model loaded."

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))

print 'Error:' + str(rmse)

data_utils.save_output(predicted, domain='freq')
