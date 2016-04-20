import utils.data_utils as data_utils
import utils.model_utils as model_utils

print "Loading data ..."
X_train, Y_train = data_utils.load_data()
print "Data loaded"

in_neurons = 1
out_neurons = 1

model = model_utils.create_network(in_neurons, out_neurons)

print "Training ..."
model.fit(X_train, Y_train, batch_size=500, nb_epoch=500)  
print "Training complete"

fname = 'models/trained_model.hdf5'
model.save_weights(fname)