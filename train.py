import utils.data_utils as data_utils
import utils.model_utils as model_utils

X_train, Y_train = data_utils.load_data()

in_neurons = 1
out_neurons = 1

model = model_utils.create_network(in_neurons, out_neurons)

model.fit(X_train, Y_train, nb_epoch=500)  

fname = 'models/trained_model.hdf5'
model.save_weights(fname)