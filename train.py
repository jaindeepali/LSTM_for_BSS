import utils.data_utils as data_utils
import utils.model_utils as model_utils

print "Loading data ..."
X_train, Y_train = data_utils.load_data()
print "Data loaded"

in_neurons = 1
out_neurons = 1

model = model_utils.create_network(in_neurons, out_neurons)

print "Training ..."
iterations = 50
epochs_per_iteration = 10
batch_size = 10000

for iteration in range(iterations):
	print "Iteration Number: " + str(i)
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs_per_iteration)  
	fname = os.path.join('models','trained_model_' + str(i) + '.hdf5')
	model.save_weights(fname)

print "Training complete"

