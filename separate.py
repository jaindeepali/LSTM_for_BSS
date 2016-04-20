import utils.data_utils as data_utils
import utils.model_utils as model_utils
import numpy as np

X_test, Y_test = data_utils.load_data()

in_neurons = 1
out_neurons = 1

model = model_utils.create_network(in_neurons, out_neurons)
fname = 'models/trained_model.hdf5'
model.load_weights(fname)

predicted = model.predict(X_train)
rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))

print 'Error:' + str(rmse)

pd.DataFrame(predicted).to_csv("data/predicted.csv")

plt.plot(predicted[1:1000])
plt.savefig('data/out.png')
