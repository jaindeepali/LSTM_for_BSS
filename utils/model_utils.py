from keras.models import Sequential  
from keras.layers.core import TimeDistributedDense, Activation, Dropout, Dense 
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

def create_network(in_neurons, out_neurons):

	print "Creating Network..."

	model = Sequential()  
	model.add(LSTM(output_dim=800, input_dim=in_neurons))
	model.add(Dropout(0.2))
	model.add(Dense(output_dim=out_neurons, input_dim=800))
	model.add(Activation("linear"))
	model.compile(loss="mean_squared_error", optimizer='rmsprop')  

	print "Network Created..."
	
	return model
	