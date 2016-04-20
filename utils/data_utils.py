import os
import pandas as pd  
import numpy as np
import scipy.io.wavfile as wav
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _stft(x, fs, framesz = 0.005, hop = 0.001):
	framesamp = int(framesz*fs)
	hopsamp = int(hop*fs)
	X = scipy.array([scipy.fft(x[i:i+framesamp]) 
					 for i in range(0, len(x)-framesamp, hopsamp)])
	fft_blocks = []
	for fft_block in X:
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks

def _istft(X):
	time_blocks = []
	for block in X:
		num_elems = block.shape[0] / 2
		real_chunk = block[0:num_elems]
		imag_chunk = block[num_elems:]
		new_block = real_chunk + 1.0j * imag_chunk
		time_block = scipy.ifft(new_block)
		time_blocks.append(time_block)
	return np.concatenate(time_blocks)

def _crop(audio, n):
	return audio[:n]
	
def _normalize(audio):
	audio *= 1000
	return audio / np.linalg.norm(audio)

def load_data(sample_number='sinelong', source_number=1, domain='time'):

	fs1, source = wav.read(os.path.join('sound_files', 'input', str(sample_number) ,str(source_number) + '.wav'))
	fs, mixed = wav.read(os.path.join('sound_files', 'mixed', str(sample_number) ,'1.wav'))
	mixed = _crop(mixed, 200000) 
	source = _crop(source, 200000)
	
	if domain == 'freq':
		source = _stft(source, Fs=fs1)
		mixed = _stft(mixed, Fs=fs)

	mixed = pd.DataFrame(mixed)
	source = pd.DataFrame(source)

	docX, docY = [], []
	for i in range(len(mixed)-1):
		docX.append(mixed.iloc[i:i+1].as_matrix())
		docY.append(source.iloc[i+1].as_matrix())
	
	X_Train = np.array(docX)
	Y_Train = np.array(docY)
	return X_Train, Y_Train

	
def save_output(predicted, sample_number='sinelong', domain='time'):

	outputfile = "data/output/" + str(sample_number) + "/predicted.csv"
	plotfile = "data/plots/" + str(sample_number) + "/out.png"
	os.makedirs(os.path.dirname(outputfile), exist_ok=True)
	os.makedirs(os.path.dirname(plotfile), exist_ok=True)
	
	if domain == 'freq':
		predicted = _istft(predicted)
	
	pd.DataFrame(predicted).to_csv(outputfile)
	plt.clf()
	plt.plot(predicted[1:1000])
	plt.savefig()
