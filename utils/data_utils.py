import os
import pandas as pd  
import numpy as np
import scipy.io.wavfile as wav
import scipy

def _stft(self, x, fs, framesz, hop):
	framesamp = int(framesz*fs)
	hopsamp = int(hop*fs)
	w = scipy.hanning(framesamp)
	X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
					 for i in range(0, len(x)-framesamp, hopsamp)])
	print X
	fft_blocks = []
	for fft_block in X:
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks

def _istft(self, X, fs, T, hop):
	x = scipy.zeros(T*fs)
	framesamp = X.shape[1]
	hopsamp = int(hop*fs)
	for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
		x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
		return x

def _normalize(audio):
	audio *= 1000
	return audio / np.linalg.norm(audio)

def load_data(sample_number='sine', source_number=1, domain='time'):

	fs1, source = wav.read(os.path.join('sound_files', 'input', str(sample_number) ,str(source_number) + '.wav'))
	fs, mixed = wav.read(os.path.join('sound_files', 'mixed', str(sample_number) ,'1.wav'))
	mixed = _normalize(mixed) 
	source = _normalize(source)

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

	
