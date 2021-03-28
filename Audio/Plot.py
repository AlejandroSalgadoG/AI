import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

filename = 'output.wav'

chunk = 1024
fs = 44032
seconds = 3
data_size = int(fs / chunk * seconds) * chunk

wf = wave.open(filename, 'rb')
data = wf.readframes( data_size )
signal = np.frombuffer(data, dtype=np.int16)
signal = signal[ int(fs/2): ] # ignore first half second

plt.plot(signal)
plt.show()
