import pyaudio
import wave

filename = 'output.wav'

chunk = 1024  # Record in chunks of 1024 samples
fs = 44100  # Record at 44100 samples per second
seconds = 1
data_size = int(fs / chunk * seconds)
print( data_size )

# Open the sound file 
wf = wave.open(filename, 'rb')

# Read data in chunks
data = wf.readframes(44100)
print( len(data) )
