import wave
import pyaudio

chunk = 1024
sample_format = pyaudio.paInt16 
channels = 1
fs = 44032
seconds = 3
n_frames = int(fs / chunk * seconds)

filename = "output.wav"

p = pyaudio.PyAudio()
stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

frames = []

for i in range(n_frames):
    data = stream.read(chunk)
    frames.append(data)

full_data = b''.join(frames)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth( p.get_sample_size(sample_format) ) 
wf.setframerate(fs)
wf.writeframes(full_data)
wf.close()
