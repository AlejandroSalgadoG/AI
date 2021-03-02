import time
import alsaaudio

filename = "output.dat"
device = "default:CARD=PCH"

f = open(filename, 'wb')

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, 
	                channels=1, rate=44100, format=alsaaudio.PCM_FORMAT_S16_LE, 
	                periodsize=160, device=device)

for i in range( 1000000 ):
	l, data = inp.read()
  
	if l:
		f.write(data)
		time.sleep(0.001)
