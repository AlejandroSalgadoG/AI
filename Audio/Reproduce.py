import alsaaudio

filename = "output.dat"
device = "default:CARD=PCH"

f = open(filename, 'rb')

out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, 
                    channels=1, rate=44100, format=alsaaudio.PCM_FORMAT_S16_LE,
                    periodsize=160, device=device)

data = f.read(320)
while data:
    out.write(data)
    data = f.read(320)
