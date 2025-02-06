import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioHandler:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

    def get_audio_frame(self):
        data = self.stream.read(CHUNK)
        return np.frombuffer(data, dtype=np.int16)

    def play_audio_frame(self, frame):
        self.stream.write(frame.astype(np.int16).tobytes())

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
