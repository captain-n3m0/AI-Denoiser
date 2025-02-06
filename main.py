import torch
import numpy as np
from models.denoiser import load_model
from dsp.noise_filter import remove_noise, butter_lowpass_filter
from utils.audio_io import AudioHandler
from gui import NoiseFilterGUI
import tkinter as tk

def process_audio(model, audio_handler):
    while True:
        frame = audio_handler.get_audio_frame()
        frame = frame.astype(np.float32) / 32768.0  # Normalize

        # Apply DSP filtering
        filtered_frame = butter_lowpass_filter(frame)

        # Convert to tensor for AI inference
        tensor_frame = torch.tensor(filtered_frame).unsqueeze(0).unsqueeze(0)

        # AI-Based Denoising
        with torch.no_grad():
            denoised_frame = model(tensor_frame).squeeze().numpy()

        # Convert back to 16-bit PCM
        denoised_frame = (denoised_frame * 32768.0).astype(np.int16)

        # Play denoised audio
        audio_handler.play_audio_frame(denoised_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseFilterGUI(root)
    root.mainloop()


    try:
        process_audio(model, audio_handler)
    except KeyboardInterrupt:
        audio_handler.close()
        print("Audio processing stopped.")
