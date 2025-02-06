import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.denoiser import load_model
from dsp.noise_filter import butter_lowpass_filter
from dsp.fft_processing import apply_fft
from utils.audio_io import AudioHandler

class NoiseFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Noise Filter")
        self.root.geometry("800x600")
        self.root.configure(bg="#1E1E1E")  # Dark background

        self.model = load_model()
        self.audio_handler = None
        self.is_running = False
        self.is_recording = False
        self.audio_buffer = np.zeros(1024)  # Buffer for visualization
        self.recorded_audio = []

        # Title Label
        self.label = tk.Label(root, text="AI-Enhanced Noise Filtering", font=("Arial", 14, "bold"), fg="white", bg="#1E1E1E")
        self.label.pack(pady=10)

        # Cutoff Frequency Slider
        self.cutoff_label = tk.Label(root, text="Cutoff Frequency (Hz):", fg="white", bg="#1E1E1E")
        self.cutoff_label.pack()
        self.cutoff_freq = tk.IntVar(value=3000)
        self.cutoff_slider = tk.Scale(root, from_=500, to=8000, orient=tk.HORIZONTAL, variable=self.cutoff_freq, bg="#252526", fg="white", highlightbackground="#1E1E1E")
        self.cutoff_slider.pack()

        # Noise Reduction Strength Slider
        self.noise_label = tk.Label(root, text="AI Noise Reduction Strength:", fg="white", bg="#1E1E1E")
        self.noise_label.pack()
        self.noise_strength = tk.DoubleVar(value=1.0)
        self.noise_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.noise_strength, bg="#252526", fg="white", highlightbackground="#1E1E1E")
        self.noise_slider.pack()

        # Start/Stop Buttons
        self.start_button = tk.Button(root, text="Start Filtering", command=self.start_processing, bg="green", fg="white")
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Filtering", command=self.stop_processing, bg="red", fg="white", state=tk.DISABLED)
        self.stop_button.pack()

        # Recording Buttons
        self.record_button = tk.Button(root, text="Start Recording", command=self.start_recording, bg="#0078D7", fg="white")
        self.record_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Recording", command=self.save_recording, bg="#A700D7", fg="white", state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Visualization Section
        self.setup_visualization()

    def setup_visualization(self):
        """ Setup Matplotlib for waveform and FFT visualization """
        self.fig, (self.ax_waveform, self.ax_spectrum) = plt.subplots(2, 1, figsize=(6, 4))
        self.fig.subplots_adjust(hspace=0.5)
        self.fig.patch.set_facecolor("#1E1E1E")  # Dark background

        # Waveform Plot
        self.ax_waveform.set_title("Waveform", color="white")
        self.ax_waveform.set_xlim(0, 1024)
        self.ax_waveform.set_ylim(-1, 1)
        self.ax_waveform.set_facecolor("#252526")
        self.ax_waveform.spines['bottom'].set_color("white")
        self.ax_waveform.spines['left'].set_color("white")
        self.waveform_line, = self.ax_waveform.plot(np.zeros(1024), color="cyan")

        # FFT Spectrum Plot
        self.ax_spectrum.set_title("Frequency Spectrum", color="white")
        self.ax_spectrum.set_xlim(0, 8000)
        self.ax_spectrum.set_ylim(0, 1)
        self.ax_spectrum.set_facecolor("#252526")
        self.ax_spectrum.spines['bottom'].set_color("white")
        self.ax_spectrum.spines['left'].set_color("white")
        self.spectrum_line, = self.ax_spectrum.plot(np.zeros(512), color="magenta")

        # Embed Matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Start Animation
        self.anim = animation.FuncAnimation(self.fig, self.update_visualization, interval=50)

    def update_visualization(self, frame):
        """ Update the waveform and spectrum plots """
        self.waveform_line.set_ydata(self.audio_buffer)
        fft_data = apply_fft(self.audio_buffer)[:512]
        self.spectrum_line.set_ydata(fft_data / np.max(fft_data) if np.max(fft_data) > 0 else fft_data)
        self.canvas.draw()

    def process_audio(self):
        """ Real-time audio processing loop """
        self.audio_handler = AudioHandler()
        self.is_running = True

        while self.is_running:
            frame = self.audio_handler.get_audio_frame()
            frame = frame.astype(np.float32) / 32768.0  # Normalize

            # Store waveform for visualization
            self.audio_buffer = frame.copy()

            # Apply DSP filtering
            filtered_frame = butter_lowpass_filter(frame, cutoff=self.cutoff_freq.get())

            # Convert to tensor for AI inference
            tensor_frame = torch.tensor(filtered_frame).unsqueeze(0).unsqueeze(0)

            # AI-Based Denoising
            with torch.no_grad():
                denoised_frame = self.model(tensor_frame).squeeze().numpy()

            # Adjust AI noise reduction strength
            denoised_frame *= self.noise_strength.get()

            # Convert back to 16-bit PCM
            denoised_frame = np.clip(denoised_frame, -1.0, 1.0)
            denoised_frame = (denoised_frame * 32768.0).astype(np.int16)

            # Record audio if recording is enabled
            if self.is_recording:
                self.recorded_audio.append(denoised_frame.copy())

            # Play denoised audio
            self.audio_handler.play_audio_frame(denoised_frame)

    def start_recording(self):
        """ Starts recording the filtered audio """
        self.is_recording = True
        self.recorded_audio = []
        self.record_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)

    def save_recording(self):
        """ Saves the recorded audio to a WAV file """
        self.is_recording = False
        self.record_button.config(state=tk.NORMAL)

        if self.recorded_audio:
            filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
            if filename:
                sf.write(filename, np.concatenate(self.recorded_audio), samplerate=16000)
                messagebox.showinfo("Success", f"Recording saved as {filename}")

    def start_processing(self):
        """ Starts real-time processing in a separate thread """
        if not self.is_running:
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

    def stop_processing(self):
        """ Stops real-time processing """
        self.is_running = False
        if self.audio_handler:
            self.audio_handler.close()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
