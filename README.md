# 🎧 AI-Enhanced Noise Filtering Tool 🔥

A **real-time noise filtering tool** that removes background noise from audio using **AI and DSP techniques**. The application provides a **GUI with waveform and spectrum visualization**, supports **adjustable DSP parameters**, and allows **recording and saving filtered audio**.

---

## 🚀 Features
✅ **Real-Time Noise Reduction** – Uses a **deep learning model** for denoising.  
✅ **Adjustable DSP Parameters** – Modify **cutoff frequency** and **AI denoising strength**.  
✅ **Waveform Visualization** – See real-time changes in the audio signal.  
✅ **Equalizer (FFT Spectrum Analysis)** – Monitor frequency spectrum changes.  
✅ **Start/Stop Filtering** – Control real-time processing with buttons.  
✅ **Record and Save Audio** – Save denoised audio as `.wav` files.  
✅ **Dark-Themed GUI** – Sleek and modern UI for easy usability.  

---


---

## 🛠️ Installation

### **1️⃣ Install Dependencies**
Make sure you have Python 3.8+ installed. Then, install dependencies using:
```bash
pip install -r requirements.txt
```
### **2️⃣ Run the Application**
```bash
python main.py
```

## 🎛️ How to Use
1. **Start the program** by running:
   ```bash
   python main.py
   ```
   
2. **Adjust DSP Parameters:**
**Cutoff Frequency: Controls how much noise is removed.**
**AI Noise Reduction Strength: Adjust the impact of AI denoising.**
**Press "Start Filtering" to begin real-time noise filtering.**

3. **Visualize the Audio using:**
**Waveform Graph (Top) – Shows the audio signal in real time.**
**Equalizer (FFT Spectrum) (Bottom) – Displays frequency content.**
**Press "Start Recording" to begin recording filtered audio.**
**Press "Save Recording" to save it as a .wav file.**
**Press "Stop Filtering" to end processing.**

## 🔧 Customization
**Use Your Own AI Model – Replace models/pretrained_model.pth with your trained model.**
**Adjust DSP Algorithms – Modify dsp/noise_filter.py to change noise removal techniques.**
**Enhance Visualization – Modify gui.py to add more UI elements.**

