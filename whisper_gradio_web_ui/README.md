# 🗣️➡️📝 Whisper Transcription Web UI Agent — ROCm-Ready

A **lightweight Gradio-based web interface** for running the OpenAI **Whisper** transcription model on **AMD ROCm** systems.  

This project provides an **easy-to-use web UI Agent** for:

- Drag-and-drop transcription of audio files  
- Recording audio directly in the browser  
- Generating transcriptions and translations    
- Optionally sharing the interface online  

**Optimized for Radeon AI PRO R9700 GPUs** · Supports **bf16/fp16**

[![ROCm](https://img.shields.io/badge/AMD-ROCm_6.x-red)](https://rocmdocs.amd.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0.dev%2Brocm6.4-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Gradio Quickstart](https://img.shields.io/badge/Gradio-Quickstart-blue)](https://www.gradio.app/guides/quickstart)
[![Whisper + ROCm](https://img.shields.io/badge/Whisper-ROCm_6.x-blue)](https://github.com/openai/whisper)

<img width="1265" height="743" alt="image" src="https://github.com/user-attachments/assets/d1d24c7f-0e5c-4266-a6ad-dde8a5858fe6" />

## 🚀 Installation

### 1️⃣ **System preperation**
1. Install **Ubuntu 22.04.5 LTS** or **Ubuntu 24.04.3 LTS** (Server or Desktop version).
2. Install **🧩 ROCm 6.4.2** + OpenCL 2.x + **PyTorch 2.8.0** (Stable) + Transformers + Docker Setup.
3. For detailed setup instructions, visit the [ROCm 6.4.2 PyTorch 2.8.0 RDNA4 Docker Automated Deployment Repository](https://github.com/JoergR75/rocm-6.4.2-pytorch-2.8.0-rdna4-docker-automated-deployment/tree/main).
4. Install **Gradio** (API + Web Interface).  
Choose your Ubuntu version below to view the correct installation steps.
    <details>
    <summary>🟢 Ubuntu 22.04.x (Jammy Jellyfish)</summary>

    ```bash
    pip3 install gradio
    ```
    </details>

    <details>
    <summary>🔵 Ubuntu 24.04.x (Noble Numbat)</summary>
  
    ```bash
    pip3 install gradio --break-system-packages
    ```
    </details>

6. Install **ffmpeg**
```echo
sudo apt install -y ffmpeg
```

6. Install **Whisper** (OpenAI) stable release.  
Choose your Ubuntu version below to view the correct installation steps.
    <details>
    <summary>🟢 Ubuntu 22.04.x (Jammy Jellyfish)</summary>

    ```bash
    pip3 install -U openai-whisper
    ```
    </details>

    <details>
    <summary>🔵 Ubuntu 24.04.x (Noble Numbat)</summary>
  
    ```bash
    pip3 install -U openai-whisper --break-system-packages
    ```
    </details>

### 2️⃣ Download the server script

```echo
wget https://raw.githubusercontent.com/JoergR75/whisper_rocm_transcribe/refs/heads/main/whisper_gradio_web_ui/whisper_rocm_gradio_web_ui.py
```
<img width="1092" height="186" alt="image" src="https://github.com/user-attachments/assets/21e9407c-b5ad-484e-b06b-5c2a880ae092" />

### 3️⃣ Launch the Gradio web Agent server

```echo
python3 whisper_rocm_gradio_web_ui.py
```
The Web server will be launched with following parameters:
- http://127.0.0.1:7860 → The Gradio web interface is running locally on your machine. Open this link in your browser to access the app.
- share=True → If you set this option inside the script’s launch() method, Gradio will create a temporary public URL. This allows you to share access with others outside your local network.
> ⚠️ **Attention**  
> The first time you launch the script, it will download the model weights.  
> This process can take **1–5 minutes**, depending on your hardware and internet connection.

<img width="852" height="132" alt="image" src="https://github.com/user-attachments/assets/17feaac6-361b-4b33-a15e-1215dbc612da" />

### 4️⃣ Launch the Gradio web Agent from another device connected to same network

First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:127.0.0.1:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:127.0.0.1:7860 ai1@192.168.178.xxx
```
Now you can open **http://127.0.0.1:7860** in your local browser to access the Gradio Web Agent.
