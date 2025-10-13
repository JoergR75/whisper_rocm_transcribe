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

# 🌀 Installation

### Download the the server script

```echo
wget https://raw.githubusercontent.com/JoergR75/whisper_rocm_transcribe/refs/heads/main/whisper_gradio_web_ui/whisper_rocm_gradio_web_ui.py
```
<img width="1092" height="186" alt="image" src="https://github.com/user-attachments/assets/21e9407c-b5ad-484e-b06b-5c2a880ae092" />

### Launch the Gradio web Agent server

```echo
python3 FLUX-gradio-web-agent.py
```
The Web server will be launched with following parameters:
- http://127.0.0.1:7860 → The Gradio web interface is running locally on your machine. Open this link in your browser to access the app.
- share=True → If you set this option inside the script’s launch() method, Gradio will create a temporary public URL. This allows you to share access with others outside your local network.
> ⚠️ **Attention**  
> The first time you launch the script, it will download the model weights.  
> This process can take **15–20 minutes**, depending on your hardware and internet connection.
<img width="516" height="69" alt="{297F19AF-995D-4A17-BBA1-B49F0CB68F36}" src="https://github.com/user-attachments/assets/c8b90dfb-954e-4306-9352-2afdbcf094f9" />

### Launch the Gradio web Agent from another device connected to same network

First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:127.0.0.1:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:127.0.0.1:7860 ai1@192.168.178.xxx
```
Now you can open **http://127.0.0.1:7860** in your local browser to access the Gradio Web Agent.
