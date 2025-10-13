#!/usr/bin/env python3
"""
whisper_web_ui.py
Web UI for Whisper ROCm transcription using Gradio.
Automatically selects GPU/CPU, supports multiple output formats (txt, vtt, srt).

Requirements:
  pip3 install git+https://github.com/openai/whisper.git gradio torch
  sudo apt install ffmpeg

Author:
  Joerg Roskowetz
"""

import os
import torch
import whisper
import gradio as gr
from datetime import datetime
from shutil import which

def choose_device():
    if torch.cuda.is_available():
        device = "cuda"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "AMD GPU (ROCm)"
        print(f"[INFO] GPU available: {name} (device='{device}')")
    else:
        device = "cpu"
        print("[INFO] no GPU available — CPU will be used (slower).")
    return device

def ensure_ffmpeg():
    if which("ffmpeg") is None:
        return "[WARN] ffmpeg not found. Please install it before large audio transcriptions."
    return "[INFO] ffmpeg found."

def transcribe_file(input_path, model_name="small", device="cuda", language=None, task="transcribe", fp16=True):
    model = whisper.load_model(model_name)
    model.to(device)
    if device == "cpu":
        fp16 = False
    result = model.transcribe(input_path, language=language, task=task, fp16=fp16)
    return result

def format_vtt_srt(result, out_prefix):
    segments = result.get("segments", [])
    vtt_path = f"{out_prefix}.vtt"
    srt_path = f"{out_prefix}.srt"

    # VTT
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, start=1):
            start, end = seg["start"], seg["end"]
            def fmt(t): 
                h, m = int(t // 3600), int((t % 3600) // 60)
                s = t % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{seg['text'].strip()}\n\n")

    # SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start, end = seg["start"], seg["end"]
            def fmt_srt(t):
                h, m = int(t // 3600), int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            f.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{seg['text'].strip()}\n\n")

    return vtt_path, srt_path

def whisper_web_ui(audio_file, model_name, language, task, out_format):
    if audio_file is None:
        return "No audio uploaded.", None, None

    ffmpeg_status = ensure_ffmpeg()
    device = choose_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = f"transcript_{timestamp}"

    result = transcribe_file(audio_file, model_name=model_name, device=device, language=language, task=task)
    text_output = result.get("text", "").strip()

    vtt_path, srt_path = None, None
    formats = [x.strip().lower() for x in out_format.split(",")]
    if "vtt" in formats or "srt" in formats:
        vtt_path, srt_path = format_vtt_srt(result, out_prefix)

    files = []
    if "txt" in formats:
        txt_path = f"{out_prefix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_output + "\n")
        files.append(txt_path)
    if "vtt" in formats:
        files.append(vtt_path)
    if "srt" in formats:
        files.append(srt_path)

    return text_output, ffmpeg_status, files

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🗣️➡️📝 Whisper ROCm Transcription We UI Agent [![ROCm](https://img.shields.io/badge/AMD-ROCm_6.x-red)](https://rocmdocs.amd.com/) [![Whisper + ROCm](https://img.shields.io/badge/Whisper-ROCm_6.x-blue)](https://github.com/openai/whisper)" )
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload audio file", type="filepath")
            model_name = gr.Dropdown(["tiny","base","small","medium","large"], value="small", label="Whisper model")
            language = gr.Textbox(value="", label="Language (leave blank for auto)")
            task = gr.Radio(["transcribe","translate"], value="transcribe", label="Task")
            out_format = gr.Textbox(value="txt", label="Output format (txt,vtt,srt comma-separated)")
            submit_btn = gr.Button("Transcribe")
        with gr.Column():
            #text_output = gr.Textbox(label="Transcription", interactive=False)
            text_output = gr.Textbox(label="Transcription", interactive=False, lines=20, max_lines=40)
            ffmpeg_status_output = gr.Textbox(label="FFmpeg status", interactive=False)
            file_output = gr.File(label="Download output files", file_types=[".txt",".vtt",".srt"])

    submit_btn.click(whisper_web_ui,
                     inputs=[audio_input, model_name, language, task, out_format],
                     outputs=[text_output, ffmpeg_status_output, file_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
