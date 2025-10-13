#!/usr/bin/env python3
"""
whisper_rocm_transcribe.py
Transcribes or translates audio files using OpenAI Whisper with ROCm (AMD GPU) acceleration when available.
Automatically selects GPU or CPU, checks for ffmpeg, and saves results as .txt, .vtt, or .srt subtitle files.

Additional requirements:
  pip install git+https://github.com/openai/whisper.git
  sudo apt install ffmpeg

Usage example:
  python whisper_rocm_transcribe.py --input audio.mp3 --model small --out out_prefix --language de --format txt

author:
  Joerg Roskowetz
"""

import argparse
import os
import sys
import whisper
import torch
from datetime import datetime

def choose_device():
    # Bei ROCm ist torch.cuda.* normalerweise die API (PyTorch/ROCm mappt auf HIP).
    if torch.cuda.is_available():
        device = "cuda"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "AMD GPU (ROCm)"
        print(f"[INFO] GPU verfügbar: {name} (device='{device}')")
    else:
        device = "cpu"
        print("[INFO] Keine GPU entdeckt — benutze CPU (langsamer).")
    return device

def ensure_ffmpeg():
    # whisper will use ffmpeg; verify if available
    from shutil import which
    if which("ffmpeg") is None:
        print("[WARN] ffmpeg nicht gefunden. Installiere ffmpeg (apt/yum/pacman) bevor du große Audios transkribierst.")
    else:
        print("[INFO] ffmpeg gefunden.")

def transcribe_file(input_path, model_name="small", device="cuda", language=None, task="transcribe", fp16=True):
    print(f"[INFO] Lade Whisper-Modell '{model_name}'...")
    model = whisper.load_model(model_name)
    model.to(device)

    # Wenn GPU aber kein fp16 unterstützt oder Modell klein ist, kann fp16 abgeschaltet werden.
    if device == "cpu":
        fp16 = False

    print(f"[INFO] Transkribiere '{input_path}' auf device='{device}', fp16={fp16} ...")
    # whisper.transcribe wrapper
    result = model.transcribe(input_path, language=language, task=task, fp16=fp16)
    return model, result

def save_outputs(result, out_prefix, out_formats=("txt",)):
    text = result.get("text", "").strip()
    segments = result.get("segments", None)

    if "txt" in out_formats:
        txt_path = f"{out_prefix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[OK] TXT gespeichert: {txt_path}")

    if "vtt" in out_formats and segments:
        vtt_path = f"{out_prefix}.vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(segments, start=1):
                start = seg["start"]
                end = seg["end"]
                # WebVTT Zeitformat
                def fmt(t):
                    h = int(t // 3600)
                    m = int((t % 3600) // 60)
                    s = t % 60
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
                f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{seg['text'].strip()}\n\n")
        print(f"[OK] VTT gespeichert: {vtt_path}")

    if "srt" in out_formats and segments:
        srt_path = f"{out_prefix}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = seg["start"]
                end = seg["end"]
                def fmt_srt(t):
                    h = int(t // 3600)
                    m = int((t % 3600) // 60)
                    s = int(t % 60)
                    ms = int((t - int(t)) * 1000)
                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                f.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{seg['text'].strip()}\n\n")
        print(f"[OK] SRT gespeichert: {srt_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper on ROCm (AMD).")
    p.add_argument("--input", "-i", required=True, help="Eingabedatei (mp3/wav/m4a/...)")
    p.add_argument("--model", "-m", default="small", help="Whisper model (tiny, base, small, medium, large).")
    p.add_argument("--out", "-o", default=None, help="Output prefix (ohne extension). Default: inputname_YYYYMMDD_HHMM")
    p.add_argument("--language", "-l", default=None, help="Sprache (z.B. 'de' für Deutsch). Wenn None => automatische Erkennung.")
    p.add_argument("--format", "-f", default="txt", help="Ausgabeformat(e), kommasepariert: txt,vtt,srt")
    p.add_argument("--task", default="transcribe", choices=("transcribe","translate"), help="transcribe oder translate")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] Eingabedatei nicht gefunden: {args.input}")
        sys.exit(2)

    ensure_ffmpeg()
    device = choose_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_prefix = args.out
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_prefix = f"{base}_{timestamp}"

    model, result = transcribe_file(args.input, model_name=args.model, device=device, language=args.language, task=args.task)

    out_formats = [x.strip().lower() for x in args.format.split(",") if x.strip()]
    save_outputs(result, out_prefix, out_formats)

if __name__ == "__main__":
    main()
