#!/bin/bash

# Akhi Builder CLI Pipeline
# 1. Download, 2. Transcribe, 3. Clean & format

mkdir -p output/clips
mkdir -p output/transcripts
mkdir -p output/json

yt-dlp -a video_links.txt --extract-audio --audio-format mp3 -o "output/clips/%(title)s.%(ext)s"

for f in output/clips/*.mp3; do
  faster-whisper "$f" --output_format txt --output_dir output/transcripts
done

python3 scripts/make_quran_lora_json.py output/transcripts > output/json/akhi_lora.json
