# TASK.md

## Akhi Data Builder — Developer Task List

This system builds a clean Islamic dataset from YouTube for LLM training.

---

## 🔧 Setup Requirements

- WSL or Ubuntu
- Python 3.10+
- Install: `yt-dlp`, `ffmpeg`, `faster-whisper`, `langdetect`, `uvicorn`, `fastapi`, `tauri` (optional)

```bash
pip install faster-whisper langdetect fastapi uvicorn yt-dlp
```

---

## 🧱 Folder Structure

```
backend/        → FastAPI backend
frontend_web/   → Flutter or React frontend
desktop_app/    → Tauri or Flutter desktop UI
pipeline/       → CLI pipeline: yt-dlp + whisper + json
```

---

## 🚀 Run Pipeline Locally

1. Add YouTube URLs to `pipeline/video_links.txt`
2. Run:

```bash
cd pipeline
bash run_pipeline.sh
```

3. Output:
- MP3s in `output/clips`
- Transcripts in `output/transcripts`
- LoRA JSON in `output/json/akhi_lora.json`

---

## 🔗 Next Phases

- [ ] Add Web UI to review transcripts
- [ ] Add Backend endpoints to trigger downloads/transcribes
- [ ] Send `akhi_lora.json` to Axolotl LoRA trainer
- [ ] Add speaker/topic classifier module (future)

---

### ✅ Goal

Build a clean, repeatable dataset pipeline to fine-tune Islamic LLMs (like Akhi) using YouTube lectures.

