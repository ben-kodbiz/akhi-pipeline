Alright, Ben — here's your detailed `task.md` outline for integrating the [Akhi YouTube Search Pipeline](https://github.com/ben-kodbiz/akhi-pipeline/tree/search_feature) into a CrewAI agentic system.

---

## `task.md` — CrewAI Agentic Integration Plan for Akhi YouTube Search

### 🧠 Objective

Transform the Akhi YouTube Search Pipeline into a fully autonomous CrewAI workflow where agents can:

1. Search YouTube.
2. Download & transcribe selected videos.
3. Chunk and embed transcripts.
4. Store/retrieve embeddings using FAISS.
5. Generate summaries or answer questions based on video content.

---

### 🔧 Required Dependencies

You **still need**:

* `yt-dlp`: For downloading YouTube video/audio.
* `yt-search` (or a wrapper): For YouTube video search by keyword.
* `whisper` (or OpenAI Whisper API): For transcription.
* `faiss`: For semantic vector search.
* `llama_cpp`, `GGUF model`, or similar: For local inference & embedding generation.
* `CrewAI`, `crewai.tools.BaseTool`, and custom tools for actions.

---

### 📦 Agentic Task Breakdown

#### 🧩 Phase 1: Search + Selection

**Agent: VideoResearcherAgent**

* **Role**: YouTube Research Assistant
* **Goal**: Find relevant YouTube videos based on a given topic or query.
* **Tool(s)**: Custom `YouTubeSearchTool` using `yt-search`.
* **Task**: Accept topic → return top N results with title, URL, duration.

---

#### 🎞️ Phase 2: Download + Transcribe

**Agent: TranscriberAgent**

* **Role**: Audio-to-Text Transcriber
* **Goal**: Download audio + generate accurate transcript.
* **Tool(s)**:

  * `YouTubeDownloaderTool` → wraps `yt-dlp`.
  * `TranscriptionTool` → wraps Whisper or OpenAI API.
* **Task**: Take selected YouTube URL → output cleaned transcript.

---

#### 📚 Phase 3: Chunk + Embed + Store

**Agent: VectorIndexerAgent**

* **Role**: Embedding Engineer
* **Goal**: Break down transcripts, embed them, and store in FAISS.
* **Tool(s)**:

  * `TextChunkerTool` → sliding window chunking.
  * `EmbedderTool` → use `llama_cpp` + GGUF model for embedding.
  * `FAISSStorageTool` → add embeddings to FAISS index.
* **Task**: Accept transcript → chunk → embed → store.

---

#### ❓ Phase 4: QA or Summarize

**Agent: ContentQAAgent**

* **Role**: Semantic Retriever + Summarizer
* **Goal**: Retrieve relevant transcript chunks and respond to user queries.
* **Tool(s)**:

  * `FAISSQueryTool` → search embeddings.
  * `SummarizerTool` → summarize retrieved context.
  * `AnswerGeneratorTool` → generate answers with LLM.
* **Task**: Accept user question → retrieve relevant content → respond.

---

### 📁 Suggested Directory Structure

```
akhi_crewai/
├── tools/
│   ├── youtube_search.py
│   ├── downloader.py
│   ├── transcriber.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── faiss_store.py
│   └── faiss_query.py
├── agents/
│   ├── researcher.py
│   ├── transcriber.py
│   ├── indexer.py
│   └── qa_agent.py
├── crew/
│   ├── akhi_pipeline.py
├── data/
│   ├── transcripts/
│   ├── embeddings/
├── outputs/
│   ├── results.md
│   └── results.pdf
└── task.md
```



