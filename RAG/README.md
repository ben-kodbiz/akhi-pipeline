# Islamic RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that combines a QLoRA-fine-tuned Qwen3-1.7B model with Islamic knowledge sources to provide accurate, contextually relevant answers about Islamic topics.

## Overview

This project implements a complete RAG system that:
- Uses a QLoRA-fine-tuned Qwen3-1.7B model for text generation
- Integrates Islamic knowledge sources (Quran, Hadith, Islamic Q&A)
- Employs Chroma vector database with sentence-transformers embeddings
- Provides FastAPI-based REST API for easy integration
- Includes comprehensive evaluation using RAGAS metrics

## Features

- **QLoRA Integration**: Efficient 4-bit quantized model loading
- **Vector Search**: Semantic search using Chroma and sentence-transformers
- **Multi-Format Support**: Process HTML, PDF, DOCX, TXT, and JSON files
- **Reranking**: Optional reranking for improved retrieval quality
- **Conversation Memory**: Session-based conversation tracking
- **Evaluation**: RAGAS-based evaluation with multiple metrics
- **API Server**: FastAPI-based REST API with async support
- **Caching**: Response caching for improved performance

## Project Structure

```
RAG/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration settings
├── scripts/
│   ├── setup.py             # Setup and installation script
│   ├── data_preparation.py  # Data collection and preprocessing
│   ├── rag_pipeline.py      # Main RAG pipeline implementation
│   ├── api_server.py        # FastAPI server
│   └── evaluation.py        # RAGAS evaluation
├── data/                    # Islamic knowledge sources
│   ├── quran/
│   ├── hadith/
│   └── qa/
├── checkpoints/             # QLoRA model files
├── rag_index/              # Chroma vector database
├── evaluation_results/     # Evaluation reports
└── logs/                   # Application logs
```

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, 4GB+ VRAM)
- QLoRA-fine-tuned Qwen3-1.7B model
- 8GB+ RAM
- 10GB+ disk space

## Quick Start

### 1. Setup Environment

```bash
# Navigate to RAG directory
cd RAG

# Run setup script (installs dependencies and prepares data)
python scripts/setup.py
```

### 2. Configure the Pipeline

Edit `config.yaml` to match your setup:

```yaml
model:
  path: "../checkpoints/qwen3-1.7b-qlora"  # Path to your QLoRA model
  device: "cuda"  # or "cpu"
  load_in_4bit: true

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"

vector_store:
  persist_directory: "./rag_index"
  collection_name: "islamic_knowledge"

data:
  sources:
    - "./data/quran"
    - "./data/hadith"
    - "./data/qa"
```

### 3. Prepare Data

```bash
# Generate sample Islamic knowledge data
python scripts/data_preparation.py
```

### 4. Test the Pipeline

```bash
# Test RAG pipeline functionality
python scripts/rag_pipeline.py
```

### 5. Start API Server

```bash
# Start FastAPI server
python scripts/api_server.py

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### 6. Evaluate Performance

```bash
# Run RAGAS evaluation
python scripts/evaluation.py
```

## Supported File Formats

The RAG pipeline supports multiple document formats:

| Format | Extension | Requirements | Description |
|--------|-----------|--------------|-------------|
| **Text** | `.txt` | Built-in | Plain text files |
| **HTML** | `.html`, `.htm` | `beautifulsoup4` | Web pages and HTML documents |
| **PDF** | `.pdf` | `PyPDF2` | PDF documents |
| **Word** | `.docx`, `.doc` | `python-docx` | Microsoft Word documents |
| **JSON** | `.json` | Built-in | Structured JSON data |

### Installing Document Processing Dependencies

```bash
# Install all document processing dependencies
pip install beautifulsoup4 PyPDF2 python-docx unstructured

# Or install selectively based on your needs
pip install beautifulsoup4  # For HTML files
pip install PyPDF2          # For PDF files
pip install python-docx     # For Word documents
```

### Document Loading Examples

```python
from scripts.document_loader import EnhancedDocumentLoader
from pathlib import Path

# Initialize document loader
loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)

# Load a single document
documents = loader.load_document(Path("quran.pdf"))
print(f"Loaded {len(documents)} chunks")

# Load all documents from a directory
documents = loader.load_documents_from_directory(Path("islamic_texts/"))

# Load specific files
file_paths = ["quran.pdf", "hadith.docx", "tafsir.html"]
documents = loader.load_documents_from_paths(file_paths)

# Check supported formats
formats = loader.get_supported_formats()
for fmt, available in formats.items():
    status = "✓" if available else "✗"
    print(f".{fmt}: {status}")
```

## Usage Examples

### Python API

```python
from scripts.rag_pipeline import IslamicRAGPipeline

# Initialize pipeline
rag = IslamicRAGPipeline("config.yaml")
rag.setup()

# Query the system
response = rag.query(
    "What are the five pillars of Islam?",
    session_id="user123"
)

print(f"Answer: {response['answer']}")
print(f"Sources: {response['sources']}")
```

### REST API

```bash
# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the five pillars of Islam?",
    "session_id": "user123",
    "max_tokens": 512
  }'

# Conversation endpoint
curl -X POST "http://localhost:8000/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about prayer in Islam",
    "session_id": "user123"
  }'
```

### Response Format

```json
{
  "answer": "The five pillars of Islam are...",
  "sources": [
    {
      "content": "Relevant text from source",
      "metadata": {
        "source": "quran",
        "chapter": "2",
        "verse": "177"
      },
      "score": 0.85
    }
  ],
  "session_id": "user123",
  "response_time": 1.23,
  "model_used": "qwen3-1.7b-qlora"
}
```

## Configuration

### Model Settings

```yaml
model:
  path: "path/to/your/qlora/model"  # QLoRA model path
  device: "cuda"                    # Device: cuda/cpu
  load_in_4bit: true               # Enable 4-bit quantization
  max_new_tokens: 512              # Maximum generation length
  temperature: 0.7                 # Generation temperature
  do_sample: true                  # Enable sampling
  top_p: 0.9                      # Top-p sampling
  repetition_penalty: 1.1          # Repetition penalty
```

### Retrieval Settings

```yaml
retrieval:
  top_k: 5                        # Number of documents to retrieve
  score_threshold: 0.7            # Minimum similarity score
  chunk_size: 512                 # Document chunk size
  chunk_overlap: 50               # Overlap between chunks
  enable_reranking: true          # Enable reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### API Settings

```yaml
api:
  host: "0.0.0.0"                 # Server host
  port: 8000                      # Server port
  max_concurrent_requests: 10     # Max concurrent requests
  request_timeout: 30             # Request timeout (seconds)
  enable_cors: true               # Enable CORS
  log_level: "INFO"               # Logging level
```

## Evaluation

The system includes comprehensive evaluation using RAGAS metrics:

- **Context Relevancy**: How relevant retrieved contexts are
- **Answer Relevancy**: How relevant the answer is to the question
- **Faithfulness**: How faithful the answer is to the context
- **Context Recall**: How well the retrieval covers relevant information
- **Context Precision**: How precise the retrieved contexts are

### Running Evaluation

```bash
# Run full evaluation
python scripts/evaluation.py

# Run with custom dataset
python scripts/evaluation.py --dataset path/to/your/dataset.json

# Generate detailed report
python scripts/evaluation.py --detailed-report
```

### Evaluation Results

Results are saved in `evaluation_results/` with:
- Overall metrics scores
- Per-question analysis
- Performance benchmarks
- Recommendations for improvement

## Optimization Tips

### Performance

1. **GPU Usage**: Ensure CUDA is available for both model and embeddings
2. **Batch Processing**: Process multiple queries in batches
3. **Caching**: Enable response caching for repeated queries
4. **Model Quantization**: Use 4-bit quantization for memory efficiency

### Quality

1. **Chunk Size**: Experiment with different chunk sizes (256-1024)
2. **Reranking**: Enable reranking for better retrieval quality
3. **Prompt Engineering**: Customize prompts for your use case
4. **Data Quality**: Ensure high-quality, relevant training data

### Scalability

1. **Vector Database**: Consider using persistent Chroma storage
2. **Load Balancing**: Use multiple API instances for high traffic
3. **Async Processing**: Leverage FastAPI's async capabilities
4. **Resource Monitoring**: Monitor GPU/CPU usage and memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable 4-bit quantization
   - Use CPU for embeddings

2. **Model Loading Errors**
   - Check model path in config.yaml
   - Ensure model files are complete
   - Verify model format compatibility

3. **Slow Performance**
   - Enable GPU acceleration
   - Reduce chunk size
   - Enable caching

4. **Poor Answer Quality**
   - Improve data quality
   - Enable reranking
   - Adjust retrieval parameters
   - Fine-tune prompt templates

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/rag_pipeline.py

# Test individual components
python -c "from scripts.rag_pipeline import *; test_embeddings()"
python -c "from scripts.rag_pipeline import *; test_model_loading()"
```

## Development

### Adding New Data Sources

1. Create data processing function in `data_preparation.py`
2. Add source path to `config.yaml`
3. Update document processor in `rag_pipeline.py`
4. Test with new data

### Customizing Prompts

Edit the prompt template in `rag_pipeline.py`:

```python
CHATML_TEMPLATE = """
<|im_start|>system
You are a knowledgeable Islamic scholar assistant...
<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
```

### Adding New Evaluation Metrics

1. Implement metric function in `evaluation.py`
2. Add to evaluation pipeline
3. Update reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration
3. Check logs in `logs/` directory
4. Create an issue with detailed information

## Acknowledgments

- Qwen team for the base model
- Hugging Face for transformers and datasets
- LangChain for RAG framework
- Chroma for vector database
- RAGAS for evaluation metrics