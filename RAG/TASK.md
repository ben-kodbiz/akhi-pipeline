Model must be offline
RAG Integration Task for QLoRA-Fine-Tuned Qwen3-1.7B
Objective
Integrate a Retrieval-Augmented Generation (RAG) pipeline with a QLoRA-fine-tuned Qwen3-1.7B model (trained with Axolotl) to enhance its ability to answer basic Islamic questions accurately. The pipeline should use open-source, free tools to retrieve relevant Islamic knowledge (e.g., Quran, Hadith, Q&A datasets) and generate contextually grounded responses, mitigating context rot in long conversations.
Background

Model: Qwen3-1.7B, fine-tuned with QLoRA (4-bit quantization, lora_rank=16, lora_alpha=32, lora_dropout=0.1) using Axolotl.
Training Status: Current metrics: loss: 2.5089, grad_norm: 0.57, learning_rate: 0.00018, epoch: 0.75.
Task Context: Enhance model quality for basic Islamic questions (e.g., Quran facts, five pillars) by integrating RAG to retrieve external knowledge, reducing hallucinations and context rot.
Constraints: Use only open-source, free tools (e.g., LangChain, Chroma, Sentence Transformers, Hugging Face Transformers, Unsloth, RAGAS, FastAPI).

Task Steps
1. Set Up Environment and Dependencies

Objective: Install necessary Python libraries for RAG and QLoRA inference.
Tasks:
Install dependencies:pip install langchain chromadb sentence-transformers transformers torch accelerate
pip install unsloth  # Optional, for faster QLoRA inference
pip install ragas fastapi uvicorn  # For evaluation and deployment


Verify GPU availability for inference and embedding computation.
Create project directory structure:/project
├── data/                    # Islamic texts (Quran, Hadith, Q&A)
├── checkpoints/             # QLoRA model weights
├── rag_index/              # Chroma vector store
├── scripts/                # Python scripts for RAG pipeline
└── config.yaml             # Axolotl config





2. Collect and Preprocess Islamic Knowledge Sources

Objective: Gather and prepare a dataset of Islamic texts for the RAG knowledge base.
Tasks:
Collect open-source texts:
Quran (e.g., Sahih International translation).
Hadith (e.g., Sahih al-Bukhari, Sahih Muslim).
Q&A datasets (e.g., “100 Questions on the Holy Quran,” “300 Islamic General Knowledge MCQs”).


Preprocess data:
Clean text (remove metadata, special characters).
Split into chunks (100-200 tokens) using LangChain’s RecursiveCharacterTextSplitter:from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("data/quran.txt", "r") as f:
    text = f.read()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(text)


Convert chunks to Document objects with metadata:from langchain.docstore.document import Document

documents = [Document(page_content=chunk, metadata={"source": "quran", "chapter": i}) for i, chunk in enumerate(chunks)]




Ensure 1,000-3,000 samples covering Quran, Hadith, and Islamic practices.



3. Build RAG Index with Chroma

Objective: Create a vector database to store and retrieve document embeddings.
Tasks:
Generate embeddings using sentence-transformers/all-MiniLM-L6-v2:from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


Index documents in Chroma:from langchain.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./rag_index"
)
vector_store.persist()





4. Load QLoRA-Fine-Tuned Qwen3-1.7B

Objective: Load the Axolotl-fine-tuned model for RAG generation.
Tasks:
Load model with Hugging Face Transformers in 4-bit quantization:from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "./checkpoints/qwen3-1.7b-qlora"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


(Optional) Use Unsloth for faster inference:from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./checkpoints/qwen3-1.7b-qlora",
    max_seq_length=32768,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)





5. Build RAG Pipeline with LangChain

Objective: Create a pipeline to retrieve documents and generate responses.
Tasks:
Set up Chroma retriever (top-k=5):retriever = vector_store.as_retriever(search_kwargs={"k": 5})


Define ChatML-compatible prompt template:from langchain.prompts import PromptTemplate

template = """<|im_start|>system
You are a knowledgeable assistant on Islamic topics. Use the following context to answer the question accurately. If the context doesn't provide enough information, rely on your fine-tuned knowledge but avoid speculation.

Context: {context}

Question: {question}
<|im_end|> <|im_start|>assistant
{answer} <|im_end|>"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)


Create RAG chain:from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    device_map="auto"
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)


Mitigate context rot by limiting retrieved context to 1,000-2,000 tokens.



6. Test and Evaluate RAG Pipeline

Objective: Validate performance on Islamic questions.
Tasks:
Test sample queries:query = "How many chapters are in the Quran?"
result = rag_chain.run(query)
print(result)  # Expected: "114"


Evaluate with RAGAS on a validation set (100-300 Q&A pairs):from ragas import evaluate
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": ["How many chapters are in the Quran?", "What are the five pillars of Islam?"],
    "answer": [rag_chain.run(q) for q in ["How many chapters are in the Quran?", "What are the five pillars of Islam?"]],
    "ground_truth": ["114", "Shahada, Salah, Zakat, Sawm, Hajj"]
})
result = evaluate(eval_dataset, metrics=["context_relevance", "answer_faithfulness"])
print(result)


Aim for 80-90% accuracy and minimal hallucinations.



7. Optimize and Deploy

Objective: Improve performance and deploy the pipeline.
Tasks:
Optimize retrieval with reranking:from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)
retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=reranker)


Cache frequent queries in Chroma to reduce latency.
Deploy with FastAPI:from fastapi import FastAPI
from langchain.chains import RetrievalQA

app = FastAPI()

@app.post("/query")
async def query_rag(query: str):
    result = rag_chain.run(query)
    return {"answer": result}

Run: uvicorn app:app --host 0.0.0.0 --port 8000
Handle context rot:
Summarize conversations after 10-20 turns using /think mode.
Store summaries in Chroma:summary = llm("Summarize the conversation: [conversation history]")
vector_store.add_documents([Document(page_content=summary, metadata={"source": "conversation"})])







8. Iterate and Improve

Objective: Refine based on performance and feedback.
Tasks:
Log user queries to identify knowledge gaps.
Fine-tune Qwen3-1.7B further with new Q&A pairs using Axolotl if needed.
Expand knowledge base with additional texts (e.g., Tafsir).
Adjust chunk size (100-200 tokens) and k (retrieved documents) based on evaluation.



Deliverables

Python scripts in scripts/ implementing the RAG pipeline.
Chroma vector store in rag_index/ with 1,000-3,000 indexed Islamic text chunks.
FastAPI app for querying the RAG pipeline.
Evaluation report (using RAGAS) with metrics (context relevance, answer faithfulness).

Notes

Ensure dataset covers Quran, Hadith, and Islamic practices (1,000-3,000 samples).
Use GPU for inference and embedding computation (Qwen3-1.7B needs ~3-4GB VRAM in 4-bit).
Monitor loss (<1.0) and accuracy (80-90%) on validation set.
Address context rot by limiting context and summarizing conversations.
