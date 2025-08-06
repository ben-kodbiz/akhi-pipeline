#!/usr/bin/env python3
"""
Document Loader for Islamic RAG Pipeline
Supports multiple file formats: HTML, PDF, DOCX, TXT, JSON
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Core imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document loaders
try:
    from langchain.document_loaders import (
        TextLoader,
        PyPDFLoader,
        UnstructuredHTMLLoader,
        Docx2txtLoader,
        JSONLoader
    )
    LANGCHAIN_LOADERS_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADERS_AVAILABLE = False
    logging.warning("LangChain document loaders not available. Installing fallback parsers...")

# Fallback parsers
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# OCR imports
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentLoader:
    """Enhanced document loader supporting multiple file formats"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document loader"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt': self._load_text,
            '.html': self._load_html,
            '.htm': self._load_html,
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.json': self._load_json,
            '.png': self._load_image,
            '.jpg': self._load_image,
            '.jpeg': self._load_image,
            '.tiff': self._load_image,
            '.bmp': self._load_image
        }
        
        logger.info(f"Document loader initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        self._log_available_parsers()
    
    def _log_available_parsers(self):
        """Log available parsers and their status"""
        logger.info("Available document parsers:")
        logger.info(f"  Text files (.txt): ✓ Always available")
        logger.info(f"  HTML files (.html, .htm): {'✓' if HTML_AVAILABLE else '✗'} {'Available' if HTML_AVAILABLE else 'Install beautifulsoup4'}")
        logger.info(f"  PDF files (.pdf): {'✓' if PDF_AVAILABLE else '✗'} {'Available' if PDF_AVAILABLE else 'Install PyPDF2 or pypdf'}")
        logger.info(f"  Word files (.docx, .doc): {'✓' if DOCX_AVAILABLE else '✗'} {'Available' if DOCX_AVAILABLE else 'Install python-docx'}")
        logger.info(f"  JSON files (.json): ✓ Always available")
        logger.info(f"  Image files (.png, .jpg, .jpeg, .tiff, .bmp): {'✓' if OCR_AVAILABLE else '✗'} {'Available' if OCR_AVAILABLE else 'Install pytesseract, Pillow, pdf2image'}")
        logger.info(f"  OCR capabilities: {'✓' if OCR_AVAILABLE else '✗'} {'Available' if OCR_AVAILABLE else 'Install pytesseract, Pillow, pdf2image'}")
        logger.info(f"  LangChain loaders: {'✓' if LANGCHAIN_LOADERS_AVAILABLE else '✗'} {'Available' if LANGCHAIN_LOADERS_AVAILABLE else 'Install langchain[all]'}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Arabic text and common punctuation
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF.,!?;:()\[\]"\'-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _load_text(self, file_path: Path) -> str:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode text file: {file_path}")
    
    def _load_html(self, file_path: Path) -> str:
        """Load HTML file"""
        if LANGCHAIN_LOADERS_AVAILABLE:
            try:
                loader = UnstructuredHTMLLoader(str(file_path))
                docs = loader.load()
                return "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"LangChain HTML loader failed: {e}. Using fallback.")
        
        # Fallback HTML parser
        if not HTML_AVAILABLE:
            raise ImportError("BeautifulSoup4 not available. Install with: pip install beautifulsoup4")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            raise ValueError(f"Error parsing HTML file {file_path}: {str(e)}")
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file with OCR fallback"""
        text = ""
        
        # First, try text extraction
        if LANGCHAIN_LOADERS_AVAILABLE:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                text = "\n".join([doc.page_content for doc in docs])
                if text.strip():
                    logger.info(f"Successfully extracted text from PDF using LangChain")
                    return text
            except Exception as e:
                logger.warning(f"LangChain PDF loader failed: {e}. Using fallback.")
        
        # Fallback PDF parser
        if PDF_AVAILABLE:
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                text += f"\n\n--- Page {page_num + 1} ---\n\n"
                                text += page_text
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                            continue
                
                if text.strip():
                    logger.info(f"Successfully extracted text from PDF using PyPDF2")
                    return text
                    
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # If no text extracted, try OCR
        if not text.strip() and OCR_AVAILABLE:
            logger.info(f"No extractable text found in PDF. Attempting OCR...")
            try:
                return self._extract_text_with_ocr_from_pdf(file_path)
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                raise ValueError(f"Could not extract text from PDF using either text extraction or OCR: {str(e)}")
        
        # If OCR not available and no text extracted
        if not text.strip():
            if not OCR_AVAILABLE:
                raise ValueError(f"No text could be extracted from PDF and OCR is not available. Install pytesseract, Pillow, and pdf2image for OCR support.")
            else:
                raise ValueError(f"No text could be extracted from PDF")
        
        return text
    
    def _extract_text_with_ocr_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using OCR"""
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries not available. Install with: pip install pytesseract Pillow pdf2image")
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images for OCR processing: {file_path}")
            images = convert_from_path(str(file_path))
            total_pages = len(images)
            logger.info(f"PDF converted to {total_pages} images. Starting OCR processing...")
            
            text = ""
            processed_pages = 0
            
            for page_num, image in enumerate(images):
                try:
                    # Log progress every 10 pages or for large PDFs
                    if page_num % 10 == 0 or total_pages > 50:
                        logger.info(f"Processing page {page_num + 1}/{total_pages} with OCR...")
                    
                    # Configure Tesseract for better Arabic support
                    custom_config = r'--oem 3 --psm 6 -l eng+ara'
                    page_text = pytesseract.image_to_string(image, config=custom_config)
                    
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} (OCR) ---\n\n"
                        text += page_text
                        processed_pages += 1
                        
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    continue
            
            if not text.strip():
                raise ValueError("No text could be extracted using OCR")
            
            logger.info(f"Successfully extracted text from PDF using OCR ({processed_pages}/{total_pages} pages processed)")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {str(e)}")
            raise ValueError(f"OCR extraction failed: {str(e)}")
    
    def _load_image(self, file_path: Path) -> str:
        """Load image file using OCR"""
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries not available. Install with: pip install pytesseract Pillow")
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Configure Tesseract for better Arabic support
            custom_config = r'--oem 3 --psm 6 -l eng+ara'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            if not text.strip():
                raise ValueError("No text could be extracted from image")
            
            logger.info(f"Successfully extracted text from image using OCR")
            return text
            
        except Exception as e:
            raise ValueError(f"Error processing image file {file_path}: {str(e)}")
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file"""
        if LANGCHAIN_LOADERS_AVAILABLE:
            try:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                return "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"LangChain DOCX loader failed: {e}. Using fallback.")
        
        # Fallback DOCX parser
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text += " | ".join(row_text) + "\n"
            
            return text
            
        except Exception as e:
            raise ValueError(f"Error parsing DOCX file {file_path}: {str(e)}")
    
    def _load_json(self, file_path: Path) -> str:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to text
            if isinstance(data, list):
                # Handle list of objects
                text_parts = []
                for item in data:
                    if isinstance(item, dict):
                        # Extract text from common fields
                        text_fields = ['text', 'content', 'body', 'description', 'question', 'answer']
                        for field in text_fields:
                            if field in item and isinstance(item[field], str):
                                text_parts.append(item[field])
                    elif isinstance(item, str):
                        text_parts.append(item)
                
                return "\n\n".join(text_parts)
            
            elif isinstance(data, dict):
                # Handle single object
                text_parts = []
                
                def extract_text_from_dict(obj, prefix=""):
                    for key, value in obj.items():
                        if isinstance(value, str) and len(value.strip()) > 10:
                            text_parts.append(f"{prefix}{key}: {value}")
                        elif isinstance(value, dict):
                            extract_text_from_dict(value, f"{prefix}{key}.")
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, str) and len(item.strip()) > 10:
                                    text_parts.append(f"{prefix}{key}[{i}]: {item}")
                                elif isinstance(item, dict):
                                    extract_text_from_dict(item, f"{prefix}{key}[{i}].")
                
                extract_text_from_dict(data)
                return "\n\n".join(text_parts)
            
            else:
                return str(data)
                
        except Exception as e:
            raise ValueError(f"Error parsing JSON file {file_path}: {str(e)}")
    
    def load_document(self, file_path) -> List[Document]:
        """Load a single document and split into chunks"""
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        logger.info(f"Loading document: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file extension is supported
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: {list(self.supported_extensions.keys())}")
        
        try:
            # Load text content
            loader_func = self.supported_extensions[extension]
            raw_text = loader_func(file_path)
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text.strip():
                logger.warning(f"No text content extracted from {file_path}")
                return []
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.stem,
                            "file_path": str(file_path),
                            "file_type": extension,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents_from_directory(self, directory: Path, recursive: bool = True) -> List[Document]:
        """Load all supported documents from a directory"""
        logger.info(f"Loading documents from directory: {directory}")
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found or not a directory: {directory}")
        
        all_documents = []
        supported_patterns = [f"*{ext}" for ext in self.supported_extensions.keys()]
        
        for pattern in supported_patterns:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            for file_path in files:
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} total document chunks from {directory}")
        return all_documents
    
    def load_documents_from_paths(self, file_paths: List[str]) -> List[Document]:
        """Load documents from a list of file paths"""
        logger.info(f"Loading {len(file_paths)} documents")
        
        all_documents = []
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            try:
                if file_path.is_file():
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                elif file_path.is_dir():
                    documents = self.load_documents_from_directory(file_path)
                    all_documents.extend(documents)
                else:
                    logger.warning(f"Path not found: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(all_documents)} total document chunks")
        return all_documents
    
    def get_supported_formats(self) -> Dict[str, bool]:
        """Get supported file formats and their availability"""
        return {
            'txt': True,
            'html': HTML_AVAILABLE,
            'htm': HTML_AVAILABLE,
            'pdf': PDF_AVAILABLE or OCR_AVAILABLE,  # PDF can be processed with OCR if text extraction fails
            'docx': DOCX_AVAILABLE,
            'doc': DOCX_AVAILABLE,
            'json': True,
            'png': OCR_AVAILABLE,
            'jpg': OCR_AVAILABLE,
            'jpeg': OCR_AVAILABLE,
            'tiff': OCR_AVAILABLE,
            'bmp': OCR_AVAILABLE
        }
    
    def install_missing_dependencies(self) -> str:
        """Generate installation commands for missing dependencies"""
        missing = []
        
        if not HTML_AVAILABLE:
            missing.append("beautifulsoup4")
        if not PDF_AVAILABLE:
            missing.append("PyPDF2")
        if not DOCX_AVAILABLE:
            missing.append("python-docx")
        if not OCR_AVAILABLE:
            missing.extend(["pytesseract", "Pillow", "pdf2image"])
        if not LANGCHAIN_LOADERS_AVAILABLE:
            missing.append("langchain[all]")
        
        if missing:
            cmd = f"pip install {' '.join(missing)}"
            if not OCR_AVAILABLE:
                cmd += "\n# Note: Also install Tesseract OCR system dependency:\n# Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara\n# macOS: brew install tesseract tesseract-lang\n# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            return cmd
        else:
            return "All dependencies are available!"

def main():
    """Demo function"""
    # Initialize loader
    loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)
    
    # Show supported formats
    print("Supported file formats:")
    formats = loader.get_supported_formats()
    for fmt, available in formats.items():
        status = "✓" if available else "✗"
        print(f"  .{fmt}: {status}")
    
    # Show installation command for missing dependencies
    install_cmd = loader.install_missing_dependencies()
    print(f"\nTo install missing dependencies: {install_cmd}")
    
    # Example usage
    print("\nExample usage:")
    print("  loader = EnhancedDocumentLoader()")
    print("  documents = loader.load_document(Path('document.pdf'))")
    print("  documents = loader.load_documents_from_directory(Path('data/'))")

if __name__ == "__main__":
    main()