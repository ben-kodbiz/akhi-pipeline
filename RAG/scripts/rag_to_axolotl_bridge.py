#!/usr/bin/env python3
"""
RAG to Axolotl Bridge Script
Converts RAG-processed documents into Axolotl training format

This script extracts document chunks from the RAG system and converts them
into conversational training data suitable for QLoRA fine-tuning with Axolotl.

Usage:
    python rag_to_axolotl_bridge.py --output ./training_data
    python rag_to_axolotl_bridge.py --config config.yaml --format conversation

Author: Akhi Data Builder Team
Date: January 2025
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_loader import EnhancedDocumentLoader
from rag_pipeline import IslamicRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGToAxolotlBridge:
    """
    Bridge class to convert RAG documents into Axolotl training format
    """
    
    def __init__(self, config_path: str = "config.yaml", output_dir: str = "training_data"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load RAG configuration
        self.config = self.load_config()
        
        # Initialize document loader
        self.document_loader = EnhancedDocumentLoader(
            chunk_size=self.config.get('retrieval', {}).get('chunk_size', 1000),
            chunk_overlap=self.config.get('retrieval', {}).get('chunk_overlap', 200)
        )
        
        # Statistics
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'generated_conversations': 0,
            'skipped_chunks': 0,
            'avg_chunk_length': 0
        }
        
        logger.info(f"RAG to Axolotl Bridge initialized with output directory: {self.output_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load RAG configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return {
                'data': {'sources': []},
                'retrieval': {'chunk_size': 1000, 'chunk_overlap': 200}
            }
    
    def extract_rag_documents(self) -> List[Dict[str, Any]]:
        """Extract all processed documents from RAG system"""
        logger.info("Extracting documents from RAG system...")
        
        # Get data sources from config
        data_sources = self.config.get('data', {}).get('sources', [])
        
        if not data_sources:
            logger.warning("No data sources found in config")
            return []
        
        all_documents = []
        
        for source in data_sources:
            try:
                # Load documents from each source
                documents = self.document_loader.load_documents_from_paths([source])
                
                for doc in documents:
                    doc_data = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source': source,
                        'length': len(doc.page_content)
                    }
                    all_documents.append(doc_data)
                
                logger.info(f"Extracted {len(documents)} chunks from {source}")
                self.stats['total_documents'] += 1
                
            except Exception as e:
                logger.error(f"Error processing source {source}: {str(e)}")
                continue
        
        self.stats['total_chunks'] = len(all_documents)
        if all_documents:
            self.stats['avg_chunk_length'] = sum(doc['length'] for doc in all_documents) / len(all_documents)
        
        logger.info(f"Total extracted: {len(all_documents)} document chunks")
        return all_documents
    
    def generate_qa_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate question-answer pairs from a document chunk"""
        content = chunk['content'].strip()
        
        # Skip very short chunks
        if len(content) < 100:
            self.stats['skipped_chunks'] += 1
            return []
        
        # Generate different types of questions based on content
        conversations = []
        
        # 1. Direct content question
        if self.is_islamic_content(content):
            qa_pair = self.create_islamic_qa(content, chunk['metadata'])
            if qa_pair:
                conversations.append(qa_pair)
        
        # 2. Summary question
        if len(content) > 500:
            summary_qa = self.create_summary_qa(content, chunk['metadata'])
            if summary_qa:
                conversations.append(summary_qa)
        
        # 3. Specific detail question
        detail_qa = self.create_detail_qa(content, chunk['metadata'])
        if detail_qa:
            conversations.append(detail_qa)
        
        return conversations
    
    def is_islamic_content(self, content: str) -> bool:
        """Check if content is Islamic-related"""
        islamic_keywords = [
            'allah', 'prophet', 'muhammad', 'quran', 'hadith', 'islam', 'muslim',
            'prayer', 'salah', 'hajj', 'ramadan', 'zakat', 'shahada', 'sunnah',
            'dua', 'mosque', 'masjid', 'imam', 'ummah', 'jihad', 'halal', 'haram'
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in islamic_keywords)
    
    def create_islamic_qa(self, content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create Islamic knowledge Q&A pair"""
        # Extract key concepts for question generation
        if 'allah' in content.lower():
            question = "What does Islam teach about Allah?"
        elif 'prophet' in content.lower() or 'muhammad' in content.lower():
            question = "What can you tell me about Prophet Muhammad (PBUH)?"
        elif 'prayer' in content.lower() or 'salah' in content.lower():
            question = "Can you explain about Islamic prayer?"
        elif 'quran' in content.lower():
            question = "What does the Quran say about this topic?"
        elif 'hadith' in content.lower():
            question = "What do the Hadith teach about this?"
        else:
            question = "Can you explain this Islamic concept?"
        
        return {
            'conversations': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': content}
            ],
            'source': metadata.get('source', 'unknown'),
            'source_file': metadata.get('file_path', 'unknown'),
            'processed_at': datetime.now().isoformat(),
            'content_type': 'islamic_knowledge'
        }
    
    def create_summary_qa(self, content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create summary-based Q&A pair"""
        # Create a summary question
        question = "Can you summarize the main points from this Islamic text?"
        
        # Create a condensed version of the content
        sentences = content.split('. ')
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '...'
        else:
            summary = content
        
        return {
            'conversations': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': summary}
            ],
            'source': metadata.get('source', 'unknown'),
            'source_file': metadata.get('file_path', 'unknown'),
            'processed_at': datetime.now().isoformat(),
            'content_type': 'summary'
        }
    
    def create_detail_qa(self, content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create detail-focused Q&A pair"""
        # Look for specific details in the content
        if re.search(r'\d+', content):  # Contains numbers
            question = "What are the specific details mentioned in this Islamic teaching?"
        elif 'step' in content.lower() or 'first' in content.lower():
            question = "What are the steps or procedures mentioned?"
        elif 'example' in content.lower():
            question = "Can you provide examples from this Islamic text?"
        else:
            question = "What specific information does this Islamic text provide?"
        
        return {
            'conversations': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': content}
            ],
            'source': metadata.get('source', 'unknown'),
            'source_file': metadata.get('file_path', 'unknown'),
            'processed_at': datetime.now().isoformat(),
            'content_type': 'detailed_info'
        }
    
    def convert_to_axolotl_format(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert conversations to Axolotl training format"""
        logger.info(f"Converting {len(conversations)} conversations to Axolotl format...")
        
        axolotl_data = []
        
        for conv in conversations:
            # Ensure proper conversation format
            if 'conversations' in conv and len(conv['conversations']) >= 2:
                axolotl_sample = {
                    'conversations': conv['conversations'],
                    'source': conv.get('source', 'rag_system'),
                    'source_file': conv.get('source_file', 'unknown'),
                    'processed_at': conv.get('processed_at', datetime.now().isoformat()),
                    'content_type': conv.get('content_type', 'general')
                }
                axolotl_data.append(axolotl_sample)
                self.stats['generated_conversations'] += 1
        
        logger.info(f"Generated {len(axolotl_data)} Axolotl training samples")
        return axolotl_data
    
    def save_training_data(self, data: List[Dict[str, Any]], filename: str = "rag_training_data.jsonl") -> Path:
        """Save training data in JSONL format"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} training samples to {output_path}")
        return output_path
    
    def save_statistics(self) -> Path:
        """Save conversion statistics"""
        stats_path = self.output_dir / 'conversion_stats.json'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'conversion_stats': self.stats,
                'generated_at': datetime.now().isoformat(),
                'config_used': str(self.config_path)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved statistics to {stats_path}")
        return stats_path
    
    def generate_readme(self, training_file: Path) -> Path:
        """Generate README for the training data"""
        readme_path = self.output_dir / 'README.md'
        
        readme_content = f"""# RAG to Axolotl Training Data

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This directory contains training data converted from the RAG system for Axolotl fine-tuning.

## Statistics

- **Total Documents Processed**: {self.stats['total_documents']}
- **Total Chunks Extracted**: {self.stats['total_chunks']}
- **Generated Conversations**: {self.stats['generated_conversations']}
- **Skipped Chunks**: {self.stats['skipped_chunks']}
- **Average Chunk Length**: {self.stats['avg_chunk_length']:.1f} characters

## Files

- `{training_file.name}` - Training data in JSONL format
- `conversion_stats.json` - Detailed conversion statistics
- `README.md` - This file

## Usage with Axolotl

### Step 1: Prepare Dataset

```bash
# Navigate to akhi_crewai directory
cd ../akhi_crewai

# Run the Axolotl dataset preparation
python prepare_axolotl_dataset.py \
  --input ../RAG/training_data \
  --output ./axolotl_ready \
  --model microsoft/DialoGPT-medium
```

### Step 2: Train Model

```bash
# Navigate to prepared dataset
cd axolotl_ready

# Run training
./train_model.sh
```

## Data Format

Each line in the JSONL file contains:

```json
{{
  "conversations": [
    {{"role": "user", "content": "Question about Islamic topic"}},
    {{"role": "assistant", "content": "Islamic guidance and answer"}}
  ],
  "source": "path/to/source/file",
  "source_file": "filename.pdf",
  "processed_at": "2025-01-XX...",
  "content_type": "islamic_knowledge"
}}
```

## Quality Assurance

- ✅ Islamic content validation
- ✅ Minimum content length filtering
- ✅ Conversation format validation
- ✅ Source tracking
- ✅ Metadata preservation

## Next Steps

1. Review the generated training data
2. Run the Axolotl dataset preparation
3. Configure training parameters
4. Start QLoRA fine-tuning
5. Validate the trained model
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Generated README: {readme_path}")
        return readme_path
    
    def run_conversion(self) -> Dict[str, Any]:
        """Run the complete RAG to Axolotl conversion process"""
        logger.info("Starting RAG to Axolotl conversion...")
        
        # Extract documents from RAG system
        documents = self.extract_rag_documents()
        
        if not documents:
            raise ValueError("No documents found in RAG system")
        
        # Generate conversations from documents
        all_conversations = []
        
        for doc in documents:
            conversations = self.generate_qa_from_chunk(doc)
            all_conversations.extend(conversations)
        
        if not all_conversations:
            raise ValueError("No conversations generated from documents")
        
        # Convert to Axolotl format
        axolotl_data = self.convert_to_axolotl_format(all_conversations)
        
        # Save training data
        training_file = self.save_training_data(axolotl_data)
        
        # Save statistics
        stats_file = self.save_statistics()
        
        # Generate README
        readme_file = self.generate_readme(training_file)
        
        result = {
            'training_data': str(training_file),
            'statistics': str(stats_file),
            'readme': str(readme_file),
            'stats': self.stats.copy()
        }
        
        logger.info("RAG to Axolotl conversion completed successfully!")
        return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Convert RAG documents to Axolotl training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_to_axolotl_bridge.py --output ./training_data
  python rag_to_axolotl_bridge.py --config config.yaml --output ./axolotl_input
  python rag_to_axolotl_bridge.py --config config.yaml --output ./training_data --verbose

The script will:
1. Extract all processed documents from the RAG system
2. Generate question-answer pairs from document chunks
3. Convert to Axolotl training format
4. Save as JSONL files ready for dataset preparation
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='RAG configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='training_data',
        help='Output directory for training data (default: training_data)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize bridge
        bridge = RAGToAxolotlBridge(
            config_path=args.config,
            output_dir=args.output
        )
        
        # Run conversion
        result = bridge.run_conversion()
        
        # Print summary
        print("\n" + "="*60)
        print("RAG TO AXOLOTL CONVERSION COMPLETED")
        print("="*60)
        print(f"📊 Conversion Statistics:")
        print(f"   Documents processed: {result['stats']['total_documents']}")
        print(f"   Chunks extracted: {result['stats']['total_chunks']}")
        print(f"   Conversations generated: {result['stats']['generated_conversations']}")
        print(f"   Skipped chunks: {result['stats']['skipped_chunks']}")
        print(f"   Avg chunk length: {result['stats']['avg_chunk_length']:.1f} chars")
        print(f"\n📁 Generated Files:")
        print(f"   Training data: {result['training_data']}")
        print(f"   Statistics: {result['statistics']}")
        print(f"   README: {result['readme']}")
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review the generated training data")
        print(f"   2. cd ../akhi_crewai")
        print(f"   3. python prepare_axolotl_dataset.py --input ../RAG/{args.output} --output ./axolotl_ready")
        print(f"   4. cd axolotl_ready && ./train_model.sh")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()