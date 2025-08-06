#!/usr/bin/env python3
"""
Test script to verify dua.pdf integration with RAG pipeline
"""

import sys
import yaml
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

from document_loader import EnhancedDocumentLoader
from rag_pipeline import IslamicRAGPipeline

def test_dua_pdf_loading():
    """Test loading dua.pdf with document loader"""
    print("=" * 60)
    print("Testing dua.pdf Document Loading")
    print("=" * 60)
    
    try:
        # Initialize document loader
        loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)
        
        # Load dua.pdf
        pdf_path = Path('dua.pdf')
        if not pdf_path.exists():
            print("❌ dua.pdf not found in current directory")
            return False
            
        print(f"📄 Loading {pdf_path}...")
        documents = loader.load_document(pdf_path)
        
        print(f"✅ Successfully loaded {len(documents)} chunks from dua.pdf")
        
        # Show sample content
        if documents:
            print(f"\n📖 First chunk preview:")
            print(f"   Content: {documents[0].page_content[:150]}...")
            print(f"   Metadata: {documents[0].metadata}")
            
            print(f"\n📊 Document Statistics:")
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"   Total chunks: {len(documents)}")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Average chunk size: {total_chars // len(documents)} chars")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading dua.pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dua_pdf_in_rag_pipeline():
    """Test dua.pdf integration with RAG pipeline"""
    print("\n" + "=" * 60)
    print("Testing dua.pdf in RAG Pipeline")
    print("=" * 60)
    
    try:
        # Load config and modify for dua.pdf
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config to use dua.pdf
        config['data']['sources'] = ['dua.pdf']
        
        print("🔧 Initializing RAG pipeline with dua.pdf...")
        
        # Initialize pipeline with config dict
        pipeline = IslamicRAGPipeline(config)
        
        print("✅ Pipeline initialized successfully")
        
        # Test queries related to dua
        test_queries = [
            "What is the importance of dua in Islam?",
            "How should Muslims make dua?",
            "What are the benefits of making dua?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            
            try:
                response = pipeline.query(query)
                
                print(f"✅ Response received:")
                print(f"   Answer: {response['answer'][:200]}...")
                print(f"   Sources: {response.get('sources', 'N/A')}")
                print(f"   Confidence: {response.get('confidence', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Testing dua.pdf Integration with RAG System")
    print("=" * 60)
    
    # Test 1: Document loading
    loading_success = test_dua_pdf_loading()
    
    # Test 2: RAG pipeline integration (only if loading succeeded)
    if loading_success:
        pipeline_success = test_dua_pdf_in_rag_pipeline()
    else:
        pipeline_success = False
        print("\n⚠️  Skipping pipeline test due to loading failure")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"📄 Document Loading: {'✅ PASSED' if loading_success else '❌ FAILED'}")
    print(f"🔗 Pipeline Integration: {'✅ PASSED' if pipeline_success else '❌ FAILED'}")
    
    if loading_success and pipeline_success:
        print("\n🎉 All tests passed! dua.pdf is ready for use in the RAG system.")
        print("\n💡 Next steps:")
        print("   1. Add dua.pdf to your config.yaml sources")
        print("   2. Run the full pipeline with: python scripts/rag_pipeline.py")
        print("   3. Start the API server with: python scripts/api_server.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if (loading_success and pipeline_success) else 1

if __name__ == "__main__":
    exit(main())