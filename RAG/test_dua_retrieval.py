#!/usr/bin/env python3
"""
Test script to verify dua.pdf retrieval functionality without QLoRA model
"""

import sys
import yaml
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

from document_loader import EnhancedDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_dua_pdf_retrieval():
    """Test dua.pdf document retrieval without language model"""
    print("=" * 60)
    print("Testing dua.pdf Retrieval System")
    print("=" * 60)
    
    try:
        # Step 1: Load documents
        print("📄 Step 1: Loading dua.pdf...")
        loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)
        documents = loader.load_document(Path('dua.pdf'))
        print(f"✅ Loaded {len(documents)} chunks")
        
        # Step 2: Initialize embeddings
        print("\n🔤 Step 2: Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embeddings initialized")
        
        # Step 3: Create vector store
        print("\n🗃️  Step 3: Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./test_rag_index"
        )
        print("✅ Vector store created")
        
        # Step 4: Test retrieval
        print("\n🔍 Step 4: Testing document retrieval...")
        
        test_queries = [
            "What is dua?",
            "importance of dua",
            "benefits of making dua",
            "how to make dua",
            "dua in Islam"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: '{query}'")
            
            # Retrieve relevant documents
            retrieved_docs = vectorstore.similarity_search(query, k=3)
            
            print(f"   📊 Retrieved {len(retrieved_docs)} documents")
            
            for j, doc in enumerate(retrieved_docs, 1):
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   📄 Doc {j}: {content_preview}...")
                print(f"        Source: {doc.metadata.get('source', 'unknown')}")
                print(f"        Chunk: {doc.metadata.get('chunk_id', 'unknown')}")
        
        # Step 5: Test similarity search with scores
        print("\n📊 Step 5: Testing similarity search with scores...")
        query = "What is the importance of dua?"
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        
        print(f"\n🎯 Top results for: '{query}'")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            content_preview = doc.page_content[:150].replace('\n', ' ')
            print(f"   {i}. Score: {score:.4f}")
            print(f"      Content: {content_preview}...")
            print(f"      Metadata: {doc.metadata}")
            print()
        
        print("✅ All retrieval tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during retrieval testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dua_content_analysis():
    """Analyze the content of dua.pdf"""
    print("\n" + "=" * 60)
    print("Analyzing dua.pdf Content")
    print("=" * 60)
    
    try:
        # Load documents
        loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)
        documents = loader.load_document(Path('dua.pdf'))
        
        # Analyze content
        print(f"📊 Content Analysis:")
        print(f"   Total chunks: {len(documents)}")
        
        # Word frequency analysis
        all_text = " ".join([doc.page_content for doc in documents])
        words = all_text.lower().split()
        
        # Common Islamic terms
        islamic_terms = ['allah', 'dua', 'prayer', 'islam', 'muslim', 'prophet', 'quran', 'hadith']
        term_counts = {term: words.count(term) for term in islamic_terms}
        
        print(f"\n🔤 Islamic Term Frequency:")
        for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {term}: {count} occurrences")
        
        # Show sample chunks
        print(f"\n📖 Sample Chunks:")
        for i in range(min(3, len(documents))):
            print(f"\n   Chunk {i+1}:")
            print(f"   {documents[i].page_content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during content analysis: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing dua.pdf Retrieval Functionality")
    
    # Test 1: Document retrieval
    retrieval_success = test_dua_pdf_retrieval()
    
    # Test 2: Content analysis
    analysis_success = test_dua_content_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"🔍 Document Retrieval: {'✅ PASSED' if retrieval_success else '❌ FAILED'}")
    print(f"📊 Content Analysis: {'✅ PASSED' if analysis_success else '❌ FAILED'}")
    
    if retrieval_success and analysis_success:
        print("\n🎉 All tests passed! dua.pdf is working perfectly with the RAG system.")
        print("\n📋 Summary:")
        print("   ✅ PDF loading and chunking works")
        print("   ✅ Embeddings generation works")
        print("   ✅ Vector store creation works")
        print("   ✅ Similarity search works")
        print("   ✅ Content is properly indexed")
        
        print("\n💡 The dua.pdf file is ready for use in your RAG pipeline!")
        print("   You can now add it to your config.yaml sources list.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if (retrieval_success and analysis_success) else 1

if __name__ == "__main__":
    exit(main())