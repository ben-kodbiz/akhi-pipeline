#!/usr/bin/env python3
"""
Test script to demonstrate multi-format document loading capabilities
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

from document_loader import EnhancedDocumentLoader

def create_sample_files():
    """Create sample files in different formats for testing"""
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    with open(data_dir / "sample.txt", "w", encoding="utf-8") as f:
        f.write("""
Bismillah - In the name of Allah, the Most Gracious, the Most Merciful.

The Five Pillars of Islam are:
1. Shahada (Declaration of Faith)
2. Salah (Prayer)
3. Zakat (Charity)
4. Sawm (Fasting during Ramadan)
5. Hajj (Pilgrimage to Mecca)

These pillars form the foundation of a Muslim's faith and practice.
""")
    
    # Create sample HTML file
    with open(data_dir / "sample.html", "w", encoding="utf-8") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Islamic Knowledge</title>
</head>
<body>
    <h1>The Quran</h1>
    <p>The Quran is the holy book of Islam, believed by Muslims to be the word of Allah as revealed to Prophet Muhammad (peace be upon him).</p>
    
    <h2>Key Features:</h2>
    <ul>
        <li>114 chapters (Surahs)</li>
        <li>Over 6,000 verses (Ayahs)</li>
        <li>Revealed over 23 years</li>
        <li>Written in Arabic</li>
    </ul>
    
    <p>The Quran provides guidance for all aspects of life and is considered the final revelation from Allah.</p>
</body>
</html>
""")
    
    # Create sample JSON file
    import json
    sample_qa = {
        "islamic_qa": [
            {
                "question": "What is the meaning of Islam?",
                "answer": "Islam means 'submission to Allah' and 'peace'. It comes from the Arabic root s-l-m, which relates to peace, purity, submission, and obedience.",
                "category": "basics"
            },
            {
                "question": "Who is Prophet Muhammad?",
                "answer": "Prophet Muhammad (peace be upon him) is the final messenger of Allah, born in Mecca in 570 CE. He received the Quran through the angel Gabriel (Jibril) and is considered the seal of the prophets.",
                "category": "prophet"
            },
            {
                "question": "What are the 99 names of Allah?",
                "answer": "The 99 names of Allah (Asma ul-Husna) are the beautiful names and attributes of Allah mentioned in the Quran and Hadith, such as Ar-Rahman (The Compassionate), Ar-Rahim (The Merciful), Al-Malik (The King), etc.",
                "category": "theology"
            }
        ]
    }
    
    with open(data_dir / "sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_qa, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample files in {data_dir}:")
    print("  - sample.txt (Plain text)")
    print("  - sample.html (HTML)")
    print("  - sample.json (JSON)")
    print("\nNote: PDF and DOCX files require manual creation or external tools.")
    
    return data_dir

def test_document_loader():
    """Test the enhanced document loader with different file formats"""
    print("=" * 60)
    print("Testing Enhanced Document Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = EnhancedDocumentLoader(chunk_size=256, chunk_overlap=25)
    
    # Show supported formats
    print("\n1. Supported File Formats:")
    print("-" * 30)
    formats = loader.get_supported_formats()
    for fmt, available in formats.items():
        status = "✓ Available" if available else "✗ Missing dependency"
        print(f"  .{fmt:<6} {status}")
    
    # Show installation command for missing dependencies
    install_cmd = loader.install_missing_dependencies()
    print(f"\n2. Installation Command:")
    print("-" * 30)
    print(f"  {install_cmd}")
    
    # Create sample files
    print(f"\n3. Creating Sample Files:")
    print("-" * 30)
    data_dir = create_sample_files()
    
    # Test loading individual files
    print(f"\n4. Testing Individual File Loading:")
    print("-" * 30)
    
    sample_files = [
        "sample.txt",
        "sample.html", 
        "sample.json"
    ]
    
    total_chunks = 0
    for filename in sample_files:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                documents = loader.load_document(file_path)
                total_chunks += len(documents)
                print(f"  {filename:<15} → {len(documents)} chunks")
                
                # Show first chunk preview
                if documents:
                    preview = documents[0].page_content[:100].replace('\n', ' ')
                    print(f"    Preview: {preview}...")
                    print(f"    Metadata: {documents[0].metadata}")
                    
            except Exception as e:
                print(f"  {filename:<15} → Error: {str(e)}")
        else:
            print(f"  {filename:<15} → File not found")
    
    # Test loading from directory
    print(f"\n5. Testing Directory Loading:")
    print("-" * 30)
    try:
        all_documents = loader.load_documents_from_directory(data_dir, recursive=False)
        print(f"  Total documents from directory: {len(all_documents)}")
        
        # Group by file type
        by_type = {}
        for doc in all_documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            by_type[file_type] = by_type.get(file_type, 0) + 1
        
        for file_type, count in by_type.items():
            print(f"    {file_type}: {count} chunks")
            
    except Exception as e:
        print(f"  Error loading directory: {str(e)}")
    
    # Test loading from file paths
    print(f"\n6. Testing Path List Loading:")
    print("-" * 30)
    file_paths = [str(data_dir / f) for f in sample_files if (data_dir / f).exists()]
    
    try:
        path_documents = loader.load_documents_from_paths(file_paths)
        print(f"  Total documents from paths: {len(path_documents)}")
        
        # Show sources
        sources = set(doc.metadata.get('source', 'unknown') for doc in path_documents)
        print(f"  Sources: {', '.join(sources)}")
        
    except Exception as e:
        print(f"  Error loading from paths: {str(e)}")
    
    print(f"\n7. Summary:")
    print("-" * 30)
    print(f"  Total chunks processed: {total_chunks}")
    print(f"  Chunk size: {loader.chunk_size}")
    print(f"  Chunk overlap: {loader.chunk_overlap}")
    
    return loader

def demonstrate_usage():
    """Demonstrate practical usage scenarios"""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("""
# Basic Usage:
from document_loader import EnhancedDocumentLoader
from pathlib import Path

# Initialize loader
loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)

# Load single file
documents = loader.load_document(Path("quran.pdf"))

# Load directory
documents = loader.load_documents_from_directory(Path("islamic_texts/"))

# Load specific files
files = ["quran.pdf", "hadith.docx", "tafsir.html"]
documents = loader.load_documents_from_paths(files)

# Check what formats are supported
formats = loader.get_supported_formats()
print("Supported formats:", [f for f, available in formats.items() if available])

# Get installation command for missing dependencies
print("Install missing:", loader.install_missing_dependencies())
""")

def main():
    """Main function"""
    try:
        # Test the document loader
        loader = test_document_loader()
        
        # Show usage examples
        demonstrate_usage()
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())