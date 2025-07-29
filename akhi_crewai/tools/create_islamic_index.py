#!/usr/bin/env python3
"""
Create Islamic Content FAISS Index

This script creates the islamic_content.faiss index file that the demo expects.
It generates sample Islamic content and stores it in the correct embeddings directory.
"""

import os
import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedder import EmbedderTool
from faiss_store import FAISSStorageTool

def create_islamic_content_index():
    """Create the islamic_content FAISS index."""
    print("🔄 Creating Islamic Content FAISS Index...")
    
    # Initialize tools
    embedder_tool = EmbedderTool()
    faiss_tool = FAISSStorageTool()
    
    # Islamic content for indexing
    islamic_texts = [
        # Basic Islamic concepts
        "The Quran is the final revelation from Allah to humanity",
        "Prophet Muhammad (PBUH) is the last messenger of Allah",
        "The five daily prayers are obligatory for every Muslim",
        "Zakat purifies wealth and helps the needy in society",
        "Hajj is the pilgrimage to the holy city of Mecca",
        "Ramadan is the holy month of fasting for Muslims",
        "The Shahada is the declaration of faith in Islam",
        "Islamic law is called Sharia and guides Muslim life",
        
        # Pillars of Islam
        "The Five Pillars of Islam are the foundation of Muslim life",
        "Salah is the ritual prayer performed five times daily",
        "Sawm is fasting during the month of Ramadan",
        "Zakat is the giving of charity to those in need",
        "Hajj is the pilgrimage to Mecca for those who are able",
        
        # Islamic teachings
        "Islam teaches peace, compassion, and justice for all",
        "The Quran emphasizes the importance of knowledge and learning",
        "Muslims believe in the Day of Judgment and the afterlife",
        "Islamic ethics emphasize honesty, kindness, and respect",
        "The concept of Tawhid represents the oneness of Allah",
        
        # Arabic phrases with translations
        "Bismillah - In the name of Allah",
        "Alhamdulillah - All praise is due to Allah",
        "Subhan Allah - Glory be to Allah",
        "Allahu Akbar - Allah is the Greatest",
        "La hawla wa la quwwata illa billah - There is no power except with Allah",
        "Astaghfirullah - I seek forgiveness from Allah",
        
        # Islamic history and figures
        "The Prophet Muhammad was born in Mecca in 570 CE",
        "The first revelation came to Prophet Muhammad in the cave of Hira",
        "Abu Bakr was the first Caliph after Prophet Muhammad",
        "The Battle of Badr was a significant victory for early Muslims",
        "The Treaty of Hudaybiyyah demonstrated Islamic diplomacy",
        
        # Islamic practices and rituals
        "Wudu is the ritual ablution performed before prayer",
        "The Qibla is the direction Muslims face during prayer",
        "Jumu'ah is the congregational Friday prayer",
        "Eid al-Fitr celebrates the end of Ramadan",
        "Eid al-Adha commemorates Ibrahim's willingness to sacrifice",
        
        # Quranic teachings
        "The Quran teaches that there is no compulsion in religion",
        "Islamic teachings emphasize the importance of family",
        "The Quran speaks about the creation of the heavens and earth",
        "Islamic law protects the rights of all people",
        "The Quran encourages seeking knowledge from cradle to grave"
    ]
    
    print(f"📚 Processing {len(islamic_texts)} Islamic texts...")
    
    # Generate embeddings
    try:
        embedding_result = embedder_tool._run(
            text_chunks=islamic_texts,
            output_format="list",
            model_name="all-MiniLM-L6-v2"
        )
        
        if not embedding_result or len(embedding_result) == 0:
            print("❌ Embedding generation failed")
            return False
            
        print(f"✅ Generated {len(embedding_result)} embeddings")
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False
    
    # Prepare data for FAISS storage
    embeddings = [item['embedding'] for item in embedding_result]
    metadata = []
    
    for i, item in enumerate(embedding_result):
        text = item['text']
        meta = {
            "text": text,
            "chunk_id": i,
            "source": "islamic_content_demo",
            "language": "Arabic" if any(ord(c) > 127 for c in text) else "English",
            "content_type": "Islamic",
            "is_arabic": any(ord(c) > 127 for c in text),
            "is_dua": "الله" in text or "Allah" in text,
            "is_pillar": any(pillar in text.lower() for pillar in ["salah", "zakat", "hajj", "ramadan", "shahada"]),
            "dimension": item['dimension'],
            "model_name": item['model_name']
        }
        metadata.append(meta)
    
    # Set the correct embeddings directory
    embeddings_dir = "/data/work/dev/akhi_data_builder/akhi_crewai/data/embeddings"
    
    # Create directory if it doesn't exist
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    
    # Store in FAISS
    try:
        storage_result = faiss_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="islamic_content",
            index_type="flat",
            save_index=True,
            index_dir=embeddings_dir
        )
        
        if storage_result.success:
            print(f"✅ Islamic Content Index Created Successfully!")
            print(f"   📁 Index file: {storage_result.index_path}")
            print(f"   📄 Metadata file: {storage_result.metadata_path}")
            print(f"   📊 Vectors stored: {storage_result.total_vectors}")
            print(f"   📏 Dimension: {storage_result.dimension}")
            print(f"   ⏱️  Processing time: {storage_result.processing_time:.3f}s")
            print(f"   💾 Index size: {storage_result.index_size_mb:.2f} MB")
            
            # Verify the files exist
            index_file = Path(storage_result.index_path)
            metadata_file = Path(storage_result.metadata_path)
            
            if index_file.exists() and metadata_file.exists():
                print(f"\n🎉 Index files verified:")
                print(f"   ✅ {index_file}")
                print(f"   ✅ {metadata_file}")
                return True
            else:
                print(f"❌ Index files not found after creation")
                return False
        else:
            print(f"❌ Failed to create index: {storage_result.error}")
            return False
            
    except Exception as e:
        print(f"❌ Error storing in FAISS: {e}")
        return False

if __name__ == "__main__":
    print("Islamic Content FAISS Index Creator")
    print("=" * 40)
    
    success = create_islamic_content_index()
    
    if success:
        print("\n🎉 Islamic content index created successfully!")
        print("   The demo should now work with batch processing.")
    else:
        print("\n❌ Failed to create Islamic content index.")
        print("   Please check the error messages above.")