#!/usr/bin/env python3
"""
Enhanced RAG Checker with SQLite+FAISS Backend
Fact-checking system for Islamic content using Quran/Hadith sources

Usage:
    python enhanced_rag_checker.py --build-index --quran-path quran.json --hadith-path hadith.json
    python enhanced_rag_checker.py --check-content content.txt
    python enhanced_rag_checker.py --interactive

Author: Akhi CrewAI Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import sqlite3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import re

try:
    import faiss
    import numpy as np
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers not installed. Install with: pip install sentence-transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IslamicRAGChecker:
    """Enhanced RAG checker for Islamic content validation"""
    
    def __init__(self, db_path: str = "islamic_knowledge.db", index_path: str = "islamic_index.faiss"):
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.embedding_model = None
        self.faiss_index = None
        self.db_connection = None
        
        # Islamic content patterns
        self.islamic_keywords = {
            'core_concepts': [
                'allah', 'god', 'prophet', 'muhammad', 'quran', 'hadith', 'islam', 'muslim',
                'tawheed', 'shirk', 'iman', 'kufr', 'halal', 'haram', 'sunnah', 'bidah'
            ],
            'practices': [
                'salah', 'prayer', 'zakat', 'charity', 'hajj', 'pilgrimage', 'sawm', 'fasting',
                'ramadan', 'eid', 'jummah', 'friday', 'wudu', 'ablution', 'qibla', 'mecca'
            ],
            'knowledge': [
                'fiqh', 'jurisprudence', 'aqeedah', 'creed', 'tafsir', 'interpretation',
                'seerah', 'biography', 'sahaba', 'companions', 'tabi\'een', 'scholars'
            ],
            'ethics': [
                'akhlaq', 'character', 'taqwa', 'piety', 'sabr', 'patience', 'shukr', 'gratitude',
                'tawakkul', 'trust', 'husn', 'excellence', 'adab', 'etiquette'
            ]
        }
        
        # Initialize components
        self.initialize_embedding_model()
        self.initialize_database()
        
        logger.info(f"Islamic RAG Checker initialized")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"FAISS Index: {self.index_path}")
    
    def initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            # Use a model that works well with Arabic/Islamic text
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def initialize_database(self):
        """Initialize SQLite database"""
        try:
            self.db_connection = sqlite3.connect(str(self.db_path))
            self.db_connection.row_factory = sqlite3.Row
            
            # Create tables
            self.create_tables()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create database tables"""
        cursor = self.db_connection.cursor()
        
        # Quran verses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quran_verses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                surah_number INTEGER NOT NULL,
                verse_number INTEGER NOT NULL,
                arabic_text TEXT NOT NULL,
                english_translation TEXT,
                transliteration TEXT,
                content_hash TEXT UNIQUE,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(surah_number, verse_number)
            )
        """)
        
        # Hadith table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hadith (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                book TEXT,
                hadith_number TEXT,
                arabic_text TEXT,
                english_text TEXT NOT NULL,
                narrator TEXT,
                grade TEXT,
                content_hash TEXT UNIQUE,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Content chunks table (for large texts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,  -- 'quran' or 'hadith'
                source_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content_hash TEXT UNIQUE,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Fact check results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_check_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                input_hash TEXT UNIQUE,
                islamic_score REAL,
                authenticity_score REAL,
                supporting_sources TEXT,  -- JSON array
                contradicting_sources TEXT,  -- JSON array
                recommendations TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quran_surah_verse ON quran_verses(surah_number, verse_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hadith_collection ON hadith(collection)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_source ON content_chunks(source_type, source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_check_hash ON fact_check_results(input_hash)")
        
        self.db_connection.commit()
        logger.info("Database tables created successfully")
    
    def build_knowledge_base(self, quran_path: Optional[str] = None, hadith_path: Optional[str] = None):
        """Build knowledge base from Quran and Hadith sources"""
        logger.info("Building Islamic knowledge base...")
        
        embeddings = []
        
        # Load Quran
        if quran_path and os.path.exists(quran_path):
            quran_embeddings = self.load_quran_data(quran_path)
            embeddings.extend(quran_embeddings)
            logger.info(f"Loaded {len(quran_embeddings)} Quran verse embeddings")
        
        # Load Hadith
        if hadith_path and os.path.exists(hadith_path):
            hadith_embeddings = self.load_hadith_data(hadith_path)
            embeddings.extend(hadith_embeddings)
            logger.info(f"Loaded {len(hadith_embeddings)} Hadith embeddings")
        
        if not embeddings:
            logger.warning("No data loaded. Please provide valid Quran and/or Hadith files.")
            return
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        logger.info(f"Knowledge base built successfully with {len(embeddings)} embeddings")
    
    def load_quran_data(self, quran_path: str) -> List[np.ndarray]:
        """Load Quran data and generate embeddings"""
        logger.info(f"Loading Quran data from {quran_path}")
        
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        
        embeddings = []
        cursor = self.db_connection.cursor()
        
        for surah in quran_data.get('surahs', []):
            surah_number = surah.get('number', 0)
            
            for verse in surah.get('verses', []):
                verse_number = verse.get('number', 0)
                arabic_text = verse.get('text', '')
                english_translation = verse.get('translation', '')
                transliteration = verse.get('transliteration', '')
                
                # Create content for embedding
                content_for_embedding = f"{english_translation} {transliteration}".strip()
                if not content_for_embedding:
                    content_for_embedding = arabic_text
                
                # Generate content hash
                content_hash = hashlib.md5(content_for_embedding.encode()).hexdigest()
                
                # Check if already exists
                cursor.execute("SELECT id FROM quran_verses WHERE content_hash = ?", (content_hash,))
                if cursor.fetchone():
                    continue
                
                # Generate embedding
                try:
                    embedding = self.embedding_model.encode(content_for_embedding)
                    embeddings.append(embedding)
                    embedding_id = len(embeddings) - 1
                    
                    # Store in database
                    cursor.execute("""
                        INSERT INTO quran_verses 
                        (surah_number, verse_number, arabic_text, english_translation, 
                         transliteration, content_hash, embedding_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (surah_number, verse_number, arabic_text, english_translation,
                          transliteration, content_hash, embedding_id))
                    
                except Exception as e:
                    logger.error(f"Error processing Quran {surah_number}:{verse_number}: {e}")
        
        self.db_connection.commit()
        return embeddings
    
    def load_hadith_data(self, hadith_path: str) -> List[np.ndarray]:
        """Load Hadith data and generate embeddings"""
        logger.info(f"Loading Hadith data from {hadith_path}")
        
        with open(hadith_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
        
        embeddings = []
        cursor = self.db_connection.cursor()
        
        for hadith in hadith_data.get('hadiths', []):
            collection = hadith.get('collection', '')
            book = hadith.get('book', '')
            hadith_number = hadith.get('number', '')
            arabic_text = hadith.get('arabic', '')
            english_text = hadith.get('english', '')
            narrator = hadith.get('narrator', '')
            grade = hadith.get('grade', '')
            
            # Create content for embedding
            content_for_embedding = english_text or arabic_text
            if not content_for_embedding:
                continue
            
            # Generate content hash
            content_hash = hashlib.md5(content_for_embedding.encode()).hexdigest()
            
            # Check if already exists
            cursor.execute("SELECT id FROM hadith WHERE content_hash = ?", (content_hash,))
            if cursor.fetchone():
                continue
            
            # Generate embedding
            try:
                embedding = self.embedding_model.encode(content_for_embedding)
                embeddings.append(embedding)
                embedding_id = len(embeddings) - 1
                
                # Store in database
                cursor.execute("""
                    INSERT INTO hadith 
                    (collection, book, hadith_number, arabic_text, english_text, 
                     narrator, grade, content_hash, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (collection, book, hadith_number, arabic_text, english_text,
                      narrator, grade, content_hash, embedding_id))
                
            except Exception as e:
                logger.error(f"Error processing Hadith {collection}:{hadith_number}: {e}")
        
        self.db_connection.commit()
        return embeddings
    
    def build_faiss_index(self, embeddings: List[np.ndarray]):
        """Build FAISS index from embeddings"""
        if not embeddings:
            logger.warning("No embeddings to index")
            return
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        dimension = embedding_matrix.shape[1]
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Add embeddings to index
        self.faiss_index.add(embedding_matrix)
        
        # Save index
        faiss.write_index(self.faiss_index, str(self.index_path))
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} embeddings")
    
    def load_faiss_index(self):
        """Load existing FAISS index"""
        if not self.index_path.exists():
            logger.warning(f"FAISS index not found: {self.index_path}")
            return False
        
        try:
            self.faiss_index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} embeddings")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def check_content(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Check content against Islamic knowledge base"""
        # Generate hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM fact_check_results WHERE input_hash = ?", (text_hash,))
        cached_result = cursor.fetchone()
        
        if cached_result:
            logger.info("Using cached fact-check result")
            return {
                'islamic_score': cached_result['islamic_score'],
                'authenticity_score': cached_result['authenticity_score'],
                'supporting_sources': json.loads(cached_result['supporting_sources'] or '[]'),
                'contradicting_sources': json.loads(cached_result['contradicting_sources'] or '[]'),
                'recommendations': cached_result['recommendations'],
                'cached': True
            }
        
        # Load FAISS index if not loaded
        if self.faiss_index is None:
            if not self.load_faiss_index():
                return {
                    'error': 'Knowledge base not available. Please build the index first.',
                    'islamic_score': 0.0,
                    'authenticity_score': 0.0
                }
        
        # Generate embedding for input text
        try:
            query_embedding = self.embedding_model.encode([text])
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return {'error': f'Failed to process text: {e}'}
        
        # Search similar content
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Retrieve matching sources
        supporting_sources = []
        contradicting_sources = []
        
        for score, idx in zip(scores[0], indices[0]):
            if score < 0.3:  # Similarity threshold
                continue
            
            source_info = self.get_source_by_embedding_id(idx)
            if source_info:
                source_info['similarity_score'] = float(score)
                
                # Determine if supporting or contradicting
                if score > 0.7:
                    supporting_sources.append(source_info)
                elif score > 0.5:
                    # Analyze content for contradiction
                    if self.is_contradicting(text, source_info.get('content', '')):
                        contradicting_sources.append(source_info)
                    else:
                        supporting_sources.append(source_info)
        
        # Calculate scores
        islamic_score = self.calculate_islamic_score(text)
        authenticity_score = self.calculate_authenticity_score(supporting_sources, contradicting_sources)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(islamic_score, authenticity_score, 
                                                      supporting_sources, contradicting_sources)
        
        result = {
            'islamic_score': islamic_score,
            'authenticity_score': authenticity_score,
            'supporting_sources': supporting_sources,
            'contradicting_sources': contradicting_sources,
            'recommendations': recommendations,
            'cached': False
        }
        
        # Cache result
        try:
            cursor.execute("""
                INSERT INTO fact_check_results 
                (input_text, input_hash, islamic_score, authenticity_score, 
                 supporting_sources, contradicting_sources, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (text, text_hash, islamic_score, authenticity_score,
                  json.dumps(supporting_sources), json.dumps(contradicting_sources),
                  recommendations))
            self.db_connection.commit()
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        return result
    
    def get_source_by_embedding_id(self, embedding_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve source information by embedding ID"""
        cursor = self.db_connection.cursor()
        
        # Check Quran verses
        cursor.execute("""
            SELECT 'quran' as source_type, surah_number, verse_number, 
                   arabic_text, english_translation, transliteration
            FROM quran_verses WHERE embedding_id = ?
        """, (embedding_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                'type': 'quran',
                'surah': result['surah_number'],
                'verse': result['verse_number'],
                'arabic': result['arabic_text'],
                'translation': result['english_translation'],
                'transliteration': result['transliteration'],
                'content': result['english_translation'] or result['arabic_text'],
                'reference': f"Quran {result['surah_number']}:{result['verse_number']}"
            }
        
        # Check Hadith
        cursor.execute("""
            SELECT 'hadith' as source_type, collection, book, hadith_number,
                   arabic_text, english_text, narrator, grade
            FROM hadith WHERE embedding_id = ?
        """, (embedding_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                'type': 'hadith',
                'collection': result['collection'],
                'book': result['book'],
                'number': result['hadith_number'],
                'arabic': result['arabic_text'],
                'english': result['english_text'],
                'narrator': result['narrator'],
                'grade': result['grade'],
                'content': result['english_text'] or result['arabic_text'],
                'reference': f"{result['collection']} {result['book']}:{result['hadith_number']}"
            }
        
        return None
    
    def calculate_islamic_score(self, text: str) -> float:
        """Calculate how Islamic the content is"""
        text_lower = text.lower()
        
        # Count Islamic keywords
        total_keywords = 0
        found_keywords = 0
        
        for category, keywords in self.islamic_keywords.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword in text_lower:
                    found_keywords += 1
        
        # Base score from keyword presence
        keyword_score = found_keywords / total_keywords if total_keywords > 0 else 0
        
        # Boost for Islamic phrases
        islamic_phrases = [
            'in the name of allah', 'bismillah', 'alhamdulillah', 'subhanallah',
            'allahu akbar', 'la ilaha illa allah', 'prophet muhammad', 'peace be upon him',
            'may allah', 'insha allah', 'masha allah', 'astaghfirullah'
        ]
        
        phrase_boost = 0
        for phrase in islamic_phrases:
            if phrase in text_lower:
                phrase_boost += 0.1
        
        # Penalty for non-Islamic content
        non_islamic_terms = [
            'trinity', 'jesus is god', 'son of god', 'crucifixion for salvation',
            'polytheism', 'idol worship', 'multiple gods'
        ]
        
        penalty = 0
        for term in non_islamic_terms:
            if term in text_lower:
                penalty += 0.2
        
        final_score = min(1.0, max(0.0, keyword_score + phrase_boost - penalty))
        return final_score
    
    def calculate_authenticity_score(self, supporting_sources: List[Dict], 
                                   contradicting_sources: List[Dict]) -> float:
        """Calculate authenticity score based on sources"""
        if not supporting_sources and not contradicting_sources:
            return 0.5  # Neutral when no sources found
        
        # Weight sources by type and grade
        support_weight = 0
        contradict_weight = 0
        
        for source in supporting_sources:
            weight = 1.0
            
            if source['type'] == 'quran':
                weight = 2.0  # Quran has highest authority
            elif source['type'] == 'hadith':
                grade = source.get('grade', '').lower()
                if 'sahih' in grade:
                    weight = 1.8
                elif 'hasan' in grade:
                    weight = 1.5
                elif 'daif' in grade or 'weak' in grade:
                    weight = 0.8
            
            # Factor in similarity score
            similarity = source.get('similarity_score', 0.5)
            support_weight += weight * similarity
        
        for source in contradicting_sources:
            weight = 1.0
            
            if source['type'] == 'quran':
                weight = 2.0
            elif source['type'] == 'hadith':
                grade = source.get('grade', '').lower()
                if 'sahih' in grade:
                    weight = 1.8
                elif 'hasan' in grade:
                    weight = 1.5
                elif 'daif' in grade or 'weak' in grade:
                    weight = 0.8
            
            similarity = source.get('similarity_score', 0.5)
            contradict_weight += weight * similarity
        
        # Calculate final score
        total_weight = support_weight + contradict_weight
        if total_weight == 0:
            return 0.5
        
        authenticity_score = support_weight / total_weight
        return authenticity_score
    
    def is_contradicting(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection"""
        # This is a simplified implementation
        # In practice, you might want to use more sophisticated NLP
        
        contradiction_patterns = [
            (r'\bnot\b', r'\bis\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bforbidden\b', r'\ballowed\b'),
            (r'\bharam\b', r'\bhalal\b'),
            (r'\bwrong\b', r'\bright\b')
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for neg_pattern, pos_pattern in contradiction_patterns:
            if (re.search(neg_pattern, text1_lower) and re.search(pos_pattern, text2_lower)) or \
               (re.search(pos_pattern, text1_lower) and re.search(neg_pattern, text2_lower)):
                return True
        
        return False
    
    def generate_recommendations(self, islamic_score: float, authenticity_score: float,
                               supporting_sources: List[Dict], contradicting_sources: List[Dict]) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Islamic content recommendations
        if islamic_score < 0.3:
            recommendations.append("⚠️ Low Islamic content detected. Consider adding more Islamic context and references.")
        elif islamic_score < 0.6:
            recommendations.append("📝 Moderate Islamic content. Could benefit from additional Islamic references.")
        else:
            recommendations.append("✅ Good Islamic content with appropriate terminology and context.")
        
        # Authenticity recommendations
        if authenticity_score < 0.3:
            recommendations.append("❌ Content may contradict Islamic sources. Please verify against Quran and authentic Hadith.")
        elif authenticity_score < 0.6:
            recommendations.append("⚠️ Mixed evidence from Islamic sources. Consider reviewing and clarifying.")
        else:
            recommendations.append("✅ Content appears to align well with Islamic sources.")
        
        # Source-specific recommendations
        if supporting_sources:
            quran_sources = [s for s in supporting_sources if s['type'] == 'quran']
            hadith_sources = [s for s in supporting_sources if s['type'] == 'hadith']
            
            if quran_sources:
                recommendations.append(f"📖 Found {len(quran_sources)} supporting Quranic reference(s).")
            if hadith_sources:
                recommendations.append(f"📚 Found {len(hadith_sources)} supporting Hadith reference(s).")
        
        if contradicting_sources:
            recommendations.append(f"⚠️ Found {len(contradicting_sources)} potentially contradicting source(s). Please review carefully.")
        
        if not supporting_sources and not contradicting_sources:
            recommendations.append("ℹ️ No direct Islamic sources found. Consider adding references to Quran or authentic Hadith.")
        
        return " ".join(recommendations)
    
    def interactive_mode(self):
        """Interactive fact-checking mode"""
        print("\n" + "="*60)
        print("ISLAMIC CONTENT FACT CHECKER - INTERACTIVE MODE")
        print("="*60)
        print("Enter Islamic content to check (type 'quit' to exit)")
        print("Type 'help' for commands")
        print("")
        
        while True:
            try:
                user_input = input("\n📝 Enter content: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    continue
                
                print("\n🔍 Checking content...")
                result = self.check_content(user_input)
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information"""
        print("""
📋 COMMANDS:
  help     - Show this help
  quit     - Exit the program
  
📊 SCORING:
  Islamic Score    - How Islamic the content is (0.0 - 1.0)
  Authenticity     - How well it aligns with Islamic sources (0.0 - 1.0)
  
📖 SOURCES:
  Quran verses and authentic Hadith are used for fact-checking
  Higher similarity scores indicate stronger matches
        """)
    
    def display_result(self, result: Dict[str, Any]):
        """Display fact-check result"""
        print("\n" + "="*50)
        print("FACT-CHECK RESULT")
        print("="*50)
        
        # Scores
        islamic_score = result.get('islamic_score', 0)
        authenticity_score = result.get('authenticity_score', 0)
        
        print(f"📊 Islamic Content Score: {islamic_score:.2f}/1.00")
        print(f"📊 Authenticity Score: {authenticity_score:.2f}/1.00")
        
        if result.get('cached'):
            print("💾 (Cached result)")
        
        # Supporting sources
        supporting = result.get('supporting_sources', [])
        if supporting:
            print(f"\n✅ SUPPORTING SOURCES ({len(supporting)}):")
            for i, source in enumerate(supporting[:3], 1):  # Show top 3
                print(f"  {i}. {source['reference']} (similarity: {source.get('similarity_score', 0):.2f})")
                print(f"     {source['content'][:100]}...")
        
        # Contradicting sources
        contradicting = result.get('contradicting_sources', [])
        if contradicting:
            print(f"\n❌ CONTRADICTING SOURCES ({len(contradicting)}):")
            for i, source in enumerate(contradicting[:3], 1):  # Show top 3
                print(f"  {i}. {source['reference']} (similarity: {source.get('similarity_score', 0):.2f})")
                print(f"     {source['content'][:100]}...")
        
        # Recommendations
        recommendations = result.get('recommendations', '')
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   {recommendations}")
        
        print("="*50)
    
    def close(self):
        """Close database connection"""
        if self.db_connection:
            self.db_connection.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Islamic RAG Checker with SQLite+FAISS Backend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build knowledge base
  python enhanced_rag_checker.py --build-index --quran-path quran.json --hadith-path hadith.json
  
  # Check content from file
  python enhanced_rag_checker.py --check-content content.txt
  
  # Interactive mode
  python enhanced_rag_checker.py --interactive
  
  # Check specific text
  python enhanced_rag_checker.py --text "Prayer is the pillar of Islam"
"""
    )
    
    # Actions
    parser.add_argument('--build-index', action='store_true', help='Build knowledge base index')
    parser.add_argument('--check-content', type=str, help='Check content from file')
    parser.add_argument('--text', type=str, help='Check specific text')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    # Data sources
    parser.add_argument('--quran-path', type=str, help='Path to Quran JSON file')
    parser.add_argument('--hadith-path', type=str, help='Path to Hadith JSON file')
    
    # Configuration
    parser.add_argument('--db-path', type=str, default='islamic_knowledge.db', help='Database path')
    parser.add_argument('--index-path', type=str, default='islamic_index.faiss', help='FAISS index path')
    parser.add_argument('--top-k', type=int, default=5, help='Number of similar sources to retrieve')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize checker
        checker = IslamicRAGChecker(args.db_path, args.index_path)
        
        if args.build_index:
            # Build knowledge base
            if not args.quran_path and not args.hadith_path:
                print("Error: Please provide --quran-path and/or --hadith-path for building index")
                sys.exit(1)
            
            checker.build_knowledge_base(args.quran_path, args.hadith_path)
            print("✅ Knowledge base built successfully!")
        
        elif args.check_content:
            # Check content from file
            if not os.path.exists(args.check_content):
                print(f"Error: File not found: {args.check_content}")
                sys.exit(1)
            
            with open(args.check_content, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = checker.check_content(content, args.top_k)
            checker.display_result(result)
        
        elif args.text:
            # Check specific text
            result = checker.check_content(args.text, args.top_k)
            checker.display_result(result)
        
        elif args.interactive:
            # Interactive mode
            checker.interactive_mode()
        
        else:
            parser.print_help()
        
        # Cleanup
        checker.close()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()