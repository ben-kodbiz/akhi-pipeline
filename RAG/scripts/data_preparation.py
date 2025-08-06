#!/usr/bin/env python3
"""
Data Preparation Script for Islamic Knowledge Base
Collects and preprocesses Islamic texts for RAG pipeline.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import requests
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IslamicDataCollector:
    """Collects and preprocesses Islamic knowledge sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Arabic text
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF.,!?;:()\[\]"\'-]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        return text.strip()
    
    def create_sample_quran_data(self) -> None:
        """Create sample Quran data for testing"""
        logger.info("Creating sample Quran data...")
        
        # Sample Quran verses and facts
        quran_content = """
# The Holy Quran - Sample Content

## Basic Facts about the Quran

The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him) over a period of approximately 23 years.

The Quran contains 114 chapters, called Surahs. The chapters vary in length, with the longest being Al-Baqarah (The Cow) with 286 verses.

The Quran was revealed in Arabic and is considered by Muslims to be the direct word of Allah (God).

The first revelation came to Prophet Muhammad in the cave of Hira through the Angel Gabriel (Jibril in Arabic).

## Chapter 1: Al-Fatiha (The Opening)

In the name of Allah, the Most Gracious, the Most Merciful.
Praise be to Allah, the Lord of all the worlds.
The Most Gracious, the Most Merciful.
Master of the Day of Judgment.
You alone we worship, and You alone we ask for help.
Guide us to the straight path.
The path of those You have blessed, not of those who have incurred Your wrath, nor of those who have gone astray.

## Chapter 2: Al-Baqarah (The Cow) - Selected Verses

This is the Book in which there is no doubt, a guidance for the righteous.
Who believe in the unseen, and who establish prayer, and spend out of what We have provided for them.
And who believe in what has been revealed to you, and what was revealed before you, and who are certain of the Hereafter.

## Chapter 112: Al-Ikhlas (The Sincerity)

Say: He is Allah, the One.
Allah, the Eternal, Absolute.
He begets not, nor is He begotten.
And there is none like unto Him.

## Key Themes in the Quran

The Quran emphasizes the oneness of Allah (Tawhid), which is the central concept in Islam.
It provides guidance for personal conduct, social justice, and spiritual development.
The Quran contains stories of previous prophets including Adam, Noah, Abraham, Moses, and Jesus.
It emphasizes the importance of prayer (Salah), charity (Zakat), fasting (Sawm), and pilgrimage (Hajj).

## Compilation and Preservation

The Quran was compiled into a single book during the caliphate of Abu Bakr and standardized during the caliphate of Uthman ibn Affan.
Muslims believe the Quran has been perfectly preserved since its revelation.
The Quran is memorized by millions of Muslims worldwide, known as Huffaz.
"""
        
        # Write to file
        quran_file = self.data_dir / "quran.txt"
        with open(quran_file, 'w', encoding='utf-8') as f:
            f.write(self.clean_text(quran_content))
        
        logger.info(f"Sample Quran data saved to {quran_file}")
    
    def create_sample_hadith_data(self) -> None:
        """Create sample Hadith data for testing"""
        logger.info("Creating sample Hadith data...")
        
        # Sample Hadith content
        hadith_content = """
# Hadith Collection - Sample Content

## About Hadith

Hadith are the recorded sayings, actions, and approvals of Prophet Muhammad (peace be upon him).
They serve as the second source of Islamic law and guidance after the Quran.
The most authentic collections are known as Sahih al-Bukhari and Sahih Muslim.

## Sahih al-Bukhari - Selected Hadith

### Hadith 1: The Importance of Intention
Narrated by Umar ibn al-Khattab:
The Prophet (peace be upon him) said: "Actions are but by intention, and every man shall have only that which he intended."

### Hadith 2: The Five Pillars
Narrated by Ibn Umar:
The Prophet (peace be upon him) said: "Islam is built upon five pillars: testifying that there is no god but Allah and that Muhammad is His messenger, establishing prayer, giving charity, fasting in Ramadan, and performing pilgrimage to the House for whoever is able."

### Hadith 3: Seeking Knowledge
Narrated by Anas ibn Malik:
The Prophet (peace be upon him) said: "Seek knowledge from the cradle to the grave." And he also said: "Seek knowledge, even if you have to go to China."

### Hadith 4: Treatment of Parents
Narrated by Abu Hurairah:
A man came to the Prophet (peace be upon him) and asked: "Who is most deserving of my good treatment?" He replied: "Your mother." The man asked: "Then who?" He replied: "Your mother." The man asked again: "Then who?" He replied: "Your mother." The man asked once more: "Then who?" He replied: "Your father."

## Sahih Muslim - Selected Hadith

### Hadith 5: The Golden Rule
Narrated by Anas:
The Prophet (peace be upon him) said: "None of you believes until he loves for his brother what he loves for himself."

### Hadith 6: Cleanliness
Narrated by Abu Malik al-Ash'ari:
The Prophet (peace be upon him) said: "Cleanliness is half of faith."

### Hadith 7: Charity
Narrated by Abu Hurairah:
The Prophet (peace be upon him) said: "Every day the sun rises, charity is due on every joint of a person: to judge justly between people is charity, to help a man with his mount and lift his luggage onto it is charity, a good word is charity, every step you take towards prayer is charity, and removing harmful things from the road is charity."

## Categories of Hadith

Sahih: Authentic hadith with reliable chain of narrators.
Hasan: Good hadith with acceptable chain of narrators.
Da'if: Weak hadith with questionable chain of narrators.

## The Science of Hadith

Islamic scholars developed detailed methodologies to verify the authenticity of hadith.
The chain of narrators (isnad) and the text (matn) are both carefully examined.
Famous hadith scholars include Imam Bukhari, Imam Muslim, and Imam Ahmad ibn Hanbal.
"""
        
        # Write to file
        hadith_file = self.data_dir / "hadith.txt"
        with open(hadith_file, 'w', encoding='utf-8') as f:
            f.write(self.clean_text(hadith_content))
        
        logger.info(f"Sample Hadith data saved to {hadith_file}")
    
    def create_sample_qa_data(self) -> None:
        """Create sample Islamic Q&A data for testing"""
        logger.info("Creating sample Islamic Q&A data...")
        
        # Sample Q&A content
        qa_content = """
# Islamic Questions and Answers - Sample Content

## Basic Islamic Knowledge

### Question: How many chapters are in the Quran?
Answer: The Quran contains 114 chapters, called Surahs.

### Question: What are the five pillars of Islam?
Answer: The five pillars of Islam are: 1) Shahada (declaration of faith), 2) Salah (prayer), 3) Zakat (charity), 4) Sawm (fasting during Ramadan), and 5) Hajj (pilgrimage to Mecca).

### Question: Who was the first Caliph after Prophet Muhammad?
Answer: Abu Bakr al-Siddiq was the first Caliph after Prophet Muhammad (peace be upon him).

### Question: What is the meaning of Shahada?
Answer: Shahada is the Islamic declaration of faith, stating "There is no god but Allah, and Muhammad is His messenger" (La ilaha illa Allah, Muhammad rasul Allah).

### Question: In which month do Muslims fast?
Answer: Muslims fast during the month of Ramadan, the ninth month of the Islamic lunar calendar.

### Question: What is the Kaaba?
Answer: The Kaaba is the sacred cube-shaped building in Mecca, Saudi Arabia, which Muslims face during prayer and circumambulate during Hajj and Umrah.

### Question: How many times a day do Muslims pray?
Answer: Muslims pray five times a day: Fajr (dawn), Dhuhr (midday), Asr (afternoon), Maghrib (sunset), and Isha (night).

### Question: What is Zakat?
Answer: Zakat is the obligatory charity that Muslims must give, typically 2.5% of their wealth annually, to help the poor and needy.

### Question: What is the Night of Power (Laylat al-Qadr)?
Answer: Laylat al-Qadr is the night when the first verses of the Quran were revealed to Prophet Muhammad. It occurs during the last ten nights of Ramadan and is considered better than a thousand months.

### Question: What is the difference between Hajj and Umrah?
Answer: Hajj is the major pilgrimage to Mecca that occurs during specific dates in the Islamic calendar and is obligatory for those who are able. Umrah is the minor pilgrimage that can be performed at any time of the year.

## Islamic History

### Question: When was Prophet Muhammad born?
Answer: Prophet Muhammad (peace be upon him) was born in approximately 570 CE in Mecca.

### Question: What was the first mosque built by Muslims?
Answer: The first mosque built by Muslims was Masjid Quba in Medina, built by Prophet Muhammad upon his arrival in Medina.

### Question: What is the Hijra?
Answer: The Hijra refers to the migration of Prophet Muhammad and his followers from Mecca to Medina in 622 CE, which marks the beginning of the Islamic calendar.

### Question: Who were the four Rightly-Guided Caliphs?
Answer: The four Rightly-Guided Caliphs were Abu Bakr, Umar ibn al-Khattab, Uthman ibn Affan, and Ali ibn Abi Talib.

## Islamic Practices

### Question: What is Wudu?
Answer: Wudu is the ritual washing performed by Muslims before prayer, involving washing the hands, mouth, nose, face, arms, head, and feet.

### Question: What is the Qibla?
Answer: The Qibla is the direction that Muslims face when praying, which is towards the Kaaba in Mecca.

### Question: What is Sunnah?
Answer: Sunnah refers to the practices, sayings, and approvals of Prophet Muhammad, which serve as a model for Muslim behavior.

### Question: What is Iftar?
Answer: Iftar is the meal eaten by Muslims to break their fast at sunset during Ramadan.

### Question: What is Tawhid?
Answer: Tawhid is the fundamental Islamic concept of the oneness and uniqueness of Allah (God).

### Question: What is Jihad?
Answer: Jihad literally means "struggle" or "effort" and refers to the spiritual struggle against sin and the effort to live according to Islamic principles. It can also refer to defending Islam when under attack.
"""
        
        # Write to file
        qa_file = self.data_dir / "islamic_qa.txt"
        with open(qa_file, 'w', encoding='utf-8') as f:
            f.write(self.clean_text(qa_content))
        
        logger.info(f"Sample Islamic Q&A data saved to {qa_file}")
    
    def create_evaluation_dataset(self) -> None:
        """Create evaluation dataset for RAGAS testing"""
        logger.info("Creating evaluation dataset...")
        
        # Evaluation questions with ground truth answers
        eval_data = {
            "questions": [
                "How many chapters are in the Quran?",
                "What are the five pillars of Islam?",
                "Who was the first Caliph after Prophet Muhammad?",
                "What is the meaning of Shahada?",
                "In which month do Muslims fast?",
                "How many times a day do Muslims pray?",
                "What is Zakat?",
                "What is the Kaaba?",
                "What is Wudu?",
                "What is Tawhid?"
            ],
            "ground_truth": [
                "The Quran contains 114 chapters, called Surahs.",
                "The five pillars of Islam are: Shahada (declaration of faith), Salah (prayer), Zakat (charity), Sawm (fasting during Ramadan), and Hajj (pilgrimage to Mecca).",
                "Abu Bakr al-Siddiq was the first Caliph after Prophet Muhammad.",
                "Shahada is the Islamic declaration of faith, stating 'There is no god but Allah, and Muhammad is His messenger'.",
                "Muslims fast during the month of Ramadan.",
                "Muslims pray five times a day: Fajr, Dhuhr, Asr, Maghrib, and Isha.",
                "Zakat is the obligatory charity that Muslims must give, typically 2.5% of their wealth annually.",
                "The Kaaba is the sacred cube-shaped building in Mecca that Muslims face during prayer and circumambulate during pilgrimage.",
                "Wudu is the ritual washing performed by Muslims before prayer.",
                "Tawhid is the fundamental Islamic concept of the oneness and uniqueness of Allah."
            ],
            "contexts": [
                "Basic facts about the Quran",
                "Islamic pillars and practices",
                "Islamic history and caliphate",
                "Islamic declarations and beliefs",
                "Islamic calendar and fasting",
                "Islamic prayer times",
                "Islamic charity and obligations",
                "Islamic sacred places",
                "Islamic ritual practices",
                "Islamic theological concepts"
            ]
        }
        
        # Save evaluation dataset
        eval_file = self.data_dir / "evaluation_dataset.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation dataset saved to {eval_file}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the collected data"""
        stats = {
            "files": [],
            "total_characters": 0,
            "total_words": 0,
            "estimated_chunks": 0
        }
        
        for file_path in self.data_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            char_count = len(content)
            word_count = len(content.split())
            estimated_chunks = max(1, char_count // 200)  # Assuming 200 char chunks
            
            file_stats = {
                "name": file_path.name,
                "characters": char_count,
                "words": word_count,
                "estimated_chunks": estimated_chunks
            }
            
            stats["files"].append(file_stats)
            stats["total_characters"] += char_count
            stats["total_words"] += word_count
            stats["estimated_chunks"] += estimated_chunks
        
        return stats
    
    def prepare_all_data(self) -> None:
        """Prepare all Islamic knowledge data"""
        logger.info("Starting data preparation...")
        
        # Create sample data
        self.create_sample_quran_data()
        self.create_sample_hadith_data()
        self.create_sample_qa_data()
        self.create_evaluation_dataset()
        
        # Get statistics
        stats = self.get_data_statistics()
        
        logger.info("Data preparation completed!")
        logger.info(f"Total files: {len(stats['files'])}")
        logger.info(f"Total characters: {stats['total_characters']:,}")
        logger.info(f"Total words: {stats['total_words']:,}")
        logger.info(f"Estimated chunks: {stats['estimated_chunks']:,}")
        
        # Save statistics
        stats_file = self.data_dir / "data_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")

def main():
    """Main function for data preparation"""
    # Initialize data collector
    collector = IslamicDataCollector("../data")
    
    # Prepare all data
    collector.prepare_all_data()
    
    print("\nData preparation completed successfully!")
    print("\nFiles created:")
    print("- data/quran.txt")
    print("- data/hadith.txt")
    print("- data/islamic_qa.txt")
    print("- data/evaluation_dataset.json")
    print("- data/data_statistics.json")
    
    print("\nNext steps:")
    print("1. Review the generated data files")
    print("2. Add more authentic Islamic texts if needed")
    print("3. Run the RAG pipeline to build the vector index")
    print("4. Test the system with sample queries")

if __name__ == "__main__":
    main()