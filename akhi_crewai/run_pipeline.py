#!/usr/bin/env python3
"""
QLoRA Pipeline Orchestrator
Unified entry point for the complete Islamic AI training pipeline

Usage:
    python run_pipeline.py --query "mufti menk trauma" --max-videos 10
    python run_pipeline.py --urls urls.txt --output ./custom_output
    python run_pipeline.py --config pipeline_config.yaml

Author: Akhi CrewAI Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import yaml
import argparse
import logging
import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from tools.youtube_downloader import YouTubeDownloaderTool
from tools.youtube_search import YouTubeSearchTool
from tools.transcriber import TranscriptionTool
from tools.summarizer import SummarizerTool
from tools.qlora_formatter import QLoRAFormatterTool
from tools.axolotl_trainer import AxolotlTrainerTool
from agents.content_qa import ContentQAAgent
from agents.vector_indexer import VectorIndexerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QLoRAPipelineOrchestrator:
    """Main orchestrator for the QLoRA training pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './pipeline_output'))
        self.setup_directories()
        
        # Initialize tools
        self.youtube_search = YouTubeSearchTool()
        self.youtube_downloader = YouTubeDownloaderTool()
        self.transcriber = TranscriptionTool()
        self.summarizer = SummarizerTool()
        self.qlora_formatter = QLoRAFormatterTool()
        self.axolotl_trainer = AxolotlTrainerTool()
        
        # Initialize agents
        self.content_qa = ContentQAAgent()
        self.vector_indexer = VectorIndexerAgent()
        
        # Pipeline state
        self.state = {
            'stage': 'initialized',
            'processed_videos': 0,
            'total_videos': 0,
            'start_time': datetime.now().isoformat(),
            'errors': [],
            'outputs': {}
        }
        
        logger.info(f"Pipeline orchestrator initialized with output dir: {self.output_dir}")
    
    def setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            'videos', 'audio', 'transcripts', 'summaries', 
            'jsonl', 'datasets', 'configs', 'logs', 'models'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directories in {self.output_dir}")
    
    def check_existing_audio(self, video_urls: List[str]) -> Dict[str, Any]:
        """Check for existing audio files to avoid re-downloading"""
        audio_dir = self.output_dir / 'audio'
        existing_files = {}
        new_urls = []
        
        for url in video_urls:
            # Create a hash of the URL to use as filename identifier
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            
            # Check for existing audio files with this hash
            audio_files = list(audio_dir.glob(f"*{url_hash}*.mp3"))
            if audio_files:
                existing_files[url] = {
                    'audio_path': str(audio_files[0]),
                    'title': audio_files[0].stem,
                    'url': url,
                    'video_id': url.split('v=')[-1] if 'v=' in url else url_hash
                }
                logger.info(f"Found existing audio for: {url}")
            else:
                new_urls.append(url)
        
        return {
            'existing_files': list(existing_files.values()),
            'new_urls': new_urls,
            'skipped_count': len(existing_files)
        }
    
    def check_existing_transcripts(self, audio_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for existing transcripts to avoid re-transcribing"""
        transcript_dir = self.output_dir / 'transcripts'
        existing_transcripts = []
        files_to_transcribe = []
        
        for file_info in audio_files:
            audio_path = file_info.get('audio_path', '')
            audio_name = Path(audio_path).stem if audio_path else ''
            
            # Look for existing transcript files
            transcript_files = list(transcript_dir.glob(f"{audio_name}*.txt"))
            if transcript_files:
                existing_transcripts.append({
                    **file_info,
                    'transcript_path': str(transcript_files[0]),
                    'success': True
                })
                logger.info(f"Found existing transcript for: {audio_name}")
            else:
                files_to_transcribe.append(file_info)
        
        return {
            'existing_transcripts': existing_transcripts,
            'files_to_transcribe': files_to_transcribe,
            'skipped_count': len(existing_transcripts)
        }
    
    def check_existing_qlora_data(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for existing QLoRA formatted data"""
        jsonl_dir = self.output_dir / 'jsonl'
        existing_qlora = []
        summaries_to_format = []
        
        for summary_info in summaries:
            summary_path = summary_info.get('summary_path', '')
            summary_name = Path(summary_path).stem if summary_path else ''
            
            # Look for existing QLoRA files
            qlora_files = list(jsonl_dir.glob(f"{summary_name}*.jsonl"))
            if qlora_files:
                existing_qlora.append({
                    **summary_info,
                    'output_path': str(qlora_files[0]),
                    'success': True
                })
                logger.info(f"Found existing QLoRA data for: {summary_name}")
            else:
                summaries_to_format.append(summary_info)
        
        return {
            'existing_qlora': existing_qlora,
            'summaries_to_format': summaries_to_format,
            'skipped_count': len(existing_qlora)
        }
    
    def deduplicate_qlora_dataset(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entries from QLoRA dataset"""
        seen_hashes: Set[str] = set()
        deduplicated = []
        
        for sample in all_samples:
            # Create hash based on instruction and input content
            content = sample.get('instruction', '') + sample.get('input', '') + sample.get('output', '')
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(sample)
            else:
                logger.debug(f"Removed duplicate sample with hash: {content_hash[:12]}...")
        
        removed_count = len(all_samples) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate samples from dataset")
        
        return deduplicated
    
    async def run_pipeline(self, query: Optional[str] = None, urls: Optional[List[str]] = None, duration: str = 'any') -> Dict[str, Any]:
        """Execute the complete pipeline"""
        try:
            logger.info("Starting QLoRA Pipeline Orchestration")
            
            # Stage 1: Search/Collect URLs
            video_urls = await self.stage_1_search_collect(query, urls, duration)
            
            # Stage 2: Download Videos
            downloaded_files = await self.stage_2_download(video_urls)
            
            # Stage 3: Transcribe Audio
            transcripts = await self.stage_3_transcribe(downloaded_files)
            
            # Stage 4: Summarize Content
            summaries = await self.stage_4_summarize(transcripts)
            
            # Stage 5: Format for QLoRA
            qlora_data = await self.stage_5_format_qlora(summaries)
            
            # Stage 6: Prepare Axolotl Dataset
            dataset_info = await self.stage_6_prepare_dataset(qlora_data)
            
            # Stage 7: Generate Axolotl Config
            config_info = await self.stage_7_generate_config(dataset_info)
            
            # Generate final report
            report = self.generate_report(config_info)
            
            logger.info("Pipeline completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.state['errors'].append({
                'stage': self.state['stage'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def stage_1_search_collect(self, query: Optional[str], urls: Optional[List[str]], duration: str = 'any') -> List[str]:
        """Stage 1: Search YouTube or use provided URLs"""
        logger.info("Stage 1: Searching/Collecting video URLs")
        self.state['stage'] = 'search_collect'
        
        video_urls = []
        
        if urls:
            # Use provided URLs
            video_urls = urls
            logger.info(f"Using {len(urls)} provided URLs")
        elif query:
            # Search YouTube
            max_results = self.config.get('max_videos', 10)
            search_filters = self.config.get('search_filters', {
                'duration': duration,  # Use provided duration filter
                'type': 'video',
                'features': ['subtitles']
            })
            
            search_result = await self.youtube_search.search_videos(
                query=query,
                max_results=max_results,
                filters=search_filters
            )
            
            if search_result['success']:
                video_urls = [video['url'] for video in search_result['videos']]
                logger.info(f"Found {len(video_urls)} videos for query: {query}")
            else:
                raise Exception(f"Search failed: {search_result.get('error', 'Unknown error')}")
        else:
            raise Exception("Either query or urls must be provided")
        
        self.state['total_videos'] = len(video_urls)
        self.state['outputs']['video_urls'] = video_urls
        
        # Save URLs for reference
        urls_file = self.output_dir / 'logs' / 'video_urls.txt'
        with open(urls_file, 'w') as f:
            for url in video_urls:
                f.write(f"{url}\n")
        
        return video_urls
    
    async def stage_2_download(self, video_urls: List[str]) -> List[Dict[str, Any]]:
        """Stage 2: Download videos and extract audio (with existing file checks)"""
        logger.info("Stage 2: Downloading videos and extracting audio")
        self.state['stage'] = 'download'
        
        # Check for existing audio files first
        audio_check = self.check_existing_audio(video_urls)
        existing_files = audio_check['existing_files']
        new_urls = audio_check['new_urls']
        
        logger.info(f"Found {audio_check['skipped_count']} existing audio files, downloading {len(new_urls)} new files")
        
        downloaded_files = list(existing_files)  # Start with existing files
        
        for i, url in enumerate(new_urls):
            try:
                logger.info(f"Downloading video {i+1}/{len(new_urls)}: {url}")
                
                result = await self.youtube_downloader.download_video(
                    url=url,
                    output_dir=str(self.output_dir / 'videos'),
                    extract_audio=True,
                    audio_dir=str(self.output_dir / 'audio'),
                    quality='best[height<=720]'  # Limit quality for faster download
                )
                
                if result['success']:
                    downloaded_files.append(result)
                    self.state['processed_videos'] += 1
                    logger.info(f"Downloaded: {result['title']}")
                else:
                    logger.warning(f"Failed to download: {url} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                self.state['errors'].append({
                    'stage': 'download',
                    'url': url,
                    'error': str(e)
                })
        
        self.state['outputs']['downloaded_files'] = downloaded_files
        
        # Save download report
        download_report = {
            'total_attempted': len(video_urls),
            'existing_files': len(existing_files),
            'new_downloads': len(downloaded_files) - len(existing_files),
            'total_available': len(downloaded_files),
            'success_rate': len(downloaded_files) / len(video_urls) * 100,
            'files': downloaded_files
        }
        
        with open(self.output_dir / 'logs' / 'download_report.json', 'w') as f:
            json.dump(download_report, f, indent=2)
        
        logger.info(f"Total files available: {len(downloaded_files)}/{len(video_urls)} (existing: {len(existing_files)}, new: {len(downloaded_files) - len(existing_files)})")
        return downloaded_files
    
    async def stage_3_transcribe(self, downloaded_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Transcribe audio files (with existing transcript checks)"""
        logger.info("Stage 3: Transcribing audio files")
        self.state['stage'] = 'transcribe'
        
        # Check for existing transcripts first
        transcript_check = self.check_existing_transcripts(downloaded_files)
        existing_transcripts = transcript_check['existing_transcripts']
        files_to_transcribe = transcript_check['files_to_transcribe']
        
        logger.info(f"Found {transcript_check['skipped_count']} existing transcripts, transcribing {len(files_to_transcribe)} new files")
        
        transcripts = list(existing_transcripts)  # Start with existing transcripts
        
        for file_info in files_to_transcribe:
            try:
                audio_path = file_info.get('audio_path')
                if not audio_path or not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue
                
                logger.info(f"Transcribing: {os.path.basename(audio_path)}")
                
                result = await self.transcriber.transcribe_audio(
                    audio_path=audio_path,
                    output_dir=str(self.output_dir / 'transcripts'),
                    model_size=self.config.get('whisper_model', 'base'),
                    device='cpu',  # Force CPU to avoid CUDA issues
                    language='auto'
                )
                
                if result['success']:
                    # Add metadata
                    result['video_info'] = file_info
                    transcripts.append(result)
                    logger.info(f"Transcribed: {result['transcript_path']}")
                else:
                    logger.warning(f"Transcription failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error transcribing {file_info.get('title', 'unknown')}: {str(e)}")
                self.state['errors'].append({
                    'stage': 'transcribe',
                    'file': file_info.get('title', 'unknown'),
                    'error': str(e)
                })
        
        self.state['outputs']['transcripts'] = transcripts
        
        # Save transcription report
        transcription_report = {
            'total_files': len(downloaded_files),
            'existing_transcripts': len(existing_transcripts),
            'new_transcriptions': len(transcripts) - len(existing_transcripts),
            'total_available': len(transcripts),
            'success_rate': len(transcripts) / len(downloaded_files) * 100 if downloaded_files else 0,
            'transcripts': transcripts
        }
        
        with open(self.output_dir / 'logs' / 'transcription_report.json', 'w') as f:
            json.dump(transcription_report, f, indent=2)
        
        logger.info(f"Total transcripts available: {len(transcripts)}/{len(downloaded_files)} (existing: {len(existing_transcripts)}, new: {len(transcripts) - len(existing_transcripts)})")
        return transcripts
    
    async def stage_4_summarize(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Summarize and extract key content"""
        logger.info("Stage 4: Summarizing content")
        self.state['stage'] = 'summarize'
        
        summaries = []
        
        for transcript_info in transcripts:
            try:
                transcript_path = transcript_info.get('transcript_path')
                if not transcript_path or not os.path.exists(transcript_path):
                    logger.warning(f"Transcript file not found: {transcript_path}")
                    continue
                
                logger.info(f"Summarizing: {os.path.basename(transcript_path)}")
                
                # Quality check first - read transcript content
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_content = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read transcript {transcript_path}: {e}")
                    continue
                
                qa_result = self.content_qa.assess_content(
                    content=transcript_content,
                    content_type="transcript"
                )
                
                # Only summarize high-quality Islamic content
                quality_threshold = self.config.get('quality_threshold', 0.7)
                if qa_result.get('quality_score', 0) < quality_threshold:
                    logger.info(f"Skipping low-quality content: {os.path.basename(transcript_path)}")
                    continue
                
                # Use SummarizerTool's _run method
                result_str = self.summarizer._run(
                    text=transcript_content,
                    summary_length='medium',
                    strategy='abstractive',
                    preserve_islamic_terms=True,
                    include_key_points=True
                )
                
                # Parse the result (it returns a JSON string)
                import json
                try:
                    result = json.loads(result_str)
                except json.JSONDecodeError:
                    result = {'success': False, 'error': 'Failed to parse summarizer output'}
                
                # Save summary to file if successful
                if result.get('success', False):
                    summary_filename = os.path.basename(transcript_path).replace('.txt', '_summary.txt')
                    summary_path = self.output_dir / 'summaries' / summary_filename
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(result.get('summary', ''))
                    result['summary_path'] = str(summary_path)
                
                if result.get('success', False):
                    # Add metadata
                    result['transcript_info'] = transcript_info
                    result['quality_assessment'] = qa_result
                    summaries.append(result)
                    logger.info(f"Summarized: {result['summary_path']}")
                else:
                    logger.warning(f"Summarization failed: {result.get('error', 'Unknown error')}")
                    logger.warning(f"Full result: {result}")
                    
            except Exception as e:
                logger.error(f"Error summarizing {transcript_info.get('transcript_path', 'unknown')}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.state['errors'].append({
                    'stage': 'summarize',
                    'file': transcript_info.get('transcript_path', 'unknown'),
                    'error': str(e)
                })
        
        self.state['outputs']['summaries'] = summaries
        
        # Save summarization report
        summary_report = {
            'total_transcripts': len(transcripts),
            'successful_summaries': len(summaries),
            'success_rate': len(summaries) / len(transcripts) * 100 if transcripts else 0,
            'summaries': summaries
        }
        
        with open(self.output_dir / 'logs' / 'summary_report.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Summarized {len(summaries)}/{len(transcripts)} transcripts successfully")
        return summaries
    
    async def stage_5_format_qlora(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 5: Format content for QLoRA training (with existing data checks)"""
        logger.info("Stage 5: Formatting content for QLoRA")
        self.state['stage'] = 'format_qlora'
        
        # Check for existing QLoRA data first
        qlora_check = self.check_existing_qlora_data(summaries)
        existing_qlora = qlora_check['existing_qlora']
        summaries_to_format = qlora_check['summaries_to_format']
        
        logger.info(f"Found {qlora_check['skipped_count']} existing QLoRA files, formatting {len(summaries_to_format)} new files")
        
        qlora_data = list(existing_qlora)  # Start with existing QLoRA data
        
        for summary_info in summaries_to_format:
            try:
                summary_path = summary_info.get('summary_path')
                if not summary_path or not os.path.exists(summary_path):
                    logger.warning(f"Summary file not found: {summary_path}")
                    continue
                
                logger.info(f"Formatting for QLoRA: {os.path.basename(summary_path)}")
                
                result_str = self.qlora_formatter._run(
                    transcript_dir=str(self.output_dir / 'summaries'),
                    output_file=str(self.output_dir / 'jsonl' / f'{os.path.basename(summary_path).replace(".txt", ".jsonl")}'),
                    min_segment_words=50,
                    max_segment_words=500,
                    include_metadata=True,
                    filter_islamic_content=True
                )
                
                # Parse the result (it returns a JSON string)
                try:
                    result = json.loads(result_str)
                except json.JSONDecodeError:
                    result = {'success': False, 'error': 'Failed to parse QLoRA formatter output'}
                
                if result['success']:
                    # Add metadata
                    result['summary_info'] = summary_info
                    qlora_data.append(result)
                    logger.info(f"Formatted: {result['output_path']} ({result.get('samples_generated', 0)} samples)")
                else:
                    logger.warning(f"QLoRA formatting failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error formatting {summary_info.get('summary_path', 'unknown')}: {str(e)}")
                self.state['errors'].append({
                    'stage': 'format_qlora',
                    'file': summary_info.get('summary_path', 'unknown'),
                    'error': str(e)
                })
        
        self.state['outputs']['qlora_data'] = qlora_data
        
        # Save formatting report
        formatting_report = {
            'total_summaries': len(summaries),
            'existing_qlora': len(existing_qlora),
            'new_formatting': len(qlora_data) - len(existing_qlora),
            'total_available': len(qlora_data),
            'success_rate': len(qlora_data) / len(summaries) * 100 if summaries else 0,
            'total_samples': sum(item.get('samples_generated', 0) for item in qlora_data),
            'formatted_files': qlora_data
        }
        
        with open(self.output_dir / 'logs' / 'formatting_report.json', 'w') as f:
            json.dump(formatting_report, f, indent=2)
        
        logger.info(f"Total QLoRA files available: {len(qlora_data)}/{len(summaries)} (existing: {len(existing_qlora)}, new: {len(qlora_data) - len(existing_qlora)})")
        return qlora_data
    
    async def stage_6_prepare_dataset(self, qlora_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 6: Merge QLoRA files into training dataset"""
        logger.info("Stage 6: Preparing final training dataset")
        self.state['stage'] = 'prepare_dataset'
        
        try:
            # Collect all formatted data
            all_samples = []
            
            for qlora_info in qlora_data:
                output_path = qlora_info.get('output_path')
                if output_path and os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        if output_path.endswith('.jsonl'):
                            # JSONL format
                            for line in f:
                                line = line.strip()
                                if line:
                                    all_samples.append(json.loads(line))
                        else:
                            # JSON format
                            data = json.load(f)
                            if isinstance(data, list):
                                all_samples.extend(data)
                            else:
                                all_samples.append(data)
            
            # Deduplicate the dataset
            logger.info(f"Deduplicating dataset: {len(all_samples)} samples before deduplication")
            all_samples = self.deduplicate_qlora_dataset(all_samples)
            logger.info(f"After deduplication: {len(all_samples)} samples")
            
            # Split into train/validation
            val_split = self.config.get('validation_split', 0.1)
            val_size = int(len(all_samples) * val_split)
            
            train_samples = all_samples[val_size:]
            val_samples = all_samples[:val_size]
            
            # Save training dataset
            train_path = self.output_dir / 'datasets' / 'train_dataset.jsonl'
            with open(train_path, 'w', encoding='utf-8') as f:
                for sample in train_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Save validation dataset
            val_path = self.output_dir / 'datasets' / 'val_dataset.jsonl'
            with open(val_path, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            dataset_info = {
                'train_path': str(train_path),
                'val_path': str(val_path),
                'train_samples': len(train_samples),
                'val_samples': len(val_samples),
                'total_samples': len(all_samples),
                'validation_split': val_split
            }
            
            self.state['outputs']['dataset_info'] = dataset_info
            
            # Save dataset info
            with open(self.output_dir / 'logs' / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info(f"Prepared dataset: {len(train_samples)} train, {len(val_samples)} val samples")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            self.state['errors'].append({
                'stage': 'prepare_dataset',
                'error': str(e)
            })
            raise
    
    async def stage_7_generate_config(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 7: Generate Axolotl configuration"""
        logger.info("Stage 7: Generating Axolotl configuration")
        self.state['stage'] = 'generate_config'
        
        try:
            # Generate Axolotl config
            axolotl_config = {
                'base_model': self.config.get('base_model', 'microsoft/DialoGPT-medium'),
                'model_type': 'AutoModelForCausalLM',
                'tokenizer_type': 'AutoTokenizer',
                
                'load_in_8bit': False,
                'load_in_4bit': True,
                'strict': False,
                
                'datasets': [
                    {
                        'path': dataset_info['train_path'],
                        'type': self.config.get('dataset_type', 'alpaca')
                    }
                ],
                
                'val_set_size': dataset_info['val_samples'],
                'output_dir': str(self.output_dir / 'models'),
                
                'sequence_len': self.config.get('max_sequence_length', 2048),
                'sample_packing': True,
                'pad_to_sequence_len': True,
                
                'adapter': 'qlora',
                'lora_r': self.config.get('lora_r', 16),
                'lora_alpha': self.config.get('lora_alpha', 32),
                'lora_dropout': self.config.get('lora_dropout', 0.05),
                'lora_target_modules': [
                    'q_proj', 'v_proj', 'k_proj', 'o_proj',
                    'gate_proj', 'down_proj', 'up_proj'
                ],
                
                # 'wandb_project': f"akhi-islamic-ai-{datetime.now().strftime('%Y%m%d')}",
            # 'wandb_name': f"islamic-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                
                'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 4),
                'micro_batch_size': self.config.get('batch_size', 4),
                'num_epochs': self.config.get('num_epochs', 3),
                'learning_rate': self.config.get('learning_rate', 2e-4),
                
                'optimizer': 'adamw_bnb_8bit',
                'lr_scheduler': 'cosine',
                'warmup_steps': 10,
                
                'bf16': True,
                'fp16': False,
                'gradient_checkpointing': True,
                'flash_attention': True,
                
                'logging_steps': 1,
                'saves_per_epoch': 1,
                'evals_per_epoch': 4
            }
            
            # Save Axolotl config
            config_path = self.output_dir / 'configs' / 'axolotl_config.yml'
            with open(config_path, 'w') as f:
                yaml.dump(axolotl_config, f, default_flow_style=False)
            
            # Generate training script
            training_script = self.generate_training_script(str(config_path))
            script_path = self.output_dir / 'configs' / 'train_model.sh'
            with open(script_path, 'w') as f:
                f.write(training_script)
            script_path.chmod(0o755)
            
            config_info = {
                'config_path': str(config_path),
                'training_script': str(script_path),
                'axolotl_config': axolotl_config
            }
            
            self.state['outputs']['config_info'] = config_info
            
            logger.info(f"Generated Axolotl config: {config_path}")
            return config_info
            
        except Exception as e:
            logger.error(f"Error generating config: {str(e)}")
            self.state['errors'].append({
                'stage': 'generate_config',
                'error': str(e)
            })
            raise
    
    def generate_training_script(self, config_path: str) -> str:
        """Generate shell script for Axolotl training"""
        return f"""#!/bin/bash
# Axolotl QLoRA Training Script
# Generated by Akhi CrewAI Pipeline
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "=== Akhi Islamic AI Training Pipeline ==="
echo "Config: {config_path}"
echo "Start time: $(date)"
echo "========================================="

# Check if Axolotl is installed
if ! python -c "import axolotl" 2>/dev/null; then
    echo "Installing Axolotl..."
    pip install -e git+https://github.com/OpenAccess-AI-Collective/axolotl.git
fi

# Preprocess dataset
echo "Preprocessing dataset..."
python -m axolotl.cli.preprocess {config_path}

# Start training
echo "Starting QLoRA training..."
python -m axolotl.cli.train {config_path}

# Merge LoRA adapters
echo "Merging LoRA adapters..."
python -m axolotl.cli.merge_lora {config_path}

echo "========================================="
echo "Training completed successfully!"
echo "End time: $(date)"
echo "Model saved to: $(dirname {config_path})/../models/"
echo "========================================="
"""
    
    def generate_report(self, config_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final pipeline report"""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.state['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        report = {
            'pipeline_summary': {
                'status': 'completed' if not self.state['errors'] else 'completed_with_errors',
                'start_time': self.state['start_time'],
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_videos': self.state['total_videos'],
                'processed_videos': self.state['processed_videos']
            },
            'stage_outputs': self.state['outputs'],
            'errors': self.state['errors'],
            'ready_for_training': {
                'axolotl_config': config_info['config_path'],
                'training_script': config_info['training_script'],
                'dataset_path': self.state['outputs']['dataset_info']['train_path'],
                'total_samples': self.state['outputs']['dataset_info']['total_samples']
            },
            'next_steps': [
                f"cd {self.output_dir}",
                f"bash {config_info['training_script']}",
                "Monitor training logs",
                "Validate trained model",
                "Deploy to production"
            ]
        }
        
        # Save final report
        report_path = self.output_dir / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline report saved to: {report_path}")
        return report

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_urls_file(urls_file: str) -> List[str]:
    """Parse URLs from text file"""
    urls = []
    with open(urls_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='QLoRA Pipeline Orchestrator for Islamic AI Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --query "mufti menk trauma" --max-videos 10
  python run_pipeline.py --urls video_urls.txt --output ./my_output
  python run_pipeline.py --config custom_config.yaml
  python run_pipeline.py --query "islamic lectures" --config config.yaml
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', '-q', type=str, help='YouTube search query')
    input_group.add_argument('--urls', '-u', type=str, help='File containing YouTube URLs (one per line)')
    
    # Configuration options
    parser.add_argument('--config', '-c', type=str, help='YAML configuration file')
    parser.add_argument('--output', '-o', type=str, default='./pipeline_output', help='Output directory')
    parser.add_argument('--max-videos', type=int, default=10, help='Maximum videos to process')
    parser.add_argument('--duration', type=str, default='any', choices=['short', 'medium', 'long', 'any'], help='Video duration filter: short (<4min), medium (4-20min), long (>20min), any')
    parser.add_argument('--quality-threshold', type=float, default=0.7, help='Content quality threshold')
    
    # Model options
    parser.add_argument('--base-model', type=str, default='microsoft/DialoGPT-medium', help='Base model for fine-tuning')
    parser.add_argument('--whisper-model', type=str, default='base', help='Whisper model size')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--log-file', type=str, default='pipeline.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = {}
        
        # Override config with command line arguments
        config.update({
            'output_dir': args.output,
            'max_videos': args.max_videos,
            'quality_threshold': args.quality_threshold,
            'base_model': args.base_model,
            'whisper_model': args.whisper_model,
            'max_sequence_length': args.max_length,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })
        
        # Parse input
        urls = None
        if args.urls:
            urls = parse_urls_file(args.urls)
            logger.info(f"Loaded {len(urls)} URLs from {args.urls}")
        
        # Initialize and run pipeline
        orchestrator = QLoRAPipelineOrchestrator(config)
        report = await orchestrator.run_pipeline(query=args.query, urls=urls, duration=args.duration)
        
        # Print summary
        print("\n" + "="*60)
        print("QLORA PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Status: {report['pipeline_summary']['status']}")
        print(f"Duration: {report['pipeline_summary']['duration_seconds']:.1f} seconds")
        print(f"Videos processed: {report['pipeline_summary']['processed_videos']}/{report['pipeline_summary']['total_videos']}")
        print(f"Training samples: {report['ready_for_training']['total_samples']}")
        print(f"\nReady for training:")
        print(f"  Config: {report['ready_for_training']['axolotl_config']}")
        print(f"  Script: {report['ready_for_training']['training_script']}")
        print(f"  Dataset: {report['ready_for_training']['dataset_path']}")
        print(f"\nNext steps:")
        for step in report['next_steps']:
            print(f"  {step}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())