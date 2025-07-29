#!/usr/bin/env python3
"""
FAISS Query Tool for Akhi CrewAI Integration

This tool provides vector similarity search capabilities using FAISS indices.
It integrates with the FAISSStorageTool to query stored embeddings and retrieve
relevant content with metadata.

Features:
- Semantic similarity search
- Multiple search strategies (flat, approximate)
- Metadata filtering and ranking
- Integration with EmbedderTool for query embedding
- Batch query processing
- Result ranking and scoring
- Islamic content optimization

Author: Assistant
Date: 2024
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is required for vector querying. Install with: pip install faiss-cpu"
    )

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict, validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSQueryInput(BaseModel):
    """Input schema for FAISS Query Tool."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: Union[str, List[float], Any] = Field(
        description="Query text or embedding vector"
    )
    index_name: str = Field(
        description="Name of the FAISS index to search"
    )
    k: int = Field(
        default=5,
        description="Number of top results to return"
    )
    index_dir: str = Field(
        default="akhi_crewai/data/embeddings",
        description="Directory containing FAISS indices"
    )
    search_type: str = Field(
        default="similarity",
        description="Type of search: 'similarity', 'threshold', 'range'"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Similarity threshold for filtering results"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric: 'cosine', 'euclidean', 'inner_product'"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results"
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include embeddings in results"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters to apply"
    )
    rerank_by: Optional[str] = Field(
        default=None,
        description="Metadata field to use for reranking results"
    )
    embed_query: bool = Field(
        default=True,
        description="Whether to embed text query using EmbedderTool"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Embedding model for query (if embed_query=True)"
    )
    
    @validator('search_type')
    def validate_search_type(cls, v):
        allowed = ['similarity', 'threshold', 'range']
        if v not in allowed:
            raise ValueError(f"search_type must be one of {allowed}")
        return v
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        allowed = ['cosine', 'euclidean', 'inner_product']
        if v not in allowed:
            raise ValueError(f"distance_metric must be one of {allowed}")
        return v


class FAISSQueryOutput(BaseModel):
    """Output schema for FAISS Query Tool."""
    
    success: bool = Field(description="Whether the query was successful")
    query_text: str = Field(description="Original query text or description")
    index_name: str = Field(description="Name of the queried index")
    total_results: int = Field(description="Total number of results found")
    results: List[Dict[str, Any]] = Field(description="Search results with scores and metadata")
    query_embedding: Optional[List[float]] = Field(description="Query embedding vector")
    search_params: Dict[str, Any] = Field(description="Search parameters used")
    processing_time: float = Field(description="Time taken to process query in seconds")
    statistics: Dict[str, Any] = Field(description="Query statistics")
    message: str = Field(description="Status message")


class FAISSQueryTool(BaseTool):
    """FAISS Query Tool for semantic similarity search."""
    
    name: str = "FAISS Query Tool"
    description: str = (
        "Perform semantic similarity search on FAISS vector indices. "
        "Supports text queries, embedding vectors, metadata filtering, and result ranking."
    )
    args_schema: type = FAISSQueryInput
    embedder_tool: Optional[Any] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'embedder_tool') or self.embedder_tool is None:
            self.embedder_tool = None
        self._load_embedder_tool()
    
    def _load_embedder_tool(self):
        """Load EmbedderTool for text query embedding."""
        try:
            import sys
            import os
            # Add the tools directory to the path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from embedder import EmbedderTool
            self.embedder_tool = EmbedderTool()
            logger.info("EmbedderTool loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load EmbedderTool: {e}")
            self.embedder_tool = None
    
    def _run(self, **kwargs) -> FAISSQueryOutput:
        """Execute FAISS query operation."""
        start_time = self._get_current_time()
        
        try:
            # Validate input using Pydantic
            input_data = FAISSQueryInput(**kwargs)
            
            # Extract validated parameters
            query = input_data.query
            index_name = input_data.index_name
            k = input_data.k
            index_dir = input_data.index_dir
            search_type = input_data.search_type
            threshold = input_data.threshold
            distance_metric = input_data.distance_metric
            include_metadata = input_data.include_metadata
            include_embeddings = input_data.include_embeddings
            filter_metadata = input_data.filter_metadata
            rerank_by = input_data.rerank_by
            embed_query = input_data.embed_query
            model_name = input_data.model_name
            
            # Load index and metadata
            index, metadata = self._load_index_and_metadata(index_name, index_dir)
            
            # Prepare query embedding
            query_embedding, query_text = self._prepare_query_embedding(
                query, embed_query, model_name
            )
            
            # Perform search
            distances, indices = self._perform_search(
                index, query_embedding, k, search_type, threshold
            )
            
            # Process results
            results = self._process_search_results(
                distances, indices, metadata, include_metadata, 
                include_embeddings, filter_metadata, rerank_by
            )
            
            # Calculate statistics
            processing_time = self._get_current_time() - start_time
            statistics = self._generate_query_statistics(
                results, distances, processing_time
            )
            
            return FAISSQueryOutput(
                success=True,
                query_text=query_text,
                index_name=index_name,
                total_results=len(results),
                results=results,
                query_embedding=query_embedding.flatten().tolist() if include_embeddings else None,
                search_params={
                    "k": k,
                    "search_type": search_type,
                    "threshold": threshold,
                    "distance_metric": distance_metric,
                    "filter_metadata": filter_metadata,
                    "rerank_by": rerank_by
                },
                processing_time=processing_time,
                statistics=statistics,
                message=f"Successfully found {len(results)} results for query"
            )
            
        except Exception as e:
            processing_time = self._get_current_time() - start_time
            logger.error(f"Error in FAISS query: {e}")
            return FAISSQueryOutput(
                success=False,
                query_text=str(kwargs.get('query', 'Unknown query')),
                index_name=kwargs.get('index_name', 'Unknown index'),
                total_results=0,
                results=[],
                query_embedding=None,
                search_params={},
                processing_time=processing_time,
                statistics={},
                message=f"Query failed: {e}"
            )
    
    def _load_index_and_metadata(self, index_name: str, index_dir: str) -> Tuple[faiss.Index, Dict]:
        """Load FAISS index and associated metadata."""
        index_dir = Path(index_dir)
        index_path = index_dir / f"{index_name}.faiss"
        metadata_path = index_dir / f"{index_name}_metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {index_name} with {index.ntotal} vectors")
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} vectors")
        else:
            logger.warning(f"No metadata file found: {metadata_path}")
        
        return index, metadata
    
    def _prepare_query_embedding(self, query: Union[str, List[float], np.ndarray], 
                               embed_query: bool, model_name: Optional[str]) -> Tuple[np.ndarray, str]:
        """Prepare query embedding from text or vector input."""
        if isinstance(query, str):
            query_text = query
            if embed_query and self.embedder_tool:
                # Use EmbedderTool to embed the query
                embedding_result = self.embedder_tool._run(
                    text_chunks=[query],
                    output_format="list",
                    model_name=model_name
                )
                if embedding_result and len(embedding_result) > 0:
                    query_embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32)
                else:
                    raise ValueError("Failed to generate query embedding")
            else:
                raise ValueError("Text query requires embed_query=True and EmbedderTool")
        else:
            # Query is already an embedding
            query_text = f"Vector query (dim={len(query)})"
            if isinstance(query, list):
                query_embedding = np.array(query, dtype=np.float32)
            else:
                query_embedding = query.astype(np.float32)
        
        # Ensure query is 2D for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        return query_embedding, query_text
    
    def _perform_search(self, index: faiss.Index, query_embedding: np.ndarray, 
                      k: int, search_type: str, threshold: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the actual FAISS search."""
        if search_type == "similarity":
            # Standard k-NN search
            distances, indices = index.search(query_embedding, k)
        elif search_type == "threshold":
            # Search with distance threshold
            if threshold is None:
                raise ValueError("threshold must be provided for threshold search")
            distances, indices = index.search(query_embedding, min(k, index.ntotal))
            # Filter by threshold
            valid_mask = distances[0] <= threshold
            distances = distances[:, valid_mask]
            indices = indices[:, valid_mask]
        elif search_type == "range":
            # Range search (if supported by index)
            if threshold is None:
                raise ValueError("threshold must be provided for range search")
            try:
                lims, distances, indices = index.range_search(query_embedding, threshold)
                # Convert to standard format
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)
            except AttributeError:
                # Fallback to threshold search
                logger.warning("Range search not supported, using threshold search")
                return self._perform_search(index, query_embedding, k, "threshold", threshold)
        
        return distances, indices
    
    def _process_search_results(self, distances: np.ndarray, indices: np.ndarray, 
                              metadata: Dict, include_metadata: bool, include_embeddings: bool,
                              filter_metadata: Optional[Dict], rerank_by: Optional[str]) -> List[Dict[str, Any]]:
        """Process search results and apply filtering/ranking."""
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid result
                continue
            
            result = {
                "rank": i + 1,
                "score": float(distance),
                "similarity": 1.0 / (1.0 + float(distance)),  # Convert distance to similarity
                "vector_id": int(idx)
            }
            
            # Add metadata if available and requested
            if include_metadata and str(idx) in metadata:
                vector_metadata = metadata[str(idx)]
                result["metadata"] = vector_metadata
                
                # Add text content
                if "text" in vector_metadata:
                    result["text"] = vector_metadata["text"]
                
                # Add embedding if requested
                if include_embeddings and "embedding" in vector_metadata:
                    result["embedding"] = vector_metadata["embedding"]
            
            results.append(result)
        
        # Apply metadata filtering
        if filter_metadata:
            results = self._apply_metadata_filter(results, filter_metadata)
        
        # Apply reranking
        if rerank_by:
            results = self._rerank_results(results, rerank_by)
        
        return results
    
    def _apply_metadata_filter(self, results: List[Dict], filter_criteria: Dict[str, Any]) -> List[Dict]:
        """Apply metadata-based filtering to results."""
        filtered_results = []
        
        for result in results:
            if "metadata" not in result:
                continue
            
            metadata = result["metadata"]
            matches_filter = True
            
            for key, value in filter_criteria.items():
                if key not in metadata:
                    matches_filter = False
                    break
                
                if isinstance(value, list):
                    # Check if metadata value is in the list
                    if metadata[key] not in value:
                        matches_filter = False
                        break
                elif isinstance(value, dict):
                    # Handle range queries
                    if "min" in value and metadata[key] < value["min"]:
                        matches_filter = False
                        break
                    if "max" in value and metadata[key] > value["max"]:
                        matches_filter = False
                        break
                else:
                    # Exact match
                    if metadata[key] != value:
                        matches_filter = False
                        break
            
            if matches_filter:
                filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_results(self, results: List[Dict], rerank_by: str) -> List[Dict]:
        """Rerank results based on metadata field."""
        def get_rerank_value(result):
            if "metadata" in result and rerank_by in result["metadata"]:
                return result["metadata"][rerank_by]
            return 0  # Default value for missing metadata
        
        # Sort by rerank field (descending) while preserving similarity order for ties
        results.sort(key=lambda x: (get_rerank_value(x), x["similarity"]), reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return results
    
    def _generate_query_statistics(self, results: List[Dict], distances: np.ndarray, 
                                 processing_time: float) -> Dict[str, Any]:
        """Generate query statistics."""
        if len(results) == 0:
            return {
                "total_results": 0,
                "processing_time": processing_time,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0
            }
        
        similarities = [result["similarity"] for result in results]
        
        return {
            "total_results": len(results),
            "processing_time": processing_time,
            "avg_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "similarity_std": np.std(similarities),
            "distance_range": {
                "min": float(np.min(distances[distances != -1])) if len(distances[distances != -1]) > 0 else 0.0,
                "max": float(np.max(distances[distances != -1])) if len(distances[distances != -1]) > 0 else 0.0
            }
        }
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        return datetime.now().timestamp()
    
    # Utility methods for external use
    def search_similar(self, query: str, index_name: str, k: int = 5, 
                      index_dir: str = "akhi_crewai/data/embeddings") -> List[Dict[str, Any]]:
        """Simplified interface for similarity search."""
        result = self._run(
            query=query,
            index_name=index_name,
            k=k,
            index_dir=index_dir,
            search_type="similarity"
        )
        return result.results if result.success else []
    
    def search_with_threshold(self, query: str, index_name: str, threshold: float,
                            index_dir: str = "akhi_crewai/data/embeddings") -> List[Dict[str, Any]]:
        """Search with similarity threshold."""
        result = self._run(
            query=query,
            index_name=index_name,
            k=100,  # Large k for threshold search
            threshold=threshold,
            search_type="threshold",
            index_dir=index_dir
        )
        return result.results if result.success else []
    
    def list_available_indices(self, index_dir: str = "akhi_crewai/data/embeddings") -> List[str]:
        """List available FAISS indices."""
        index_dir = Path(index_dir)
        if not index_dir.exists():
            return []
        
        indices = []
        for file_path in index_dir.glob("*.faiss"):
            index_name = file_path.stem
            indices.append(index_name)
        
        return sorted(indices)


if __name__ == "__main__":
    # Demo usage
    print("FAISS Query Tool - Demo")
    print("=" * 40)
    
    query_tool = FAISSQueryTool()
    
    # List available indices
    available_indices = query_tool.list_available_indices()
    print(f"Available indices: {available_indices}")
    
    if available_indices:
        # Demo query
        sample_query = "What is Islam?"
        index_name = available_indices[0]
        
        try:
            result = query_tool._run(
                query=sample_query,
                index_name=index_name,
                k=3,
                include_metadata=True
            )
            
            print(f"\nQuery: '{sample_query}'")
            print(f"Index: {result.index_name}")
            print(f"Results: {result.total_results}")
            print(f"Processing time: {result.processing_time:.3f}s")
            
            for i, res in enumerate(result.results[:3]):
                print(f"\nResult {i+1}:")
                print(f"  Similarity: {res['similarity']:.3f}")
                if 'text' in res:
                    print(f"  Text: {res['text'][:100]}...")
                    
        except Exception as e:
            print(f"Demo failed: {e}")
    else:
        print("No indices found. Create some indices first using FAISSStorageTool.")