import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    search_type: str  # 'dense', 'sparse', 'hybrid', 'reranked'

@dataclass
class SearchConfig:
    """Configuration for search behavior"""
    dense_top_k: int = 40
    sparse_top_k: int = 40
    final_top_k: int = 10
    rerank_top_k: int = 100
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    enable_reranking: bool = True
    rerank_model: str = "bge-reranker-v2-m3"
    min_score_threshold: float = 0.2  # Filter out results below this score

class HybridRAGSystem:
    """
    Advanced RAG system with hybrid search and cascading retrieval.
    
    Implements the following search strategies:
    1. Dense-only search (semantic)
    2. Sparse-only search (keyword/lexical)
    3. Hybrid search (dense + sparse combined)
    4. Cascading retrieval with reranking
    """
    
    def __init__(self, dense_index_name: str = "nvidia-docs", sparse_index_name: str = "nvidia-docs-sparse"):
        """Initialize the RAG system with Pinecone configuration"""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.dense_index_name = dense_index_name
        self.sparse_index_name = sparse_index_name
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize indexes (will be connected when needed)
        self.dense_index = None
        self.sparse_index = None
        
        # Model configurations
        self.dense_model = "llama-text-embed-v2"
        self.sparse_model = "pinecone-sparse-english-v0"
    
    def _connect_indexes(self) -> None:
        """Connect to Pinecone indexes"""
        if self.dense_index is None:
                self.dense_index = self.pc.Index(self.dense_index_name)
                logger.info(f"Connected to dense index: {self.dense_index_name}")
        else:
            logger.info(f"Reusing existing connection to dense index: {self.dense_index_name}")
        
        if self.sparse_index is None:
                self.sparse_index = self.pc.Index(self.sparse_index_name)
                logger.info(f"Connected to sparse index: {self.sparse_index_name}")
        else:
            logger.info(f"Reusing existing connection to sparse index: {self.sparse_index_name}")
    
    def dense_search(self, query: str, namespace: str = "__default__", top_k: int = 10) -> List[SearchResult]:
        """
        Perform semantic search using dense vectors.
        
        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        self._connect_indexes()
        assert self.dense_index is not None
        
        logger.info(f"Performing dense search for: '{query}' in namespace: '{namespace}'")
        
        logger.info("Attempting search with 'search_records' API...")
        # Search using integrated embedding with correct API format
        results = self.dense_index.search_records(
            namespace=namespace,
            query=SearchQuery(
                inputs={"text": query},
                top_k=top_k
            ),
            fields=["text", "filename", "source", "chunk_id", "chunk_index", "total_chunks", "file_type"]
        )
        
        logger.info(f"API returned {len(results['result']['hits'])} raw hits.")
            
        search_results = []
        for hit in results['result']['hits']:
            # Extract metadata from fields
            fields = hit.get('fields', {})
            metadata = {
                'filename': fields.get('filename', ''),
                'source': fields.get('source', ''),
                'chunk_id': fields.get('chunk_id', ''),
                'chunk_index': fields.get('chunk_index', 0),
                'total_chunks': fields.get('total_chunks', 1),
                'file_type': fields.get('file_type', '')
            }
            
            search_results.append(SearchResult(
                id=hit['_id'],
                score=hit['_score'],
                text=fields.get('text', ''),
                metadata=metadata,
                search_type='dense'
            ))
        
        logger.info(f"Dense search returned {len(search_results)} results")
        return search_results
    
    def sparse_search(self, query: str, namespace: str = "__default__", top_k: int = 10) -> List[SearchResult]:
        """
        Perform lexical/keyword search using sparse vectors.
        
        Args:
            query: Search query
            namespace: Pinecone namespace
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        self._connect_indexes()
        
        if self.sparse_index is None:
            logger.warning("Sparse index not available, cannot perform sparse search.")
            return []
        
        assert self.sparse_index is not None
        logger.info(f"Performing sparse search for: '{query}' in namespace: '{namespace}'")
        
        logger.info("Attempting search with 'search_records' API...")
        # Search using integrated sparse embedding
        results = self.sparse_index.search_records(
            namespace=namespace,
            query=SearchQuery(
                inputs={"text": query},
                top_k=top_k
            ),
            fields=["text", "filename", "source", "chunk_id", "chunk_index", "total_chunks", "file_type"]
        )
        
        logger.info(f"API returned {len(results['result']['hits'])} raw hits.")
            
        search_results = []
        for hit in results['result']['hits']:
            # Extract metadata from fields
            fields = hit.get('fields', {})
            metadata = {
                'filename': fields.get('filename', ''),
                'source': fields.get('source', ''),
                'chunk_id': fields.get('chunk_id', ''),
                'chunk_index': fields.get('chunk_index', 0),
                'total_chunks': fields.get('total_chunks', 1),
                'file_type': fields.get('file_type', '')
            }
            
            search_results.append(SearchResult(
                id=hit['_id'],
                score=hit['_score'],
                text=fields.get('text', ''),
                metadata=metadata,
                search_type='sparse'
            ))
        
        logger.info(f"Sparse search returned {len(search_results)} results")
        return search_results
    
    def hybrid_search(self, 
                     query: str, 
                     namespace: str = "__default__",
                     config: SearchConfig = SearchConfig()) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse results.
        
        Args:
            query: Search query
            namespace: Pinecone namespace
            config: Search configuration
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Performing hybrid search for: '{query}'")
        
        # Perform both dense and sparse searches
        dense_results = self.dense_search(query, namespace, config.dense_top_k)
        sparse_results = self.sparse_search(query, namespace, config.sparse_top_k)
        
        # Merge and deduplicate results
        merged_results = self._merge_results(dense_results, sparse_results, config)
        
        logger.info(f"Hybrid search merged to {len(merged_results)} unique results")
        return merged_results
    
    def _merge_results(self, 
                      dense_results: List[SearchResult], 
                      sparse_results: List[SearchResult],
                      config: SearchConfig) -> List[SearchResult]:
        """
        Merge dense and sparse results with score normalization.
        
        Uses a weighted combination approach for scoring.
        """
        logger.info(f"Merging {len(dense_results)} dense results and {len(sparse_results)} sparse results.")
        # Create a dictionary to store merged results
        merged_dict = {}
        
        # Normalize scores to 0-1 range for fair combination
        def normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
            if not results:
                return results
            
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return results
            
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
            
            return results
        
        # Normalize scores within each result set
        dense_normalized = normalize_scores(dense_results.copy())
        sparse_normalized = normalize_scores(sparse_results.copy())
        
        # Add dense results with weighting
        for result in dense_normalized:
            result.score *= config.dense_weight
            result.search_type = 'hybrid'
            merged_dict[result.id] = result
        
        # Add sparse results with weighting (or combine if duplicate)
        for result in sparse_normalized:
            result.score *= config.sparse_weight
            if result.id in merged_dict:
                # Combine scores for duplicate results
                merged_dict[result.id].score += result.score
                merged_dict[result.id].search_type = 'hybrid'
            else:
                result.search_type = 'hybrid'
                merged_dict[result.id] = result
        
        logger.info(f"Merged to {len(merged_dict)} unique results before sorting.")
        
        # Sort by combined score
        merged_results = sorted(merged_dict.values(), key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    def rerank_results(self, 
                      query: str, 
                      results: List[SearchResult],
                      top_k: int = 10,
                      model: str = "bge-reranker-v2-m3") -> List[SearchResult]:
        """
        Rerank search results using Pinecone's hosted reranking models.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return
            model: Reranking model to use
            
        Returns:
            List of reranked SearchResult objects
        """
        if not results:
            return results
        
        logger.info(f"Reranking {len(results)} results using {model}. Top score before rerank: {results[0].score:.4f}")
        
        # Prepare documents for reranking
        documents = []
        for result in results:
            documents.append({
                '_id': result.id,
                'text': result.text  # Use 'text' field instead of 'chunk_text'
            })
            
        # Call Pinecone reranker
        rerank_response = self.pc.inference.rerank(
            model=model,
            query=query,
            documents=documents,
            rank_fields=["text"],  # Use 'text' field instead of 'chunk_text'
            top_n=top_k,
            return_documents=True,
            parameters={"truncate": "END"}
        )
        
        logger.info(f"Reranker API call successful. Processing {len(rerank_response.data)} returned documents.")
            
        # Convert reranked results back to SearchResult objects
        reranked_results = []
        for row in rerank_response.data:
            doc = row['document']
            # Find original result to preserve metadata
            original_result = next((r for r in results if r.id == doc['_id']), None)
            
            if original_result:
                reranked_results.append(SearchResult(
                    id=doc['_id'],
                    score=row['score'],
                    text=doc['text'],  # Use 'text' field
                    metadata=original_result.metadata,
                    search_type='reranked'
                ))
            
        if reranked_results:
            logger.info(f"Reranking finished. Top score after rerank: {reranked_results[0].score:.4f}")
        else:
            logger.info("Reranking finished but no results were returned.")
        logger.info(f"Reranking returned {len(reranked_results)} results")
        return reranked_results
    
    def cascading_search(self, 
                        query: str, 
                        namespace: str = "__default__",
                        config: SearchConfig = SearchConfig()) -> List[SearchResult]:
        """
        Perform cascading retrieval: hybrid search → reranking.
        
        This is the recommended approach for maximum accuracy.
        
        Args:
            query: Search query
            namespace: Pinecone namespace
            config: Search configuration
            
        Returns:
            List of reranked SearchResult objects
        """
        logger.info(f"Performing cascading search for: '{query}'")
        
        # Stage 1: Hybrid search to get initial candidates
        logger.info("Cascading Search: Kicking off Stage 1 (Hybrid Search)...")
        hybrid_results = self.hybrid_search(query, namespace, config)
        logger.info(f"Cascading Search Stage 1 (Hybrid) completed with {len(hybrid_results)} candidates.")
        
        # Limit results for reranking stage
        rerank_candidates = hybrid_results[:config.rerank_top_k]
        
        # Stage 2: Reranking for final precision
        if config.enable_reranking and rerank_candidates:
            logger.info(f"Cascading Search: Proceeding to Stage 2 (Reranking top {len(rerank_candidates)} candidates)...")
            final_results = self.rerank_results(
                query, 
                rerank_candidates, 
                config.final_top_k, 
                config.rerank_model
            )
        else:
            if not config.enable_reranking:
                logger.info("Cascading Search: Reranking is disabled. Skipping Stage 2.")
            else:
                logger.info("Cascading Search: No candidates from Stage 1. Skipping Stage 2.")
            final_results = rerank_candidates[:config.final_top_k]
        
        logger.info(f"Cascading search completed with {len(final_results)} final results")
        return final_results
    
    def search(self, 
               query: str, 
               namespace: str = "__default__",
               strategy: str = "cascading",
               config: SearchConfig = SearchConfig()) -> List[SearchResult]:
        """
        Main search interface with multiple strategies.
        
        Args:
            query: Search query
            namespace: Pinecone namespace
            strategy: Search strategy ('dense', 'sparse', 'hybrid', 'cascading')
            config: Search configuration
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Executing {strategy} search for: '{query}'")
        
        if strategy == "dense":
            return self.dense_search(query, namespace, config.final_top_k)
        elif strategy == "sparse":
            return self.sparse_search(query, namespace, config.final_top_k)
        elif strategy == "hybrid":
            results = self.hybrid_search(query, namespace, config)
            return results[:config.final_top_k]
        elif strategy == "cascading":
            return self.cascading_search(query, namespace, config)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for both indexes"""
        stats = {}
        
        self._connect_indexes()
        
        if self.dense_index:
            dense_stats = self.dense_index.describe_index_stats()
            stats['dense'] = dense_stats
        
        if self.sparse_index:
            sparse_stats = self.sparse_index.describe_index_stats()
            stats['sparse'] = sparse_stats
        
        return stats

def print_results(results: List[SearchResult], query: str, config: SearchConfig = SearchConfig()):
    """Pretty print search results with score filtering"""
    print(f"\n🔍 Query: '{query}'")
    print("=" * 80)
    
    if not results:
        print("No results found.")
        return
    
    # Filter results by minimum score threshold
    filtered_results = [r for r in results if r.score >= config.min_score_threshold]
    
    if not filtered_results:
        print(f"No results above minimum score threshold ({config.min_score_threshold:.2f}).")
        print(f"Total results found: {len(results)}, but all were below threshold.")
        return
    
    if len(filtered_results) < len(results):
        print(f"📊 Showing {len(filtered_results)} results above threshold ({config.min_score_threshold:.2f})")
        print(f"   (Filtered out {len(results) - len(filtered_results)} low-relevance results)")
    
    for i, result in enumerate(filtered_results, 1):
        print(f"\n{i}. [{result.search_type.upper()}] Score: {result.score:.4f}")
        
        # Extract and display filename clearly
        filename = result.metadata.get('filename', 'Unknown file')
        source = result.metadata.get('source', '')
        chunk_info = f"chunk {result.metadata.get('chunk_index', 0) + 1}/{result.metadata.get('total_chunks', 1)}"
        
        print(f"   📄 File: {filename} ({chunk_info})")
        if source:
            print(f"   📂 Path: {source}")
        
        print(f"   📏 Chunk Size: {len(result.text)} characters")
        print(f"   📝 Text: {result.text}")
        print(f"   🔧 Metadata: {result.metadata}")
        print("-" * 40)

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Advanced RAG inference with hybrid search and reranking")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--namespace", default="nvidia-docs", help="Pinecone namespace (default: nvidia-docs)")
    parser.add_argument("--strategy", choices=["dense", "sparse", "hybrid", "cascading"], 
                       default="cascading", help="Search strategy")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--dense-index", default="nvidia-docs", help="Dense index name")
    parser.add_argument("--sparse-index", default="nvidia-docs-sparse", help="Sparse index name")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--rerank-model", default="bge-reranker-v2-m3", 
                       help="Reranking model to use")
    parser.add_argument("--dense-weight", type=float, default=0.7, help="Dense search weight")
    parser.add_argument("--sparse-weight", type=float, default=0.3, help="Sparse search weight")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum relevance score threshold")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    # Create search configuration
    config = SearchConfig(
        final_top_k=args.top_k,
        enable_reranking=not args.no_rerank,
        rerank_model=args.rerank_model,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        min_score_threshold=args.min_score
    )
    
    # Initialize RAG system
    rag_system = HybridRAGSystem(
        dense_index_name=args.dense_index,
        sparse_index_name=args.sparse_index
    )
    
    # Show index statistics if requested
    if args.stats:
        stats = rag_system.get_index_stats()
        print("\n📊 Index Statistics:")
        print("=" * 40)
        for index_type, stat in stats.items():
            print(f"{index_type.upper()}: {stat}")
        print()
    
    # Perform search
    results = rag_system.search(args.query, args.namespace, args.strategy, config)
    
    # Print results
    print_results(results, args.query, config)
    
    # Show search strategy info
    print(f"\n💡 Search Strategy: {args.strategy}")
    if args.strategy in ["hybrid", "cascading"]:
        print(f"   Dense weight: {args.dense_weight}")
        print(f"   Sparse weight: {args.sparse_weight}")
    if args.strategy == "cascading" and not args.no_rerank:
        print(f"   Reranking model: {args.rerank_model}")

if __name__ == "__main__":
    main() 