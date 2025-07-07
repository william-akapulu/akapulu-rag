import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent, CallToolResult
from retrieve import HybridRAGSystem, SearchConfig, SearchResult 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. RENAME SERVER to be more generic
server = FastMCP("access-rag-db")
rag_system = None # Initialize lazily when first search is called

# 2. CREATE A WRAPPER FUNCTION with the core search logic
async def _perform_rag_search(query: str, namespace: str, tool_name: str) -> CallToolResult:
    """
    A reusable wrapper that performs a cascading RAG search for a given namespace.
    """
    global rag_system
    try:
        if rag_system is None:
            logger.info("First tool call. Initializing HybridRAGSystem...")
            rag_system = HybridRAGSystem()
        
        # Fixed search parameters
        strategy = "cascading"
        top_k = 10
        min_score = 0.1
        
        config = SearchConfig(
            final_top_k=top_k,
            min_score_threshold=min_score
        )
        
        logger.info(f"Executing search via tool '{tool_name}' in namespace '{namespace}'")
        results = rag_system.search(query, namespace, strategy, config)
        
        formatted_results = _format_search_results(results, query, tool_name)
        
        return CallToolResult(content=[TextContent(type="text", text=formatted_results)])
    
    except Exception as e:
        logger.error(f"Error handling tool call for '{tool_name}': {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )

# 3. DEFINE THE TOOL for NVIDIA docs, calling the wrapper
@server.tool()
async def search_nvidia_docs(query: str) -> CallToolResult:
    """Search the NVIDIA documentation index.'.

    Args:
        query: The search query to find relevant NVIDIA documentation.
    """
    return await _perform_rag_search(
        query=query, 
        namespace="nvidia-docs", 
        tool_name="search_nvidia_docs"
    )

# 4. READY FOR THE FUTURE: To add a new tool, just copy the one above and change the details.
# For example:
#
# @server.tool()
# async def search_pipecat_docs(query: str) -> CallToolResult:
#     """Search the Pipecat documentation index. Use for questions about the Pipecat voice AI framework, its architecture, and usage.
#
#     Args:
#         query: The search query to find relevant Pipecat documentation.
#     """
#     return await _perform_rag_search(
#         query=query, 
#         namespace="pipecat-docs", # <-- The only change needed here
#         tool_name="search_pipecat_docs"
#     )

def _format_search_results(results: List[SearchResult], query: str, tool_name: str) -> str:
    """Format search results for display"""
    if not results:
        return f"No results found for query: '{query}' in {tool_name}"
    
    output = [f"🔍 Search Results from '{tool_name}' for query: '{query}'"]
    output.append("=" * 60)
    
    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. [{result.search_type.upper()}] Score: {result.score:.4f}")
        
        # Extract metadata
        filename = result.metadata.get('filename', 'Unknown file')
        repo_name = result.metadata.get('repo_name', '')
        content_type = result.metadata.get('content_type', 'documentation')
        
        if repo_name:
            output.append(f"   📁 Repository: {repo_name}")
            output.append(f"   📄 File: {filename} ({content_type})")
        else:
            output.append(f"   📄 File: {filename}")
        
        # Add chunk info if available
        chunk_info = result.metadata.get('chunk_index')
        total_chunks = result.metadata.get('total_chunks')
        if chunk_info is not None and total_chunks is not None:
            output.append(f"   📝 Chunk: {chunk_info + 1}/{total_chunks}")
        
        # Add content preview
        preview = result.text
        output.append(f"   💬 Content: {preview}")
        output.append("-" * 40)
    
    return "\n".join(output)

# Note: _format_stats is kept but not directly used by the exposed tool
def _format_stats(stats: Dict[str, Any]) -> str:
    """Format index statistics for display"""
    output = ["📊 NVIDIA Docs Index Statistics"]
    output.append("=" * 40)
    
    for index_type, stat in stats.items():
        output.append(f"\n{index_type.upper()} INDEX:")
        if isinstance(stat, dict):
            total_vectors = stat.get('total_vector_count', 0)
            output.append(f"  Total vectors: {total_vectors:,}")
            
            namespaces = stat.get('namespaces', {})
            if namespaces:
                output.append(f"  Namespaces: {len(namespaces)}")
                for ns_name, ns_info in namespaces.items():
                    ns_count = ns_info.get('vector_count', 0)
                    output.append(f"    - {ns_name}: {ns_count:,} vectors")
    
    return "\n".join(output)

# The main function is no longer needed as the server is launched by the MCP client.
# The MCP client directly interacts with the NVIDIADocsMCPServer instance.
# async def main():
#    server = NVIDIADocsMCPServer()
#    await server.run()

# if __name__ == "__main__":
#    asyncio.run(main())

if __name__ == "__main__":
    print("Attempting to start MCP server...")
    try:
        server.run(transport='stdio')
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
    print("MCP server stopped.") 