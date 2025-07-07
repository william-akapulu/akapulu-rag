# Akapulu RAG Backend

This repository contains the backend infrastructure for a sophisticated Retrieval-Augmented Generation (RAG) system. It's designed to process documentation and source code, embed it into a hybrid vector store, and expose it as a tool for an AI assistant to use.

## Getting Started

To get the system running locally, follow these steps.

### 1. Setup Your Environment

First, clone the repository and navigate into the directory. It's highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

The system requires an API key to connect to Pinecone.

1.  Create a file named `.env` in the root of the project directory.
2.  Contact **William** to get the organization's `PINECONE_API_KEY`.
3.  Add the key to your `.env` file like this:

    ```
    PINECONE_API_KEY="your-pinecone-api-key-here"
    ```

You are now ready to use the scripts.

---

## Core Components

The system is composed of three main Python scripts, each with a distinct responsibility.

### 1. `embed_pdf_hybrid.py` (The ETL Pipeline)

-   **What it does:** This script is the heart of the ETL (Extract, Transform, Load) pipeline. It reads documents from various sources (local PDFs, HTML files, or entire Git repositories), splits them into manageable chunks, and embeds them into two separate Pinecone vector indexes: one for dense (semantic) vectors and one for sparse (keyword) vectors. This hybrid approach is key to the system's high retrieval accuracy.
-   **How to run it:** The script is run from the command line. You must provide a data source (`--folder` or `--repo-url`) and a target `namespace`.

    ```bash
    # Example: Process a local folder of documents
    python embed_pdf_hybrid.py --folder ./pdfs --namespace nvidia-docs

    # Example: Process a remote Git repository
    python embed_pdf_hybrid.py --repo-url https://github.com/NVIDIA/TensorRT.git --namespace nvidia-docs
    ```
    > **Note:** Before running, you must add your desired namespace (e.g., `nvidia-docs`) to the `ALLOWED_NAMESPACES` set within the script to pass its safety guardrails.

    > **A Note on Index Naming:** The Pinecone indexes are hardcoded to `nvidia-docs` and `nvidia-docs-sparse` due to the project's origin. However, they are generic and can store data from any source. The `--namespace` parameter is what keeps different data sources (like 'nvidia-docs', 'pipecat-docs', etc.) logically separated within the same index.

### 2. `retrieve.py` (The Retrieval Engine)

-   **What it does:** This script contains the logic for searching the vector indexes. It implements several advanced retrieval strategies, with "cascading" (hybrid search followed by a reranker model) being the most powerful and recommended approach. It can be used as a standalone script for testing or as a library by the MCP server.
-   **How to run it:** Use this script from the command line to test retrieval quality.

    ```bash
    # Example: Perform a default cascading search
    python retrieve.py "how does automatic mixed precision work?" --namespace nvidia-docs

    # Example: Test a simple sparse keyword search
    python retrieve.py "TRT_VERSION" --strategy sparse --namespace nvidia-docs
    ```

### 3. `mcp_server.py` (The AI Assistant API)

-   **What it does:** This script wraps the retrieval engine in a `FastMCP` server. MCP (Meta-Cognitive Process) is a framework that allows an AI assistant (like Cursor) to discover and use external tools. This server exposes the RAG system as a simple `search_nvidia_docs` tool that the assistant can call.
-   **How to run it:** The server is intended to be launched by the AI assistant itself, based on the configuration in `mcp.json`. You can also run it directly for debugging:
    ```bash
    python mcp_server.py
    ```

---

## The Hybrid Search Strategy

This project uses a powerful hybrid search model to retrieve information. This approach combines two distinct search methods to achieve results that are both semantically relevant and keyword-precise.

### How It Works

1.  **Parallel Search**: When a query is made, the system searches against two different vector indexes simultaneously:
    -   **Dense Index (`nvidia-docs`):** This index stores dense vectors that capture the *semantic meaning* of the text. It excels at understanding the user's intent and finding conceptually similar results, even if the keywords don't match exactly. (e.g., searching for "making models faster" might find documents about "performance optimization").
    -   **Sparse Index (`nvidia-docs-sparse`):** This index stores sparse vectors that are highly optimized for *keyword matching*. It excels at finding specific, exact terms like function names (`HybridRAGSystem`), error codes, or technical jargon.

2.  **Score Normalization and Merging**: The raw scores from dense and sparse searches are not directly comparable. The system normalizes the scores from each result set to a common scale (0 to 1). It then combines these scores using a weighted average (`dense_weight` and `sparse_weight`) to produce a single, unified list of results.

3.  **Reranking for Precision**: For the highest accuracy (the default "cascading" strategy), the top ~100 results from the merged list are passed to a more powerful reranker model. This model's sole job is to take this smaller, promising set and re-evaluate it with much higher scrutiny, pushing the absolute best matches to the top.

This multi-stage process ensures we get the "best of both worlds": the broad, conceptual understanding of semantic search and the sharp precision of keyword search.

---

## Connecting to the AI Assistant (MCP)

To allow an AI assistant like Cursor to use the tools defined in `mcp_server.py`, you need to create a configuration file that tells the editor how to launch the server.

### The `mcp.json` File

The `mcp.json` file is a manifest that registers custom tool servers with the AI assistant. When you open a repository with this file, the assistant reads it and understands that new tools are available for it to use. This allows you to create repository-specific tools that are only active when you are working on that project.

**File Location:**

Create this file at the following path relative to your project's root directory:
`.cursor/mcp.json`

So, for this project, the full path would be `nvidia-docs-backend/.cursor/mcp.json`.

**Sample Configuration:**

Here is a sample configuration. You will need to adjust the paths to be the **absolute paths** on your machine.

```json
{
  "mcpServers": {
    "access-rag-db": {
      "command": "/path/to/your/project/nvidia-docs-backend/venv/bin/python3",
      "args": ["/path/to/your/project/nvidia-docs-backend/mcp_server.py"]
    }
  }
}
```

**Configuration Breakdown:**

-   `"access-rag-db"`: This is a unique name for your tool server. It can be anything you like.
-   `"command"`: The **absolute path** to the executable that will run the server. For Python projects, this should be the `python` or `python3` binary located inside your virtual environment's `bin` directory. This is critical to ensure the server runs with the correct dependencies installed in the `venv`.
-   `"args"`: A list of arguments to pass to the command. The first and only argument here should be the **absolute path** to the server script, `mcp_server.py`.

### Verifying the Integration in Cursor

Once the `.cursor/mcp.json` file is saved with the correct absolute paths, you can verify that the AI assistant has recognized the tool server.

1.  Open Cursor's settings (press `Cmd + ,` on macOS or `Ctrl + ,` on Windows/Linux).
2.  Navigate to the **Tools & Integrations** section.
3.  You should see your `access-rag-db` server listed under "MCP Servers", indicating that Cursor has successfully loaded the configuration.

*Placeholder for screenshot of Cursor's Tools & Integrations settings here.*
![Cursor Settings](assets/Cursor%20Settings.png)

### Troubleshooting

If the tool does not appear in the settings or fails to run, you can check the logs for errors.
1. Open the terminal panel in Cursor (press `Ctrl + \``).
2. Go to the **OUTPUT** tab (usually next to TERMINAL, DEBUG CONSOLE, etc.).
3. In the dropdown menu on the right side of the OUTPUT panel, select **MCP Logs**.
This will show you the startup logs, standard output, and any errors from your `mcp_server.py` script.

### Example in Practice

With the server running, you can now use the tools in your chat with the AI. The `.cursor/mcp.json` file tells the assistant: "When you are in this project, you have access to a new set of tools. The instructions for running the server that provides these tools are right here."

For example, when you ask the assistant a question like:
> "Hey, can you search the NVIDIA docs for information on TensorRT?"

The assistant will see that it has a `search_nvidia_docs` tool available (because `mcp_server.py` defines it), recognize that it's a good fit for your query, and use the `command` and `args` from `mcp.json` to execute the search and get back the results.

![Tool Call In Action](assets/Tool%20Call%20In%20Action.png)
