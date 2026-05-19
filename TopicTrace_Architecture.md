# TopicTrace Architecture & Plan

## 1. Project Overview
**TopicTrace** is an educational predictive analytics agent (CLI) designed for students. Its core mission is to act as a "deep research" assistant that crawls the web for historical exam papers, syllabi, and study materials, extracting patterns to predict high-probability exam questions.

## 2. Core Architecture Decisions

### 2.1 Ingestion & Search (The Data Layer)
*   **Strategy:** Simple, single-tool approach. No fallbacks. Pick the best tool for each layer.
*   **Search Infrastructure:** `Tavily` (AI-optimized API, 1k free/mo) for web search.
*   **Scraping Infrastructure:** `Jina Reader` (`r.jina.ai`) — zero setup, converts any URL to clean Markdown via simple HTTP call.
*   **Non-Text Data:** Diagrams and complex equations will be handled via **Description Extraction**. The agent will extract captions, labels, and question contexts (e.g., "Label the mitochondria in Figure 1") rather than performing OCR on the images themselves.
*   **Caching:** 20-minute TTL. Search results and fetched pages are cached in session folders to avoid re-fetching.

### 2.2 Intelligence (The Brain Layer)
*   **LLM Provider:** NVIDIA NIM (`integrate.api.nvidia.com/v1`) using the `z-ai/glm-5.1` model (with `enable_thinking: True`).
*   **Agentic Flow:** An iterative "Loop" where the agent searches, extracts, reviews the data, and decides if it needs to search further before concluding.
*   **Tool Calling:** Native JSON tool calling (Option A) passed via the `openai` client `tools` parameter.
*   **Prediction Logic:** **Syllabus Weighting**. The agent will cross-reference the frequency of historical questions against the official UK syllabus mark weightings (e.g., AQA, OCR, Edexcel) to determine probability.

### 2.3 Memory & Storage (The State Layer)
*   **Strategy:** **Session Folders**. The agent does not use a massive global database.
*   **Structure:** Each time a student starts a new research query (e.g., "A-Level Biology Prep"), the CLI creates a new, isolated directory (e.g., `sessions/A-Level-Biology-2024/`).
*   **Format:** All scraped data, agent reasoning, and the final prediction report are saved as Markdown (`.md`) files within that specific session folder. This ensures token efficiency, prevents context pollution, and allows the student to easily export their study notes.
*   **Session Folder Structure:**
    ```
    sessions/
      A-Level-Biology-2024/
        search_results.md
        fetched_pages/
          page1.md
          page2.md
        summaries/
          summary1.md
        cache/
          search_cache.json
          fetch_cache.json
        final_report.md
    ```

### 2.4 User Interface & Deployment (The Presentation Layer)
*   **Strategy:** A cloneable, local Python CLI repository.
*   **User Experience:** Students clone the GitHub repository and run the agent in their local terminal.
*   **Visuals:** Uses Python's `prompt-toolkit` and `rich` libraries to provide a professional, interactive, and colorful terminal experience (e.g., streaming the agent's "thinking" process in gray text before displaying the final output).
*   **Security:** API keys (NVIDIA/Tavily) are stored locally in a `.env` file and never hardcoded in the public repository scripts.

### 2.5 Tool Architecture (Three Core Tools)

#### Tool 1: `web_search(query)`
```
Input:  search query string
Process:
  1. Check cache (20-min TTL) → return cached if fresh
  2. Call Tavily API with query
  3. Save results to session/search_results.md
  4. Cache results with timestamp
Output: List of {title, url, snippet}
```

#### Tool 2: `web_fetch(url)`
```
Input:  URL string
Process:
  1. Check cache (20-min TTL) → return cached if fresh
  2. Call Jina Reader API (r.jina.ai/URL)
  3. Returns clean Markdown content
  4. Save to session/fetched_pages/page_N.md
  5. Cache with timestamp
Output: Clean Markdown content
```

#### Tool 3: `summarize(content, query)`
```
Input:  content string + original query
Process:
  1. Send to NVIDIA NIM (GLM-5.1) via openai client
  2. LLM summarizes content relevant to query
  3. Save to session/summaries/summary_N.md
Output: Concise summary string
```

### 2.6 Dependencies & Package Management
*   **Package Manager:** `uv` (not pip)
*   **Config File:** `pyproject.toml` (not requirements.txt)
*   **Key Dependencies:**
    - `tavily-python` — web search API
    - `requests` — HTTP calls to Jina Reader
    - `openai` — NVIDIA NIM client
    - `python-dotenv` — .env file loading
    - `rich` — terminal formatting
    - `prompt-toolkit` — interactive CLI

## 3. Implementation Code Snippet (Core LLM Loop)

```python
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=os.getenv("NVIDIA_API_KEY") # Secured via .env
)

def run_agent_loop(user_query):
    # This loop will eventually include native tool calls for web search and file saving
    completion = client.chat.completions.create(
      model="z-ai/glm-5.1",
      messages=[{"role": "user", "content": user_query}],
      temperature=1,
      top_p=1,
      max_tokens=16384,
      extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
      stream=True
    )

    for chunk in completion:
      if not getattr(chunk, "choices", None):
        continue
      if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
        continue
      delta = chunk.choices[0].delta
      reasoning = getattr(delta, "reasoning_content", None)
      if reasoning:
        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
      if getattr(delta, "content", None) is not None:
        print(delta.content, end="")
```
