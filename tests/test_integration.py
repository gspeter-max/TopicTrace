import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid
from topictrace.tools.registry import run_tool


@patch("topictrace.settings.TAVILY_API_KEY", "test-key")
@patch("topictrace.settings.NVIDIA_API_KEY", "test-key")
@patch("topictrace.tools.web_search.TavilyClient")
@patch("topictrace.tools.web_fetch.requests.get")
@patch("topictrace.tools.summarize.OpenAI")
def test_full_research_chain(MockOpenAI, mock_get, MockTavilyClient):
    """Test the full research chain: search -> fetch -> summarize with caching."""
    session_name = "test-integration"
    session_path = create_session(session_name)

    # Mock web_search
    mock_tavily = MagicMock()
    MockTavilyClient.return_value = mock_tavily
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "Python Basics",
                "url": "https://example.com/python",
                "content": "Python is a programming language"
            }
        ]
    }

    # Mock web_fetch
    mock_response = MagicMock()
    mock_response.text = "# Python Basics\n\nPython is a high-level programming language..."
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Mock summarize (streaming response)
    mock_client = MagicMock()
    MockOpenAI.return_value = mock_client
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Python is a versatile programming language."
    mock_chunk.choices[0].delta.reasoning_content = None
    mock_client.chat.completions.create.return_value = [mock_chunk]

    # Step 1: Search
    search_results = run_tool("web_search", session_path=session_path, query="python basics")
    assert len(search_results) == 1
    assert search_results[0]["title"] == "Python Basics"

    # Verify search results saved to file
    assert os.path.exists(os.path.join(session_path, "search_results.md"))

    # Step 2: Fetch
    content = run_tool("web_fetch", session_path=session_path, url=search_results[0]["url"])
    assert "Python Basics" in content

    # Verify fetched content saved to file
    fetched_dir = os.path.join(session_path, "fetched_pages")
    assert len(os.listdir(fetched_dir)) >= 1

    # Step 3: Summarize
    summary = run_tool("summarize", session_path=session_path, content=content, query="python basics")
    assert "Python" in summary

    # Verify summary saved to file
    summaries_dir = os.path.join(session_path, "summaries")
    assert len(os.listdir(summaries_dir)) >= 1

    # Step 4: Cache the summary
    save_to_cache(session_path, "summary-python-basics", {"summary": summary})
    assert is_cache_valid(session_path, "summary-python-basics") is True

    # Step 5: Load from cache
    cached = load_from_cache(session_path, "summary-python-basics")
    assert cached["summary"] == summary

    # Cleanup
    shutil.rmtree(session_path)
