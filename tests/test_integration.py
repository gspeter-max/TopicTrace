import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.cache import save_to_cache, load_from_cache, is_cache_valid
from topictrace.tools.registry import run_tool


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
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Mock summarize
    mock_openai = MagicMock()
    MockOpenAI.return_value = mock_openai
    mock_message = MagicMock()
    mock_message.content = "Python is a versatile programming language used for web development, data science, and automation."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_ai_response = MagicMock()
    mock_ai_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_ai_response

    # Step 1: Search
    search_results = run_tool("web_search", query="python basics")
    assert len(search_results) == 1
    assert search_results[0]["title"] == "Python Basics"

    # Step 2: Fetch
    content = run_tool("web_fetch", url=search_results[0]["url"])
    assert "Python Basics" in content

    # Step 3: Summarize
    summary = run_tool("summarize", content=content, query="python basics")
    assert "Python" in summary
    assert "programming language" in summary

    # Step 4: Cache the summary
    save_to_cache(session_path, "summary-python-basics", {"summary": summary})
    assert is_cache_valid(session_path, "summary-python-basics") is True

    # Step 5: Load from cache
    cached = load_from_cache(session_path, "summary-python-basics")
    assert cached["summary"] == summary

    # Cleanup
    shutil.rmtree(session_path)
