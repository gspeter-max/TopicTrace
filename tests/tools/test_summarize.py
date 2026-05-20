import os
import shutil
from unittest.mock import patch, MagicMock
from topictrace.session import create_session
from topictrace.tools.summarize import summarize


@patch("topictrace.tools.summarize.call_llm")
def test_summarize_returns_summary_string(mock_call_llm):
    """Test that summarize returns a summary string from GLM-5.1."""
    session_name = "test-summarize-return"
    session_path = create_session(session_name)

    mock_call_llm.return_value = "Python is a high-level programming language."

    content = "Python is a programming language that lets you work quickly..."
    query = "What is Python?"
    result = summarize(content, query, session_path)

    assert isinstance(result, str)
    assert "Python" in result
    assert len(result) > 0

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.tools.summarize.call_llm")
def test_summarize_handles_api_error(mock_call_llm):
    """Test that summarize raises exception when API fails."""
    session_name = "test-summarize-error"
    session_path = create_session(session_name)

    mock_call_llm.side_effect = Exception("API Error")

    content = "Some content"
    query = "Summarize this"

    import pytest
    with pytest.raises(Exception, match="API Error"):
        summarize(content, query, session_path)

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.tools.summarize.call_llm")
def test_summarize_saves_to_summaries_directory(mock_call_llm):
    """Test that summarize saves the summary to summaries/ directory."""
    session_name = "test-summarize-save"
    session_path = create_session(session_name)

    mock_call_llm.return_value = "Summary saved."

    summarize("content", "query", session_path)

    summaries_dir = os.path.join(session_path, "summaries")
    files = os.listdir(summaries_dir)
    assert len(files) >= 1

    # Cleanup
    shutil.rmtree(session_path)


def test_summarize_raises_on_empty_content():
    """Test that summarize raises ValueError when content is empty."""
    session_name = "test-summarize-empty"
    session_path = create_session(session_name)

    import pytest
    with pytest.raises(ValueError, match="Content cannot be empty"):
        summarize("", "query", session_path)

    # Cleanup
    shutil.rmtree(session_path)


@patch("topictrace.settings.NVIDIA_API_KEY", None)
def test_summarize_raises_without_api_key():
    """Test that summarize raises ValueError when NVIDIA_API_KEY is not set."""
    session_name = "test-summarize-no-key"
    session_path = create_session(session_name)

    import pytest
    with pytest.raises(ValueError, match="NVIDIA_API_KEY not found"):
        summarize("content", "query", session_path)

    # Cleanup
    shutil.rmtree(session_path)
