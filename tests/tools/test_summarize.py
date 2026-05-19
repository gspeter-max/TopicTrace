from unittest.mock import patch, MagicMock
from topictrace.tools.summarize import summarize


@patch("topictrace.tools.summarize.OpenAI")
def test_summarize_returns_summary_string(MockOpenAI):
    """Test that summarize returns a summary string from GLM-5.1."""
    mock_client = MagicMock()
    MockOpenAI.return_value = mock_client

    mock_message = MagicMock()
    mock_message.content = "Python is a high-level programming language known for readability."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    content = "Python is a programming language that lets you work quickly..."
    query = "What is Python?"
    result = summarize(content, query)

    assert isinstance(result, str)
    assert "Python" in result
    assert len(result) > 0


@patch("topictrace.tools.summarize.OpenAI")
def test_summarize_handles_api_error(MockOpenAI):
    """Test that summarize returns error message when API fails."""
    mock_client = MagicMock()
    MockOpenAI.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    content = "Some content"
    query = "Summarize this"
    result = summarize(content, query)

    assert isinstance(result, str)
    assert "Error" in result or "error" in result
