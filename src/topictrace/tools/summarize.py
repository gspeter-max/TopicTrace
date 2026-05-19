"""
Summarize tool for TopicTrace using GLM-5.1 via NVIDIA NIM.

Uses the OpenAI client to connect to NVIDIA's NIM API endpoint.
GLM-5.1 is a powerful model for summarization tasks.
"""

import os
from openai import OpenAI

# NVIDIA NIM API configuration
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "z-ai/glm-5.1"


def summarize(content: str, query: str) -> str:
    """
    Summarize content using GLM-5.1 via NVIDIA NIM API.

    Takes the raw content (from web_fetch) and a query (from the research question)
    to produce a focused, relevant summary.

    Args:
        content: The raw text content to summarize (e.g., fetched web page)
        query: The research question to focus the summary on

    Returns:
        A concise summary string focused on the query, or error message if API fails
    """
    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY")

    # Create OpenAI client pointing to NVIDIA NIM endpoint
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key
    )

    try:
        # Call GLM-5.1 to summarize the content
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant. Summarize the given content concisely and accurately."
                },
                {
                    "role": "user",
                    "content": f"Based on this content: {content}\n\nProvide a concise summary relevant to: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        # Extract the summary from the response
        return response.choices[0].message.content

    except Exception as e:
        # Return error message if API call fails
        return f"Error summarizing content: {str(e)}"
