"""
Session folder manager for TopicTrace.

Each research query gets its own isolated folder.
This keeps data organized and prevents context pollution.
"""

import os
import re

# Base directory for all sessions
SESSIONS_DIR = "sessions"


def _sanitize_session_name(session_name: str) -> str:
    """
    Sanitize session name to prevent path traversal attacks.

    Removes any character that isn't alphanumeric, dash, or underscore.
    This prevents inputs like "../../etc" from escaping the sessions directory.

    Args:
        session_name: Raw session name from user input

    Returns:
        Safe session name with only allowed characters
    """
    # Remove path separators and dangerous characters
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', session_name)
    # Collapse multiple dashes/underscores
    safe_name = re.sub(r'[-_]+', '-', safe_name).strip('-')
    return safe_name


def get_session_path(session_name: str) -> str:
    """
    Get the full path for a session folder.

    Args:
        session_name: Name of the session (e.g., "A-Level-Biology-2024")

    Returns:
        Full path to the session directory (e.g., "sessions/A-Level-Biology-2024")

    Raises:
        ValueError: If session name is empty after sanitization
    """
    safe_name = _sanitize_session_name(session_name)
    if not safe_name:
        raise ValueError(
            f"Session name '{session_name}' is empty or contains only invalid characters. "
            "Use letters, numbers, dashes, or underscores."
        )
    return os.path.join(SESSIONS_DIR, safe_name)


def create_session(session_name: str) -> str:
    """
    Create a new session folder with all required subdirectories.

    Creates:
        sessions/<session_name>/
        sessions/<session_name>/fetched_pages/
        sessions/<session_name>/summaries/
        sessions/<session_name>/cache/

    Args:
        session_name: Name of the session (e.g., "A-Level-Biology-2024")

    Returns:
        Full path to the created session directory
    """
    session_path = get_session_path(session_name)

    # Create main session directory
    os.makedirs(session_path, exist_ok=True)

    # Create subdirectories for different data types
    os.makedirs(os.path.join(session_path, "fetched_pages"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "cache"), exist_ok=True)

    return session_path
