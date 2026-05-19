"""
Session folder manager for TopicTrace.

Each research query gets its own isolated folder.
This keeps data organized and prevents context pollution.
"""

import os

# Base directory for all sessions
SESSIONS_DIR = "sessions"


def get_session_path(session_name: str) -> str:
    """
    Get the full path for a session folder.

    Args:
        session_name: Name of the session (e.g., "A-Level-Biology-2024")

    Returns:
        Full path to the session directory (e.g., "sessions/A-Level-Biology-2024")
    """
    return os.path.join(SESSIONS_DIR, session_name)


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
