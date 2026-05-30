"""Session directory manager for TopicTrace."""

import os
import re

from topictrace import settings

def get_session_path(session_name: str) -> str:
    """Return the full path for a session directory."""
    safe_name  = re.sub(r'[^a-zA-Z0-9_-]', '', session_name)
    safe_name = re.sub(r'[-_]+', '-', safe_name).strip('-')
    if not safe_name:
        raise ValueError(
            f"Session name '{session_name}' is empty or contains only invalid characters. "
            "Use letters, numbers, dashes, or underscores."
        )
    return os.path.join(settings.SESSIONS_DIR, safe_name)


def create_session(session_name: str) -> str:
    """Create a new session directory with fetched_pages, summaries, and cache subdirectories."""
    session_path = get_session_path(session_name)

    # Create main session directory
    os.makedirs(session_path, exist_ok=True)

    # Create subdirectories for different data types
    os.makedirs(os.path.join(session_path, "fetched_pages"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "cache"), exist_ok=True)

    return session_path

def save_numberd_file(subdir : str , prefix : str, content : str, session_path : str)\
    -> str :

    dir_path = os.path.join(session_path, subdir )
    os.makedirs( dir_path , exist_ok= True)
    existing_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]
    file_path = os.path.join(dir_path, f"{prefix}_{len(existing_files) + 1 }.md")
    with open(file_path , "w") as f:
        f.write(content)

    return file_path