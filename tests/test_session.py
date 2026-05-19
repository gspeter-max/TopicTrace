import os
import shutil
from topictrace.session import create_session, get_session_path


def test_create_session_creates_directory():
    """Test that create_session creates a session directory."""
    session_name = "test-session-create"
    session_path = create_session(session_name)
    assert os.path.isdir(session_path)
    # Cleanup
    shutil.rmtree(session_path)


def test_create_session_creates_subdirectories():
    """Test that create_session creates required subdirectories."""
    session_name = "test-session-subdirs"
    session_path = create_session(session_name)
    assert os.path.isdir(os.path.join(session_path, "fetched_pages"))
    assert os.path.isdir(os.path.join(session_path, "summaries"))
    assert os.path.isdir(os.path.join(session_path, "cache"))
    # Cleanup
    shutil.rmtree(session_path)


def test_get_session_path_returns_correct_path():
    """Test that get_session_path returns the correct path for a session name."""
    session_name = "test-session-path"
    expected_path = os.path.join("sessions", session_name)
    actual_path = get_session_path(session_name)
    assert actual_path == expected_path
