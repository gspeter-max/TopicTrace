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
    expected_path = os.path.join("sessions", "test-session-path")
    actual_path = get_session_path(session_name)
    assert actual_path == expected_path


def test_get_session_path_sanitizes_dangerous_input():
    """Test that get_session_path prevents path traversal attacks."""
    # Try to escape sessions directory
    dangerous_name = "../../etc/passwd"
    safe_path = get_session_path(dangerous_name)

    # Should NOT contain path traversal
    assert ".." not in safe_path
    # Should be safely contained within sessions directory
    assert safe_path.startswith("sessions/")
    # The dangerous parts should be sanitized out
    assert "etc/passwd" not in safe_path


def test_get_session_path_raises_on_empty_name():
    """Test that get_session_path raises ValueError for empty session name."""
    import pytest
    with pytest.raises(ValueError, match="empty or contains only invalid characters"):
        get_session_path("")


def test_get_session_path_raises_on_only_invalid_chars():
    """Test that get_session_path raises ValueError for names with only invalid chars."""
    import pytest
    with pytest.raises(ValueError, match="empty or contains only invalid characters"):
        get_session_path("../")
