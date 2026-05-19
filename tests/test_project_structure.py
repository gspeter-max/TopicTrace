def test_package_importable():
    """Test that the topictrace package can be imported."""
    import topictrace
    assert topictrace is not None

def test_tools_subpackage_importable():
    """Test that the tools subpackage can be imported."""
    from topictrace import tools
    assert tools is not None
