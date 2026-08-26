def test_package_imports_and_has_version():
    import anton_state

    assert isinstance(anton_state.__version__, str)
    assert anton_state.__version__
