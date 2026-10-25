from anton.config.settings import _build_env_files

def test_explicit_cowork_home_does_not_inherit_daily_app_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    isolated=tmp_path / "isolated"
    isolated.mkdir()
    (isolated / ".env").write_text("ACT_FIRST=true\n")
    monkeypatch.setenv("COWORK_HOME", str(isolated))
    assert _build_env_files() == [".env", str(isolated / ".env")]
