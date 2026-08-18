from anton.cloud_turn import session as cloud_session


def test_skills_read_from_mount_when_env_set(monkeypatch, tmp_path):
    """No copying: the mounted dir IS the skills root."""
    mount = tmp_path / "mnt" / "skills"
    (mount / "my-skill").mkdir(parents=True)
    (mount / "my-skill" / "SKILL.md").write_text("hello")
    monkeypatch.setenv("ANTON_CLOUD_SKILLS_ROOT", str(mount))

    root = cloud_session._stage_skills({"ignored": {"files": {"a.md": "x"}}})

    assert root == mount
    assert (root / "my-skill" / "SKILL.md").read_text() == "hello"
    assert not (root / "ignored").exists()   # wire payload is not staged


def test_skills_fall_back_to_tmp_staging_without_env(monkeypatch, tmp_path):
    """Desktop and CI keep the old behaviour."""
    monkeypatch.delenv("ANTON_CLOUD_SKILLS_ROOT", raising=False)

    root = cloud_session._stage_skills({"my-skill": {"files": {"SKILL.md": "x"}}})

    assert root.name.startswith(cloud_session._SKILLS_DIR_PREFIX)
    assert (root / "my-skill" / "SKILL.md").read_text() == "x"


def test_memory_read_from_mount_when_env_set(monkeypatch, tmp_path):
    mount = tmp_path / "mnt" / "memory" / "users" / "u1"
    mount.mkdir(parents=True)
    (mount / "profile.md").write_text("from efs")
    monkeypatch.setenv("ANTON_CLOUD_MEMORY_GLOBAL_ROOT", str(mount))

    global_dir = cloud_session._global_memory_dir({"global": {"profile": "from wire"}})

    assert global_dir == mount
    assert (global_dir / "profile.md").read_text() == "from efs"   # wire did not overwrite


def test_memory_falls_back_to_tmp_staging_without_env(monkeypatch):
    monkeypatch.delenv("ANTON_CLOUD_MEMORY_GLOBAL_ROOT", raising=False)

    global_dir = cloud_session._global_memory_dir({"global": {"profile": "from wire"}})

    assert (global_dir / "profile.md").read_text() == "from wire"
