from __future__ import annotations

from typer.testing import CliRunner

from anton import cli


runner = CliRunner()


def test_main_exits_after_self_update(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_ensure_dependencies", lambda console: None)

    import anton.config.settings as settings_module
    import anton.updater as updater_module

    class FakeSettings:
        disable_autoupdates = False

        def resolve_workspace(self, folder) -> None:
            self.folder = folder

    def fail_if_called(*args, **kwargs):
        raise AssertionError("startup should stop after a self-update")

    monkeypatch.setattr(settings_module, "AntonSettings", FakeSettings)
    monkeypatch.setattr(updater_module, "check_and_update", lambda console, settings: True)
    monkeypatch.setattr(cli, "_ensure_workspace", fail_if_called)
    monkeypatch.setattr(cli, "_ensure_api_key", fail_if_called)

    result = runner.invoke(cli.app, [])

    assert result.exit_code == 0
    assert "restart anton to use the updated version" in result.output.lower()
