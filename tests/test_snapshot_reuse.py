import json
from pathlib import Path
import pytest
from anton.core.tools.generate_artifact import observed_data,snapshot_reuse


def make_page(folder, data):
    (folder/'observed-data.json').write_text(json.dumps(data))
    (folder/'index.html').write_text(observed_data.bind('<html><head></head><body>Observed data</body></html>',{observed_data.identity(data):data}))


def test_only_a_matching_existing_page_offers_reuse(tmp_path):
    data={'source':'metering-service','scope':{'period':'documented interval'},'readings':[{'amount':12.5}]}
    make_page(tmp_path,data)
    assert 'same facts and' in snapshot_reuse.instructions(tmp_path,'index.html')
    assert 'not evidence of fresh service access' in snapshot_reuse.instructions(tmp_path,'index.html')
    (tmp_path/'observed-data.json').write_text(json.dumps({**data,'readings':[{'amount':15}]}))
    assert snapshot_reuse.instructions(tmp_path,'index.html')==''


@pytest.mark.parametrize('existing',['absent','unbound','invalid','credentials'])
def test_unverified_or_incomplete_artifacts_get_normal_collection(tmp_path,existing):
    data={'amount':12.5}
    make_page(tmp_path,data)
    if existing=='absent':(tmp_path/'index.html').unlink()
    elif existing=='unbound':(tmp_path/'index.html').write_text('<html><body>No data binding</body></html>')
    elif existing=='invalid':(tmp_path/'observed-data.json').write_text('not JSON')
    else:make_page(tmp_path,{'access_token':'not-for-page-data'})
    assert snapshot_reuse.instructions(tmp_path,'index.html')==''


def test_reuse_cannot_follow_an_outside_page_symlink(tmp_path):
    folder=tmp_path/'artifact';folder.mkdir();make_page(folder,{'amount':12.5})
    outside=tmp_path/'outside.html';outside.write_text((folder/'index.html').read_text())
    (folder/'index.html').unlink();(folder/'index.html').symlink_to(outside)
    assert snapshot_reuse.instructions(folder,'index.html')==''
    assert snapshot_reuse.instructions(folder,'../outside.html')==''


def test_json_types_must_match_not_just_python_equality(tmp_path):
    make_page(tmp_path,{'flag':True})
    (tmp_path/'observed-data.json').write_text('{"flag":1}')
    assert snapshot_reuse.instructions(tmp_path,'index.html')==''


async def test_a_cache_hint_does_not_complete_or_skip_collection(tmp_path,monkeypatch):
    from types import SimpleNamespace
    fast_static = pytest.importorskip("anton.core.tools.generate_artifact.fast_static", reason="Static generator integration is added later in this stack")
    from anton.core.tools.generate_artifact.discovery import engine
    from anton.core.tools.generate_artifact.state import GenState
    make_page(tmp_path,{'source':'observed readings','records':[{'amount':12.5}]})
    path=tmp_path/'observed-data.json';before=(path.read_bytes(),path.stat().st_mtime_ns)
    monkeypatch.setenv('ANTON_STATIC_REUSE_SNAPSHOT','on')
    called=[]
    async def gather(state):
        called.append(state)
        assert 'Existing artifact snapshot' in state.pipeline_system
        # A cached file alone does not establish completed collection.
    monkeypatch.setattr(engine,'run_gathering_loop',gather)
    state=GenState(session=SimpleNamespace(history=[]),artifact_path=tmp_path,artifact_type='html-app',
                   slug='readings',user_request='Refresh the readings',act_first=True)
    assert await fast_static.prepare(state) is False
    assert called==[state]
    assert before==(path.read_bytes(),path.stat().st_mtime_ns)
