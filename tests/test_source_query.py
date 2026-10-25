import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest
from anton.core.tools.generate_artifact import source_query as query


@pytest.fixture
def source():
    state=SimpleNamespace(auth=0,queries=[],sink=0,reject=0,token_body=None,
        body={'Columns':[{'ColumnName':'Key','Type':'String'},{'ColumnName':'Amount','Type':'Money'}],
              'Rows':[['A',12.5],['B',None]],'Truncated':False})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            state.sink+=1;self.send_response(200);self.end_headers()
        def do_POST(self):
            raw=self.rfile.read(int(self.headers.get('Content-Length',0)))
            if self.path=='/token':
                state.auth+=1;assert parse_qs(raw.decode())['client_secret']==['fake-secret']
                body=state.token_body or {'access_token':'test-token-'+str(state.auth),'expires_in':3600}
            elif self.path=='/redirect':
                self.send_response(302);self.send_header('Location','/sink');self.end_headers();return
            else:
                state.queries.append({'body':json.loads(raw),'auth':self.headers.get('Authorization'),'path':self.path})
                if state.reject:
                    state.reject-=1;self.send_response(401);self.end_headers();return
                body=state.body
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(body if isinstance(body,bytes) else json.dumps(body).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    client=query.ScenarioRowsClient(base_url=base,token_url=base+'/token',client_id='test-client',client_secret='fake-secret')
    yield state,client
    server.shutdown();server.server_close();thread.join(timeout=2)


def fetch(client,scenario=None):
    return client.query_rows('/query',scenario=scenario or {'Name':'Reference','Scope':'Private'},query='Measurements{Key,Amount}')


def test_normalized_complete_rows_preserve_nulls_and_reuse_one_source_token(source):
    state,client=source
    assert fetch(client)==[{'Key':'A','Amount':12.5},{'Key':'B','Amount':None}]
    assert fetch(client,{'Name':'Alternative','Scope':'Private'})[0]['Amount']==12.5
    assert state.auth==1 and len(state.queries)==2
    assert state.queries[0]['body']=={'Scenarios':[{'Name':'Reference','Scope':'Private'}],'QueryString':'Measurements{Key,Amount}'}
    assert state.queries[0]['auth']=='Bearer test-token-1'
    assert 'fake-secret' not in repr(client) and 'test-token' not in repr(client)


@pytest.mark.parametrize('body',[
    {'error':'not a row response'},
    {'Results':[]},
    {'Columns':[{'ColumnName':'Key'}],'Rows':[['A']],'Truncated':True},
    {'Columns':[{'ColumnName':'Key'}],'Rows':[['A']],'Truncated':'false'},
    {'Columns':[{'ColumnName':'Key'},{'ColumnName':'Key'}],'Rows':[['A','B']]},
    {'Columns':[{'ColumnName':'Key'}],'Rows':[['A',1]]},
    {'Columns':[{'ColumnName':'Key'}],'Rows':[{'Key':'A'}]},
    {'Columns':[{'ColumnName':''}],'Rows':[]},
    b'{"Columns":[],"Rows":[],"bad":NaN}',
    b'not JSON',
])
def test_malformed_truncated_or_unexpected_data_cannot_become_an_empty_dataset(source,body):
    state,client=source;state.body=body
    with pytest.raises(query.SourceQueryError):fetch(client)


def test_documented_empty_result_is_zero_rows(source):
    state,client=source;state.body={'Columns':[{'ColumnName':'Key'}],'Rows':[],'Truncated':False}
    assert fetch(client)==[]


def test_scenario_array_is_rejected_before_any_authentication_or_query(source):
    state,client=source
    with pytest.raises(query.SourceQueryError,match='one scenario'):
        fetch(client,[{'Name':'Reference'},{'Name':'Alternative'}])
    assert state.auth==0 and state.queries==[]


def test_external_path_and_redirect_never_forward_source_credentials(source):
    state,client=source
    for path in ['https://example.invalid/query','//example.invalid/query','http://user:password@127.0.0.1/query']:
        with pytest.raises(query.SourceQueryError):
            client.query_rows(path,scenario={'Name':'Reference'},query='Measurements{Key}')
    with pytest.raises(query.SourceHTTPError,match='302'):
        client.query_rows('/redirect',scenario={'Name':'Reference'},query='Measurements{Key}')
    assert state.sink==0


def test_relative_paths_and_same_origin_urls_follow_standard_url_resolution(source):
    state,client=source
    for path in ['query','/query',client._base+'/query']:
        assert client.query_rows(path,scenario={'Name':'Reference'},query='Measurements{Key}')[0]['Key']=='A'
    assert [q['path'] for q in state.queries]==['/query']*3 and state.auth==1
    client._base+='/service/'
    client.query_rows('query',scenario={'Name':'Reference'},query='Measurements{Key}')
    client.query_rows('/query',scenario={'Name':'Reference'},query='Measurements{Key}')
    assert [q['path'] for q in state.queries[-2:]]==['/service/query','/query']


@pytest.mark.parametrize('timeout',[0,-1,True,float('inf'),None])
def test_invalid_timeout_cannot_create_an_unbounded_client(timeout):
    with pytest.raises(query.SourceQueryError,match='timeout'):
        query.ScenarioRowsClient(base_url='http://127.0.0.1',token_url='http://127.0.0.1/token',client_id='x',client_secret='y',timeout=timeout)


def test_unauthorized_token_is_renewed_at_most_once(source):
    state,client=source;state.reject=1
    assert fetch(client)[0]['Amount']==12.5
    assert state.auth==2 and [x['auth'] for x in state.queries]==['Bearer test-token-1','Bearer test-token-2']
    state.reject=2
    with pytest.raises(query.SourceHTTPError,match='401'):fetch(client)
    assert state.auth==3 and state.reject==0


def test_expiry_refreshes_source_token_without_changing_query(source,monkeypatch):
    state,client=source;clock=[100.0];monkeypatch.setattr(query.time,'monotonic',lambda:clock[0])
    fetch(client);clock[0]+=4000;fetch(client)
    assert state.auth==2 and state.queries[0]['body']==state.queries[1]['body']


def test_response_size_is_bounded(source,monkeypatch):
    state,client=source;fetch(client);state.body=b' '*1025;monkeypatch.setattr(query,'MAX_RESPONSE_BYTES',1024)
    with pytest.raises(query.SourceQueryError,match='exceeds'):fetch(client)


def test_environment_requires_one_explicit_complete_connection(monkeypatch):
    values={'DS_ONE__BASE_URL':'http://127.0.0.1','DS_ONE__TOKEN_URL':'http://127.0.0.1/token',
            'DS_ONE__CLIENT_ID':'client','DS_ONE__CLIENT_SECRET':'private'}
    monkeypatch.setattr(query.os,'environ',dict(values))
    assert repr(query.ScenarioRowsClient.from_environment())=='ScenarioRowsClient()'
    query.os.environ.update({k.replace('DS_ONE','DS_TWO'):v for k,v in values.items()})
    with pytest.raises(query.SourceQueryError,match='exact injected'):query.ScenarioRowsClient.from_environment()
    assert repr(query.ScenarioRowsClient.from_environment('DS_TWO'))=='ScenarioRowsClient()'
