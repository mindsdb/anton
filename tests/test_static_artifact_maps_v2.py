import copy

from bs4 import BeautifulSoup
import pytest

from anton.core.artifacts.static.data import Evidence, DataError
from anton.core.artifacts.static.document import Document
from anton.core.artifacts.static.render import render


def specimen():
    evidence = Evidence.model_validate({'datasets': {'positions': {'source': {'label': 'Synthetic positions'}, 'rows': [
        {'id': '0001', 'lat': 0, 'lon': 0, 'private': 'PRIVATE_ONLY'},
        {'id': '0002', 'lat': 45, 'lon': 90, 'private': 'PRIVATE_ONLY'},
        {'id': '0003', 'lat': -45, 'lon': -90, 'private': 'PRIVATE_ONLY'},
        {'id': '0004', 'lat': None, 'lon': None, 'private': 'PRIVATE_ONLY'}]}}})
    document = Document.model_validate({'title': 'Reported locations', 'sections': [{'id': 'map', 'blocks': [
        {'type': 'map', 'title': 'Positions', 'layers': [{'label': 'Supplied positions', 'query': 'positions', 'latitude': 'lat', 'longitude': 'lon', 'label_field': 'id'}]}]}]})
    return document, evidence


def test_map_has_authentic_static_shapes_and_unshifted_coordinate_anchors():
    document, evidence = specimen()
    output = render(document, evidence)
    soup = BeautifulSoup(output.html, 'html.parser')
    assert len(soup.select('.geo-land')) == 177
    points = soup.select('.geo-point')
    assert [(float(p['cx']), float(p['cy'])) for p in points] == [(500, 250), (750, 125), (250, 375)]
    assert [(p['data-latitude'], p['data-longitude']) for p in points] == [('0', '0'), ('45', '90'), ('-45', '-90')]
    assert all(p.title for p in points)
    assert len(soup.select('.chart-data tbody tr')) == 4
    assert '0004' in soup.select_one('.chart-data').get_text() and 'Not available' in soup.select_one('.chart-data').get_text()
    assert 'PRIVATE_ONLY' not in output.html and 'private' not in output.manifest['bindings']['_view0'][0]
    assert 'Natural Earth' in soup.select_one('footer').get_text()
    assert output.manifest['bindings']['_view0'][0] == {'lat': 0, 'lon': 0, 'id': '0001'}


@pytest.mark.parametrize('field,value', [('lat', 91), ('lon', -181), ('lat', True), ('lon', '90')])
def test_invalid_coordinates_are_not_silently_plotted(field, value):
    document, evidence = specimen()
    evidence.datasets['positions'].rows[0][field] = value
    with pytest.raises(DataError):
        render(document, evidence)


def test_multiple_layers_named_queries_and_escaping():
    document, evidence = specimen()
    layer = document.sections[0].blocks[0].layers[0]
    query = {'dataset': 'positions', 'columns': ['lat', 'lon', 'id']}
    payload = document.model_dump()
    payload['bindings'] = {'mapped': query}
    payload['sections'][0]['blocks'][0]['layers'] = [{**layer.model_dump(), 'query': 'mapped'},
        {**layer.model_dump(), 'label': 'Reference <script>bad()</script>', 'query': query}]
    output = render(Document.model_validate(payload), evidence)
    soup = BeautifulSoup(output.html, 'html.parser')
    assert len(soup.select('.geo-point')) == 6 and len(soup.select('.chart-data tbody tr')) == 8
    assert not soup.select('.legend script')
    assert 'Reference <script>bad()</script>' in soup.select_one('.legend').get_text()
    assert output.manifest['view_checks'] == []


def test_map_projection_never_changes_source_rows():
    document, evidence = specimen(); before = copy.deepcopy(evidence.model_dump())
    render(document, evidence)
    assert evidence.model_dump() == before
