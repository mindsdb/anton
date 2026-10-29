"""Static equirectangular country context and exact, unshifted point anchors."""
from html import escape

from .data import DataError, number
from .geography import world_reference

COLORS = ('var(--accent)', '#bd6f28', '#4e79a7', '#9467bd', '#db4e65', '#5c7e44')


def project(longitude, latitude):
    return (float(longitude) + 180) * 1000 / 360, (90 - float(latitude)) * 500 / 180


def markup(layers, title):
    paths = []
    for feature in world_reference()['data']['features']:
        geometry = feature['geometry']
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        rings = []
        for polygon in polygons:
            for ring in polygon:
                points = [project(lon, lat) for lon, lat, *_ in ring]
                rings.append('M' + 'L'.join(f'{x:.3f},{y:.3f}' for x, y in points) + 'Z')
        paths.append(f'<path class="geo-land" fill-rule="evenodd" d="{"".join(rings)}"><title>{escape(feature["properties"]["name"])}</title></path>')
    grid = ''.join(f'<path d="M0,{project(0, lat)[1]:.3f}H1000"/>' for lat in (-60, -30, 0, 30, 60))
    grid += ''.join(f'<path d="M{project(lon, 0)[0]:.3f},0V500"/>' for lon in (-120, -60, 0, 60, 120))
    points, legend = [], []
    for index, (layer, rows) in enumerate(layers):
        color = COLORS[index]
        legend.append(f'<span><i style="background:{color}"></i>{escape(layer.label)}</span>')
        for row in rows:
            latitude, longitude = row[layer.latitude], row[layer.longitude]
            lat = number(latitude) if latitude is not None else None
            lon = number(longitude) if longitude is not None else None
            if (lat is not None and not -90 <= lat <= 90) or (lon is not None and not -180 <= lon <= 180):
                raise DataError(f'{layer.label}: coordinates outside latitude [-90,90] / longitude [-180,180]')
            if lat is None or lon is None:
                continue  # Missing coordinates remain in the exact inspector, never at (0, 0).
            x, y = project(lon, lat)
            label = row[layer.label_field] if layer.label_field else layer.label
            description = f'{label}: {layer.latitude}={latitude}, {layer.longitude}={longitude}'
            points.append(f'<circle class="geo-point" data-latitude="{lat}" data-longitude="{lon}" cx="{x:.6f}" cy="{y:.6f}" r="8" fill="{color}"><title>{escape(description)}</title></circle>')
    return f'<div class="geo-map" data-native-map role="img" tabindex="0" aria-label="{escape(title)}"><svg viewBox="0 0 1000 500" aria-hidden="true"><g class="geo-grid">{grid}</g>{"".join(paths)}{"".join(points)}</svg></div><div class="legend">{"".join(legend)}</div>'
