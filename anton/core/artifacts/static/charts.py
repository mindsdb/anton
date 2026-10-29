"""Static chart geometry with responsive HTML labels; no browser computation."""
from html import escape

COLORS = ['var(--accent)', '#4787a8', '#a16d39', '#8d6e9d', '#7d8f46', '#bb6e71', '#6f839a', '#a18b64']


def _n(value):
    return f'{value:.12g}'


def _legend(series):
    return '<div class="legend">' + ''.join(
        f'<span><i style="background:{COLORS[i]}"></i>{escape(s["label"])}</span>'
        for i, s in enumerate(series)) + '</div>'


def _ticks(low, high, step, percent):
    result = []
    for i in range(round((high-low)/step)+1):
        value = low+step*i
        if abs(value) < step*1e-9:
            value = 0
        result.append((value, f'{value:.6g}' + ('%' if percent else '')))
    return result


def _x_ticks(items):
    positions = sorted(position for position, _ in items)
    gap = min((b-a for a, b in zip(positions,positions[1:])), default=100)
    return f'<div class="x-ticks" style="--tick-width:{_n(gap*.8)}%;--edge-width:{_n(gap*.45)}%">' + ''.join(
        f'<span class="x-tick {"first" if position == 0 else "last" if position == 100 else ""}" style="left:{_n(position)}%" title="{escape(label)}">{escape(label)}</span>'
        for position, label in items) + '</div>'


def markup(spec, plotted, *, low, high, tick, label, labels):
    """Values and domain have already passed the compiler's numeric checks."""
    rows, series = spec['rows'], spec['series']
    if not rows:
        return '<p>' + escape(labels['no_records']) + '</p>'
    if not any(value is not None for row in plotted for value in row):
        return '<p>' + escape(labels['no_values']) + '</p>'
    if low == high:
        high = low + tick
    extent = high-low
    ticks = _ticks(low, high, tick, series[0]['format']['percent'] != 'none')
    x = lambda value: (value-low)/extent*1000
    y = lambda value: (high-value)/extent*300
    legend = _legend(series)
    if spec['kind'] == 'bar':
        groups = []
        for n, row in enumerate(rows):
            category = str(row[spec['x']]) if row[spec['x']] is not None else labels['missing']
            cells = [f'<span class="category" style="grid-row:span {len(series)}">{escape(category)}</span>']
            for i, s in enumerate(series):
                value, shown = plotted[n][i], spec['display'][n][i]
                description = escape(f'{category}, {s["label"]}: {shown}')
                if value is None:
                    plot = '<span class="bar-missing">—</span>'
                else:
                    grid = ''.join(f'<line x1="{_n(x(v))}" x2="{_n(x(v))}" y1="0" y2="14" class="grid" vector-effect="non-scaling-stroke"/>' for v, _ in ticks)
                    start, end = x(min(0,value)), x(max(0,value))
                    plot = f'<svg viewBox="0 0 1000 14" preserveAspectRatio="none" role="img" aria-label="{description}"><title>{description}</title>{grid}<rect x="{_n(start)}" y="0" width="{_n(max(0,end-start))}" height="14" fill="{COLORS[i]}"/></svg>'
                cells.extend([plot, f'<span class="bar-value">{escape(shown)}</span>'])
            groups.append('<div class="bar-category">' + ''.join(cells) + '</div>')
        value_chars = max(len(shown) for values in spec['display'] for shown in values)
        axis = _x_ticks([((v-low)/extent*100, shown) for v, shown in ticks])
        return legend + f'<div class="bar-chart" style="--bar-value-width:{max(7,min(22,value_chars+1))}ch">' + ''.join(groups) + '<div class="bar-axis"><span></span>' + axis + '<span></span></div></div>' + f'<p class="axis-title">{escape(spec["yLabel"])}</p>'

    scatter = spec['kind'] == 'scatter'
    xs = [float(row[spec['x']]) if scatter else n for n, row in enumerate(rows)]
    min_x, max_x = min(xs), max(xs)
    x_at = lambda value: (value-min_x)/(max_x-min_x)*1000 if max_x != min_x else 500
    grid = ''.join(f'<line x1="0" x2="1000" y1="{_n(y(v))}" y2="{_n(y(v))}" class="grid" vector-effect="non-scaling-stroke"/>' for v, _ in ticks)
    paths, marks = [], []
    for i, s in enumerate(series):
        path, connected = [], False
        for n, row in enumerate(rows):
            value = plotted[n][i]
            if value is None:
                connected = False
                continue
            px, py = x_at(xs[n]), y(value)
            path.append(f'{"L" if connected else "M"}{_n(px)},{_n(py)}')
            connected = True
            category = str(row[spec['x']]) if row[spec['x']] is not None else labels['missing']
            description = escape(f'{category}, {s["label"]}: {spec["display"][n][i]}')
            # A round-capped zero-length line keeps circular marks a constant
            # physical size when the responsive plot scales non-uniformly.
            marks.append(f'<line data-point x1="{_n(px)}" x2="{_n(px)}" y1="{_n(py)}" y2="{_n(py)}" stroke="{COLORS[i]}" stroke-width="{8 if scatter else 6}" stroke-linecap="round" vector-effect="non-scaling-stroke" role="img" aria-label="{description}"><title>{description}</title></line>')
        if not scatter:
            paths.append(f'<path d="{"".join(path)}" fill="none" stroke="{COLORS[i]}" stroke-width="2" vector-effect="non-scaling-stroke"/>')
    if scatter:
        x_ticks = [(50, f'{min_x:.6g}')] if min_x == max_x else [(n*25, f'{min_x+(max_x-min_x)*n/4:.6g}') for n in range(5)]
    else:
        indices = sorted({round(n*(len(rows)-1)/min(4,len(rows)-1)) for n in range(min(5,len(rows)))}) if len(rows)>1 else [0]
        x_ticks = [(x_at(xs[n])/10, str(rows[n][spec['x']]) if rows[n][spec['x']] is not None else labels['missing']) for n in indices]
    y_labels = ''.join(f'<span style="top:{_n(y(v)/3)}%">{escape(shown)}</span>' for v, shown in ticks)
    return (legend + f'<p class="y-title">{escape(spec["yLabel"])}</p><div class="xy-chart"><div class="y-ticks">{y_labels}</div>'
        f'<div class="plot"><svg viewBox="0 0 1000 300" preserveAspectRatio="none" role="group" aria-label="{escape(label)}">{grid}{"".join(paths)}{"".join(marks)}</svg></div>'
        '<span></span>' + _x_ticks(x_ticks) + f'</div><p class="axis-title">{escape(spec["xLabel"])}</p>')
