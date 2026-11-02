from datetime import datetime, timedelta
from . import db
from .menu_data import MENU_DATA

HARI_ISO = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']


def build_learning_context(tanggal_iso):
    try:
        tgl = datetime.strptime(tanggal_iso, "%Y-%m-%d").date()
    except Exception:
        return {}

    from_30 = (tgl - timedelta(days=30)).isoformat()
    to_date = (tgl - timedelta(days=1)).isoformat()

    ctx = {}
    for key, fn in [
        ('unmatched_history', lambda: _unmatched_history(from_30, to_date)),
        ('price_drift', lambda: _price_drift(from_30, to_date)),
        ('menu_momentum', lambda: _menu_momentum(tanggal_iso)),
        ('dow_patterns', lambda: _dow_patterns(from_30, to_date)),
        ('top_drivers', lambda: _top_drivers(from_30, to_date)),
    ]:
        try:
            ctx[key] = fn()
        except Exception:
            ctx[key] = []
    return ctx


def _unmatched_history(from_date, to_date):
    rows = db.get_unmatched_history(from_date, to_date)
    return [{'product': r['product'], 'days': int(r['days'])} for r in rows]


def _price_drift(from_date, to_date):
    rows = db.get_menu_pricing_history(from_date, to_date, min_qty=3)
    master = {m['full_name']: m for m in MENU_DATA}
    drifts = []
    for r in rows:
        name = r['menu']
        m = master.get(name)
        if not m or m['price'] <= 0:
            continue
        avg = float(r['avg_price'] or 0)
        if avg <= 0:
            continue
        gap = (avg - m['price']) / m['price']
        if abs(gap) >= 0.15:
            drifts.append({
                'menu': name,
                'avg_price': int(round(avg)),
                'master_price': m['price'],
                'gap_pct': round(gap * 100, 1),
                'qty_total': int(r['qty'] or 0),
                'days': int(r['days'] or 0),
            })
    drifts.sort(key=lambda x: abs(x['gap_pct']), reverse=True)
    return drifts[:8]


def _menu_momentum(tanggal_iso):
    rows = db.get_menu_momentum(tanggal_iso, this_days=7, prev_days=7)
    out = []
    for r in rows:
        this_q = int(r['this_week_qty'] or 0)
        last_q = int(r['last_week_qty'] or 0)
        if this_q + last_q < 3:
            continue
        delta = ((this_q - last_q) / last_q * 100) if last_q > 0 else None
        out.append({
            'menu': r['menu'],
            'this_week_qty': this_q,
            'last_week_qty': last_q,
            'delta_pct': round(delta, 1) if delta is not None else None,
        })
    out.sort(key=lambda x: x['this_week_qty'], reverse=True)
    return out[:10]


def _dow_patterns(from_date, to_date):
    rows = db.get_revenue_by_dow(from_date, to_date)
    out = []
    for r in rows:
        isodow = int(r['isodow'] or 0)
        if 1 <= isodow <= 7:
            day_name = HARI_ISO[isodow - 1]
        else:
            day_name = str(isodow)
        out.append({
            'day': day_name,
            'avg_revenue': int(float(r['avg_revenue'] or 0)),
            'avg_net': int(float(r['avg_net'] or 0)),
            'n_days': int(r['days'] or 0),
        })
    return out


def _top_drivers(from_date, to_date):
    rows = db.get_menu_revenue_aggregate(from_date, to_date, limit=8)
    return [
        {
            'menu': r['menu'],
            'revenue': int(float(r['total_revenue'] or 0)),
            'qty': int(r['total_qty'] or 0),
            'days_appeared': int(r['days'] or 0),
        }
        for r in rows
    ]


def format_learning_for_prompt(ctx):
    if not ctx:
        return "(belum ada pola historis yang cukup)"

    lines = []

    if ctx.get('top_drivers'):
        lines.append('DRIVER REVENUE TERATAS (30 hari terakhir):')
        for d in ctx['top_drivers'][:6]:
            lines.append(
                f"- {d['menu']}: revenue {d['revenue']:,}, qty {d['qty']}, "
                f"muncul di {d['days_appeared']} hari"
            )

    momentum = ctx.get('menu_momentum') or []
    rising = [m for m in momentum if m['delta_pct'] is not None and m['delta_pct'] > 20]
    falling = [m for m in momentum if m['delta_pct'] is not None and m['delta_pct'] < -20]
    if rising:
        lines.append('')
        lines.append('MENU NAIK (7 hari terakhir):')
        for m in rising[:3]:
            lines.append(
                f"- {m['menu']}: +{m['delta_pct']}% "
                f"(qty {m['last_week_qty']} → {m['this_week_qty']})"
            )
    if falling:
        lines.append('')
        lines.append('MENU TURUN (7 hari terakhir):')
        for m in falling[:3]:
            lines.append(
                f"- {m['menu']}: {m['delta_pct']}% "
                f"(qty {m['last_week_qty']} → {m['this_week_qty']})"
            )

    if ctx.get('dow_patterns'):
        lines.append('')
        lines.append('POLA HARI DALAM SEMINGGU (30 hari):')
        for d in ctx['dow_patterns']:
            lines.append(
                f"- {d['day']}: avg revenue {d['avg_revenue']:,}, "
                f"avg net {d['avg_net']:,} ({d['n_days']} hari)"
            )

    if ctx.get('price_drift'):
        lines.append('')
        lines.append('PRICE DRIFT (avg harga POS vs master menu — perlu validasi):')
        for p in ctx['price_drift'][:5]:
            direction = 'lebih rendah' if p['gap_pct'] < 0 else 'lebih tinggi'
            lines.append(
                f"- {p['menu']}: avg {p['avg_price']:,} vs master {p['master_price']:,} "
                f"({direction} {abs(p['gap_pct'])}%, {p['qty_total']} qty / {p['days']} hari)"
            )

    if ctx.get('unmatched_history'):
        lines.append('')
        lines.append('PRODUK BERULANG KALI TIDAK MATCH (kandidat alias baru):')
        for u in ctx['unmatched_history'][:5]:
            lines.append(f"- '{u['product']}': muncul {u['days']} hari tanpa HPP")

    return '\n'.join(lines) if lines else "(belum ada pola historis yang cukup)"