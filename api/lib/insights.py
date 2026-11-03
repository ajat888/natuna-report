import re
from datetime import datetime, timedelta
from . import db


SITUATION_LABELS = {
    'SEHAT': 'Sehat',
    'KRITIS': 'Kritis',
    'HPP_TINGGI': 'HPP Terlalu Tinggi',
    'HPP_KRONIS': 'HPP Tinggi Berkelanjutan',
    'TRAFFIC_PROBLEM': 'Kunjungan Rendah',
    'COST_PROBLEM': 'Biaya Operasional Tinggi',
    'MARGINAL': 'Margin Tipis',
    'STOK_BERLEBIH': 'Belanja Stok Berlebih',
}

SITUATION_CLASSES = {
    'SEHAT': 'sehat',
    'KRITIS': 'kritis',
    'HPP_TINGGI': 'hpp',
    'HPP_KRONIS': 'hpp',
    'TRAFFIC_PROBLEM': 'traffic',
    'COST_PROBLEM': 'cost',
    'MARGINAL': 'marginal',
    'STOK_BERLEBIH': 'cost',
}


def _fmt(n):
    if n is None:
        return '-'
    return f"Rp {int(n):,}".replace(",", ".")


def _parse_rp(tok):
    if not tok:
        return None
    tok = tok.strip().lower()
    tok = re.sub(r'\brp\s*', '', tok).strip()
    mult = 1
    if 'juta' in tok or 'jt' in tok:
        mult = 1_000_000
    elif 'rb' in tok or 'ribu' in tok:
        mult = 1_000
    tok = re.sub(r'\s*(juta|jt|rb|ribu)\s*', '', tok).strip()
    tok = tok.replace('.', '').replace(',', '.')
    try:
        return float(tok) * mult
    except ValueError:
        return None


def _get_overhead(r):
    oh = r.get('overhead_total')
    if oh is not None:
        return float(oh)
    exp = float(r.get('expenses') or 0)
    return exp * 0.15


def _get_op_profit(r):
    op = r.get('operating_profit')
    if op is not None:
        return float(op)
    rev = float(r.get('revenue') or 0)
    hpp = float(r.get('hpp') or 0)
    return rev - hpp - _get_overhead(r)


def _verify_single_action(action, fin, yctx):
    text = (action or '').lower()
    if not text:
        return {'action': action, 'status': 'unknown', 'detail': 'Aksi kosong'}

    m = re.search(r'harga\s+(.+?)\s+(?:menjadi|ke)\s+rp\s*([\d.,]+)', text)
    if m:
        target_name = m.group(1).strip()
        target_price = _parse_rp('Rp ' + m.group(2))
        if target_price and target_name:
            for p in fin['per_menu']:
                candidates = [(p.get('matched_name') or '').lower(), (p['product'] or '').lower()]
                if any(target_name in c or c in target_name for c in candidates if c and len(c) >= 4):
                    up = p.get('unit_price') or 0
                    if up > 0:
                        diff = abs(up - target_price)
                        tolerance = max(500, target_price * 0.02)
                        if diff <= tolerance:
                            return {
                                'action': action,
                                'status': 'likely_executed',
                                'detail': f'Harga {p["product"]} saat ini {_fmt(up)} (target {_fmt(target_price)}).',
                            }
                        return {
                            'action': action,
                            'status': 'likely_not_executed',
                            'detail': f'Harga {p["product"]} masih {_fmt(up)}, target {_fmt(target_price)}.',
                        }

    if ('batasi' in text or 'hentikan' in text or 'kurangi' in text) and \
       ('belanja' in text or 'bahan' in text or 'dapur' in text):
        today_bahan = int(fin.get('bahan_purchases', 0))
        prev_bahan = int(yctx.get('prev_bahan_purchases', 0))
        if prev_bahan > 0:
            change_pct = (today_bahan - prev_bahan) / prev_bahan * 100
            if change_pct <= -20:
                return {
                    'action': action,
                    'status': 'likely_executed',
                    'detail': f'Belanja bahan turun {abs(change_pct):.0f}% ({_fmt(prev_bahan)} → {_fmt(today_bahan)}).',
                }
            if change_pct < -5:
                return {
                    'action': action,
                    'status': 'partial',
                    'detail': f'Belanja bahan turun sedikit {abs(change_pct):.0f}% ({_fmt(prev_bahan)} → {_fmt(today_bahan)}).',
                }
            return {
                'action': action,
                'status': 'likely_not_executed',
                'detail': f'Belanja bahan masih {_fmt(today_bahan)} (kemarin {_fmt(prev_bahan)}).',
            }

    if 'audit' in text or 'stok fisik' in text:
        return {'action': action, 'status': 'unknown', 'detail': 'Audit stok tidak dapat diverifikasi dari data penjualan.'}

    return {'action': action, 'status': 'unknown', 'detail': 'Tidak dapat diverifikasi otomatis dari data.'}


def _verify_actions_list(prev_actions, fin, yctx):
    return [_verify_single_action(a, fin, yctx) for a in prev_actions]


def build_yesterday_context(tanggal_iso, today_fin):
    empty = {
        'has_previous': False,
        'text': '(belum ada laporan sebelumnya — ini laporan pertama)',
        'deltas': [],
        'comparisons': [],
        'movers': [],
        'prev_actions': [],
        'prev_headline': '',
        'prev_date': None,
        'prev_revenue': 0,
        'prev_items': 0,
        'prev_cash': 0,
        'prev_attach_rate': 0.0,
        'prev_bahan_purchases': 0,
        'prev_overhead': 0,
        'prev_expenses_total': 0,
        'verifications': [],
    }

    try:
        prev = db.get_previous_report_with_items(tanggal_iso)
    except Exception:
        return empty

    if not prev:
        return empty

    try:
        prev_menus = db.get_report_menus_by_id(prev['id'])
    except Exception:
        prev_menus = []

    prev_rev = float(prev.get('revenue') or 0)
    prev_cash = float(prev.get('net_cash_flow') or 0)
    prev_items = int(prev.get('total_items') or 0)
    prev_ar = float(prev.get('attach_rate') or 0)
    prev_bahan = int(prev.get('bahan_total') or 0)
    prev_overhead = int(prev.get('overhead_total') or 0)
    prev_expenses = int(prev.get('expenses') or 0)

    today_rev = today_fin['revenue']
    today_cash = today_fin['cash_flow']
    today_items = sum(p['qty'] for p in today_fin['per_menu'])
    today_ar = today_fin.get('attach_rate', 0) or 0

    deltas = []
    if prev_rev > 0:
        pct = (today_rev - prev_rev) / prev_rev * 100
        deltas.append({
            'label': 'Pendapatan',
            'prev': int(prev_rev),
            'today': int(today_rev),
            'delta_pct': round(pct, 1),
            'unit': 'rupiah',
        })
    if prev_items > 0 or today_items > 0:
        pct = ((today_items - prev_items) / prev_items * 100) if prev_items > 0 else 0
        deltas.append({
            'label': 'Volume Item',
            'prev': prev_items,
            'today': today_items,
            'delta_pct': round(pct, 1),
            'unit': 'item',
        })
    deltas.append({
        'label': 'Kas Bersih',
        'prev': int(prev_cash),
        'today': int(today_cash),
        'delta': int(today_cash - prev_cash),
        'unit': 'rupiah_delta',
    })
    if prev_ar > 0 or today_ar > 0:
        deltas.append({
            'label': 'Rasio Minuman/Makanan',
            'prev': round(prev_ar * 100, 1),
            'today': round(today_ar * 100, 1),
            'unit': 'pct_point',
        })

    comparisons = []
    for d in deltas:
        unit = d['unit']
        if unit == 'rupiah':
            sign = '+' if d['delta_pct'] >= 0 else ''
            display = f"{_fmt(d['prev'])} → {_fmt(d['today'])} ({sign}{d['delta_pct']}%)"
            positive = d['delta_pct'] >= 0
        elif unit == 'rupiah_delta':
            sign = '+' if d['delta'] >= 0 else ''
            display = f"{_fmt(d['prev'])} → {_fmt(d['today'])} ({sign}{_fmt(d['delta'])})"
            positive = d['delta'] >= 0
        elif unit == 'item':
            sign = '+' if d['delta_pct'] >= 0 else ''
            display = f"{d['prev']} → {d['today']} ({sign}{d['delta_pct']}%)"
            positive = d['delta_pct'] >= 0
        else:
            display = f"{d['prev']}% → {d['today']}%"
            positive = d['today'] >= d['prev']
        comparisons.append({'label': d['label'], 'display': display, 'positive': positive})

    movers = []
    prev_map = {}
    for m in prev_menus:
        name = m.get('menu')
        if name:
            prev_map[name] = int(m.get('qty') or 0)
    today_map = {(p.get('matched_name') or p['product']): p['qty'] for p in today_fin['per_menu']}
    all_names = set(prev_map) | set(today_map)
    for name in all_names:
        pq = prev_map.get(name, 0)
        tq = today_map.get(name, 0)
        if pq + tq < 3:
            continue
        diff = tq - pq
        if abs(diff) >= 2:
            movers.append({'menu': name, 'prev': pq, 'today': tq, 'diff': diff})
    movers.sort(key=lambda x: abs(x['diff']), reverse=True)
    movers = movers[:5]

    prev_analysis = prev.get('analysis_json')
    prev_actions = []
    prev_headline = ''
    if isinstance(prev_analysis, dict):
        prev_headline = prev_analysis.get('headline') or ''
        for item in (prev_analysis.get('instruksi') or [])[:4]:
            if isinstance(item, dict) and item.get('aksi'):
                prev_actions.append(item['aksi'])

    prev_date_str = prev['tanggal'].isoformat() if hasattr(prev['tanggal'], 'isoformat') else str(prev['tanggal'])

    yctx_base = {
        'prev_bahan_purchases': prev_bahan,
        'prev_overhead': prev_overhead,
        'prev_expenses_total': prev_expenses,
    }
    verifications = _verify_actions_list(prev_actions, today_fin, yctx_base) if prev_actions else []

    lines = [f"PERBANDINGAN DENGAN LAPORAN SEBELUMNYA ({prev_date_str}):"]
    for c in comparisons:
        lines.append(f"- {c['label']}: {c['display']}")
    if movers:
        lines.append("")
        lines.append("PERGERAKAN MENU TERBESAR:")
        for m in movers:
            direction = "naik" if m['diff'] > 0 else "turun"
            lines.append(f"- {m['menu']}: {m['prev']} → {m['today']} ({direction} {abs(m['diff'])})")
    if prev_headline:
        lines.append("")
        lines.append(f"HEADLINE LAPORAN KEMARIN: {prev_headline}")
    if verifications:
        lines.append("")
        lines.append("STATUS AKSI KEMARIN (verifikasi otomatis dari data hari ini):")
        status_map = {
            'likely_executed': 'TEREKSKUSI',
            'partial': 'SEBAGIAN',
            'likely_not_executed': 'BELUM DIEKSEKUSI',
            'unknown': 'TIDAK DIVERIFIKASI',
        }
        for i, v in enumerate(verifications, 1):
            lines.append(f"{i}. \"{v['action']}\"")
            lines.append(f"   Status: {status_map.get(v['status'], v['status'])}. {v['detail']}")

    return {
        'has_previous': True,
        'text': '\n'.join(lines),
        'deltas': deltas,
        'comparisons': comparisons,
        'movers': movers,
        'prev_actions': prev_actions,
        'prev_headline': prev_headline,
        'prev_date': prev_date_str,
        'prev_revenue': int(prev_rev),
        'prev_items': prev_items,
        'prev_cash': int(prev_cash),
        'prev_attach_rate': prev_ar,
        'prev_bahan_purchases': prev_bahan,
        'prev_overhead': prev_overhead,
        'prev_expenses_total': prev_expenses,
        'verifications': verifications,
    }


def build_operational_anomalies(fin, tanggal_iso, yctx):
    anomalies = []
    rev = fin['revenue']

    if rev > 0:
        for cat, data in fin['expense_breakdown'].items():
            if cat.lower() in ('dapur', 'bar'):
                ratio = data['total'] / rev
                if ratio > 0.6:
                    anomalies.append(
                        f"Belanja {cat} {_fmt(data['total'])} = {ratio*100:.0f}% dari pendapatan hari ini "
                        f"(normalnya di bawah 60%). Konfirmasi: restock besar atau pemborosan?"
                    )

    try:
        tgl = datetime.strptime(tanggal_iso, "%Y-%m-%d").date()
        from_date = (tgl - timedelta(days=30)).isoformat()
        to_date = (tgl - timedelta(days=1)).isoformat()
        medians = db.get_expense_item_median(from_date, to_date, min_days=3)
    except Exception:
        medians = {}

    for cat, data in fin['expense_breakdown'].items():
        for item in data['items']:
            sub = (item.get('sub') or '').strip().lower()
            if not sub or not medians:
                continue
            med = medians.get(sub)
            if not med or med <= 0:
                continue
            amt = item['amount']
            if amt >= med * 2.5 and (amt - med) >= 100000:
                anomalies.append(
                    f"'{item['sub']}' {_fmt(amt)} — {amt/med:.1f}x median historis {_fmt(med)}. "
                    f"Konfirmasi: belanja besar wajar atau berlebih?"
                )

    if yctx.get('has_previous'):
        prev_items = yctx.get('prev_items', 0)
        today_items = sum(p['qty'] for p in fin['per_menu'])
        if prev_items > 0 and today_items > 0:
            delta_pct = (today_items - prev_items) / prev_items * 100
            if abs(delta_pct) >= 40:
                direction = 'turun' if delta_pct < 0 else 'naik'
                anomalies.append(
                    f"Volume item {direction} {abs(delta_pct):.0f}% vs kemarin "
                    f"({prev_items} → {today_items}). Cek faktor eksternal "
                    f"(cuaca, event, jam operasional, atau perubahan harga)."
                )

        prev_ar = yctx.get('prev_attach_rate', 0)
        today_ar = fin.get('attach_rate', 0) or 0
        if prev_ar > 0 and today_ar > 0:
            diff = today_ar - prev_ar
            if abs(diff) >= 0.2:
                anomalies.append(
                    f"Rasio minuman/makanan bergeser dari {prev_ar*100:.0f}% ke {today_ar*100:.0f}%. "
                    f"Indikasi perubahan pola kunjungan (berkelompok vs sendiri)."
                )

    attach_today = fin.get('attach_rate', 0) or 0
    if attach_today > 1.5:
        anomalies.append(
            f"Rasio minuman/makanan {attach_today*100:.0f}% (di atas 150%) tidak wajar. "
            f"Kemungkinan kesalahan input kasir atau event khusus. Verifikasi data sebelum dijadikan dasar analisis."
        )

    try:
        recent = db.get_recent_price_anomalies(tanggal_iso, days=3)
        counts = {}
        for row in recent:
            for a in (row.get('price_anomalies') or []):
                prod = a.get('product', '')
                if prod:
                    counts[prod] = counts.get(prod, 0) + 1
        for a in (fin.get('price_anomalies') or []):
            prod = a.get('product', '')
            if prod and counts.get(prod, 0) >= 2:
                anomalies.append(
                    f"Harga '{prod}' menyimpang {a['gap_pct']:+.0f}% dari acuan, terjadi berulang selama {counts[prod]+1} hari. "
                    f"Kemungkinan harga acuan yang perlu diperbarui dari {_fmt(a['master_price'])} ke {_fmt(a['avg_price'])} (bukan salah input kasir)."
                )
    except Exception:
        pass

    return anomalies


def build_historical_context(tanggal_iso, days=7):
    empty_meta = {
        'n_days': 0, 'avg_revenue': 0.0, 'avg_overhead': 0.0,
        'avg_op_profit': 0.0, 'avg_hpp_ratio': 0.0,
    }
    try:
        tgl = datetime.strptime(tanggal_iso, "%Y-%m-%d").date()
        from_date = (tgl - timedelta(days=days)).isoformat()
        to_date = (tgl - timedelta(days=1)).isoformat()
    except Exception:
        return "(tanggal tidak valid)", empty_meta

    try:
        reports = db.get_reports_in_range(from_date, to_date)
    except Exception:
        return "(gagal membaca data historis)", empty_meta

    if not reports:
        return "(belum ada data historis — ini laporan pertama atau data historis belum tersedia)", empty_meta

    lines = [f"KONTEKS HISTORIS ({len(reports)} hari sebelum {tanggal_iso}):"]
    revenues, overheads, op_profits, hpp_ratios = [], [], [], []

    for r in reports:
        rev = float(r['revenue'] or 0)
        hpp = float(r['hpp'] or 0)
        oh = _get_overhead(r)
        op = _get_op_profit(r)
        lines.append(
            f"- {r['tanggal']}: Pendapatan={_fmt(rev)}, HPP={_fmt(hpp)}, "
            f"Beban Non-Bahan={_fmt(oh)}, Laba Operasional={_fmt(op)}"
        )
        revenues.append(rev)
        overheads.append(oh)
        op_profits.append(op)
        if rev > 0:
            hpp_ratios.append(hpp / rev)

    avg_rev = sum(revenues) / len(revenues) if revenues else 0
    avg_oh = sum(overheads) / len(overheads) if overheads else 0
    avg_op = sum(op_profits) / len(op_profits) if op_profits else 0
    avg_hpp = sum(hpp_ratios) / len(hpp_ratios) if hpp_ratios else 0

    lines.append("")
    lines.append(f"Rata-rata Pendapatan: {_fmt(avg_rev)}")
    lines.append(f"Rata-rata Beban Non-Bahan: {_fmt(avg_oh)}")
    lines.append(f"Rata-rata Laba Operasional: {_fmt(avg_op)}")
    if avg_hpp > 0:
        lines.append(f"Rata-rata rasio HPP: {avg_hpp * 100:.1f}%")

    try:
        top_menus = db.get_menus_aggregate(from_date, to_date, group_by='matched_name')[:5]
        if top_menus:
            lines.append("")
            lines.append("Lima menu dengan pendapatan tertinggi (periode ini):")
            for m in top_menus:
                lines.append(
                    f"- {m['group_key']}: jumlah={m['total_qty']}, "
                    f"pendapatan={_fmt(float(m['total_revenue']))}, "
                    f"laba={_fmt(float(m['total_margin']))}"
                )
    except Exception:
        pass

    try:
        medians = db.get_expenses_median_by_category(from_date, to_date)
        if medians:
            lines.append("")
            lines.append("Nilai tengah pengeluaran harian per kategori (patokan normal):")
            for cat, med in medians.items():
                lines.append(f"- {cat}: {_fmt(med)}")
    except Exception:
        pass

    meta = {
        'n_days': len(reports),
        'avg_revenue': avg_rev,
        'avg_overhead': avg_oh,
        'avg_op_profit': avg_op,
        'avg_hpp_ratio': avg_hpp,
    }
    return "\n".join(lines), meta


def compute_bep(fin, hist_meta):
    rev = fin['revenue']
    total_expenses = float(fin.get('expenses') or 0)
    hpp_ratio = (fin['hpp'] / rev) if rev > 0 else 0
    bep = total_expenses
    gap = rev - bep
    gap_pct = (gap / bep * 100) if bep > 0 else 0.0

    return {
        'bep': int(round(bep)),
        'hpp_ratio': round(hpp_ratio, 4),
        'fixed_cost': int(total_expenses),
        'gap': int(round(gap)),
        'gap_pct': round(gap_pct, 1),
        'fixed_cost_source': 'kas',
    }


def classify_situation(fin, bep_info, hist_meta):
    rev = fin['revenue']
    op = fin['operating_profit']
    cash = fin['cash_flow']
    hpp_ratio = bep_info['hpp_ratio']
    hist_hpp = hist_meta.get('avg_hpp_ratio', 0) or 0

    if rev <= 0:
        return 'KRITIS'

    if hpp_ratio > 0.65:
        return 'HPP_TINGGI'

    if hist_hpp > 0.60 and hpp_ratio > 0.60 and hist_meta.get('n_days', 0) >= 3:
        return 'HPP_KRONIS'

    if op > 0 and cash < 0 and abs(cash) > op * 0.5:
        return 'STOK_BERLEBIH'

    if op > 0:
        return 'SEHAT'

    if bep_info['bep'] > 0 and rev < bep_info['bep'] * 0.6:
        return 'TRAFFIC_PROBLEM'
    if bep_info['bep'] > 0 and rev >= bep_info['bep'] * 0.85:
        return 'COST_PROBLEM'
    return 'MARGINAL'


def compute_day_of_week(tanggal_iso):
    HARI = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    try:
        dt = datetime.strptime(tanggal_iso, "%Y-%m-%d")
        return HARI[dt.weekday()]
    except Exception:
        return '-'