from datetime import datetime, timedelta
from . import db


SITUATION_LABELS = {
    'SEHAT': 'Sehat',
    'KRITIS': 'Kritis',
    'HPP_TINGGI': 'HPP Terlalu Tinggi',
    'TRAFFIC_PROBLEM': 'Kunjungan Rendah',
    'COST_PROBLEM': 'Biaya Operasional Tinggi',
    'MARGINAL': 'Margin Tipis',
    'STOK_BERLEBIH': 'Belanja Stok Berlebih',
}

SITUATION_CLASSES = {
    'SEHAT': 'sehat',
    'KRITIS': 'kritis',
    'HPP_TINGGI': 'hpp',
    'TRAFFIC_PROBLEM': 'traffic',
    'COST_PROBLEM': 'cost',
    'MARGINAL': 'marginal',
    'STOK_BERLEBIH': 'cost',
}


def _fmt(n):
    return f"Rp {int(n):,}".replace(",", ".")


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


def compute_bep(fin, avg_overhead):
    rev = fin['revenue']
    overhead_aktual = float(fin.get('overhead_expenses') or 0)

    if rev <= 0:
        return {
            'bep': 0, 'hpp_ratio': 0.0, 'fixed_cost': int(overhead_aktual),
            'gap': -int(overhead_aktual), 'gap_pct': 0.0,
            'fixed_cost_source': 'aktual',
        }

    hpp_ratio = fin['hpp'] / rev

    if overhead_aktual > 0:
        fixed_cost = overhead_aktual
        source = 'aktual'
    elif avg_overhead and avg_overhead > 0:
        fixed_cost = avg_overhead
        source = 'historis'
    else:
        fixed_cost = 0
        source = 'nol'

    if hpp_ratio >= 0.99 or fixed_cost <= 0:
        bep = 0
    else:
        bep = fixed_cost / (1 - hpp_ratio)

    gap = rev - bep
    gap_pct = (gap / bep * 100) if bep > 0 else 0.0

    return {
        'bep': int(round(bep)),
        'hpp_ratio': round(hpp_ratio, 4),
        'fixed_cost': int(round(fixed_cost)),
        'gap': int(round(gap)),
        'gap_pct': round(gap_pct, 1),
        'fixed_cost_source': source,
    }


def classify_situation(fin, bep_info, hist_meta):
    rev = fin['revenue']
    op = fin['operating_profit']
    cash = fin['cash_flow']
    hpp_ratio = bep_info['hpp_ratio']
    bep = bep_info['bep']
    hist_hpp = hist_meta.get('avg_hpp_ratio', 0) or 0

    if rev <= 0:
        return 'KRITIS'

    if hpp_ratio > 0.68:
        return 'HPP_TINGGI'
    if hist_hpp > 0 and hpp_ratio > hist_hpp + 0.03 and hpp_ratio > 0.60:
        return 'HPP_TINGGI'

    if op > 0 and cash < 0 and abs(cash) > op * 0.5:
        return 'STOK_BERLEBIH'

    if op > 0:
        return 'SEHAT'

    if bep > 0 and rev < bep * 0.6:
        return 'TRAFFIC_PROBLEM'
    if bep > 0 and rev >= bep * 0.85:
        return 'COST_PROBLEM'
    return 'MARGINAL'


def compute_day_of_week(tanggal_iso):
    HARI = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    try:
        dt = datetime.strptime(tanggal_iso, "%Y-%m-%d")
        return HARI[dt.weekday()]
    except Exception:
        return '-'