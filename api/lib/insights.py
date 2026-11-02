from datetime import datetime, timedelta
from . import db


SITUATION_LABELS = {
    'SEHAT': 'Sehat',
    'KRITIS': 'Kritis',
    'HPP_TINGGI': 'HPP Terlalu Tinggi',
    'TRAFFIC_PROBLEM': 'Masalah Traffic',
    'COST_PROBLEM': 'Masalah Biaya Operasional',
    'MARGINAL': 'Margin Tipis',
}

SITUATION_CLASSES = {
    'SEHAT': 'sehat',
    'KRITIS': 'kritis',
    'HPP_TINGGI': 'hpp',
    'TRAFFIC_PROBLEM': 'traffic',
    'COST_PROBLEM': 'cost',
    'MARGINAL': 'marginal',
}


def _fmt(n):
    return f"Rp {n:,.0f}".replace(",", ".")


def build_historical_context(tanggal_iso, days=7):
    empty_meta = {
        'n_days': 0, 'avg_revenue': 0.0, 'avg_expense': 0.0,
        'avg_net': 0.0, 'avg_hpp_ratio': 0.0,
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
    revenues, expenses, nets, hpp_ratios = [], [], [], []

    for r in reports:
        rev = float(r['revenue'] or 0)
        hpp = float(r['hpp'] or 0)
        exp = float(r['expenses'] or 0)
        net = float(r['net_cash_flow'] or 0)
        lines.append(
            f"- {r['tanggal']}: Revenue={_fmt(rev)}, HPP={_fmt(hpp)}, "
            f"Expenses={_fmt(exp)}, Net={_fmt(net)}"
        )
        revenues.append(rev)
        expenses.append(exp)
        nets.append(net)
        if rev > 0:
            hpp_ratios.append(hpp / rev)

    avg_rev = sum(revenues) / len(revenues) if revenues else 0
    avg_exp = sum(expenses) / len(expenses) if expenses else 0
    avg_net = sum(nets) / len(nets) if nets else 0
    avg_hpp_ratio = sum(hpp_ratios) / len(hpp_ratios) if hpp_ratios else 0

    lines.append("")
    lines.append(f"Rata-rata Revenue: {_fmt(avg_rev)}")
    lines.append(f"Rata-rata Expenses: {_fmt(avg_exp)}")
    lines.append(f"Rata-rata Net: {_fmt(avg_net)}")
    if avg_hpp_ratio > 0:
        lines.append(f"Rata-rata HPP ratio: {avg_hpp_ratio * 100:.1f}%")

    try:
        top_menus = db.get_menus_aggregate(from_date, to_date, group_by='matched_name')[:5]
        if top_menus:
            lines.append("")
            lines.append("Top 5 menu berdasarkan revenue (periode ini):")
            for m in top_menus:
                lines.append(
                    f"- {m['group_key']}: qty={m['total_qty']}, "
                    f"revenue={_fmt(float(m['total_revenue']))}, "
                    f"margin={_fmt(float(m['total_margin']))}"
                )
    except Exception:
        pass

    try:
        medians = db.get_expenses_median_by_category(from_date, to_date)
        if medians:
            lines.append("")
            lines.append("Median pengeluaran harian per kategori (patokan normal):")
            for cat, med in medians.items():
                lines.append(f"- {cat}: {_fmt(med)}")
    except Exception:
        pass

    meta = {
        'n_days': len(reports),
        'avg_revenue': avg_rev,
        'avg_expense': avg_exp,
        'avg_net': avg_net,
        'avg_hpp_ratio': avg_hpp_ratio,
    }
    return "\n".join(lines), meta


def compute_bep(fin, avg_expense):
    rev = fin['revenue']
    if rev <= 0:
        return {
            'bep': 0, 'hpp_ratio': 0.0, 'fixed_cost': 0,
            'gap': -int(fin.get('expenses', 0)), 'gap_pct': 0.0,
        }
    hpp_ratio = fin['hpp'] / rev
    fixed_cost = avg_expense if avg_expense and avg_expense > 0 else fin['expenses']
    if hpp_ratio >= 0.99:
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
    }


def classify_situation(fin, bep_info, hist_avg_revenue):
    rev = fin['revenue']
    net = fin['net_cash_flow']
    hpp_ratio = bep_info['hpp_ratio']
    bep = bep_info['bep']

    if rev <= 0:
        return 'KRITIS'
    if net > 0:
        return 'SEHAT'
    if hpp_ratio > 0.68:
        return 'HPP_TINGGI'
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