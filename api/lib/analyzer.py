import os, json
from datetime import datetime, timedelta

from groq import Groq

from . import db


SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda: menganalisa data keuangan harian dan memberikan instruksi taktis (Action Plan) beserta edukasi strateginya. "
    "Gunakan bahasa Indonesia yang profesional, lugas, dan berorientasi pada solusi (Action-Oriented). "
    "Jika tersedia data historis, WAJIB bandingkan kondisi hari ini dengan tren beberapa hari terakhir sebelum menyimpulkan. "
    "Output HARUS berupa JSON yang valid."
)


def _fmt(n):
    return f"Rp {n:,.0f}".replace(",", ".")


def _build_historical_context(tanggal_iso, days=7):
    try:
        tgl = datetime.strptime(tanggal_iso, "%Y-%m-%d").date()
        from_date = (tgl - timedelta(days=days)).isoformat()
        to_date   = (tgl - timedelta(days=1)).isoformat()
    except Exception:
        return ""

    try:
        reports = db.get_reports_in_range(from_date, to_date)
    except Exception:
        return ""

    if not reports:
        return ""

    lines = [f"KONTEKS HISTORIS ({days} hari sebelum {tanggal_iso}):"]

    lines.append("Ringkasan harian:")
    revenues, nets = [], []
    for r in reports:
        lines.append(
            f"- {r['tanggal']}: Revenue={_fmt(r['revenue'])}, "
            f"HPP={_fmt(r['hpp'])}, Expenses={_fmt(r['expenses'])}, "
            f"Net={_fmt(r['net_cash_flow'])}"
        )
        revenues.append(float(r['revenue']))
        nets.append(float(r['net_cash_flow']))

    if revenues:
        avg_rev = sum(revenues) / len(revenues)
        avg_net = sum(nets) / len(nets)
        lines.append("")
        lines.append(f"Rata-rata Revenue: {_fmt(avg_rev)}")
        lines.append(f"Rata-rata Net Cash Flow: {_fmt(avg_net)}")

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

    return "\n".join(lines)


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso=None):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    menu_lines = []
    for p in sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None],
        key=lambda x: x['margin'], reverse=True
    ):
        menu_lines.append(
            f"- {p['product']}: qty={p['qty']}, revenue={_fmt(p['revenue'])}, "
            f"laba={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
        )

    cls_lines = []
    for cat, items in classification.items():
        if not items:
            continue
        names = ', '.join(
            f"{i['product']} (qty {i['qty']}, laba {_fmt(i['margin'])})"
            for i in items[:8]
        )
        cls_lines.append(f"- {cat}: {names}")

    expenses = fin['expense_breakdown']
    exp_lines = []
    for cat, data in expenses.items():
        items_str = ', '.join(
            f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:15]
        )
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_net = "RUGI" if fin['net_cash_flow'] < 0 else "UNTUNG"

    hist_text = _build_historical_context(tanggal_iso) if tanggal_iso else ""
    if not hist_text:
        hist_text = "(belum ada data historis — ini laporan pertama atau data historis belum tersedia)"

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.
Status Arus Kas: {status_net} ({_fmt(fin['net_cash_flow'])})

RINGKASAN KEUANGAN
- Revenue: {_fmt(fin['revenue'])}
- HPP: {_fmt(fin['hpp'])} ({fin['hpp']/fin['revenue']*100:.1f}% dari revenue)
- Laba Kotor: {_fmt(fin['gross_profit'])}
- Beban Operasional: {_fmt(fin['expenses'])}

DETAIL MENU TERJUAL (diurut laba tertinggi):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU:
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN:
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

{hist_text}

Tugas: Hasilkan JSON PERSIS dengan struktur ini:
{{
  "kondisi": "Rugi" atau "Untung",
  "diagnosis": "1-2 kalimat diagnosis tajam gaya Manajer. Sebutkan angka defisit/surplus. Jika ada data historis, WAJIB bandingkan dengan tren rata-rata dan sebutkan arah pergerakannya.",
  "instruksi": [
    {{"aksi": "Instruksi taktis spesifik", "alasan": "Penjelasan singkat kenapa aksi ini dilakukan"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf penjelasan gaya Konsultan Bisnis. Jika ada data historis, kaitkan dengan pola berulang."
}}

Aturan Penting:
- Jika kondisi "Rugi", instruksi dan strategi fokus pada PENYELAMATAN KAS.
- Jika kondisi "Untung", instruksi dan strategi fokus pada PENGEMBANGAN.
- Jika ada data historis: bandingkan hari ini dengan rata-rata; identifikasi anomali pengeluaran.
- Berikan 3 sampai 4 poin instruksi yang realistis untuk besok.
- Jangan mengarang nama menu jika tidak ada di data.
"""

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    return json.loads(resp.choices[0].message.content)