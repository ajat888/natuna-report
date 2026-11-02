import os
import json

from groq import Groq

from .learning import format_learning_for_prompt


SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda: menganalisa data keuangan harian dan memberikan instruksi taktis (Action Plan) beserta edukasi strateginya. "
    "Gunakan bahasa Indonesia yang profesional, lugas, dan berorientasi pada solusi. "
    "ATURAN KRITIS: "
    "(1) Headline WAJIB menyebut angka BEP dan gap dalam 1 kalimat tajam. "
    "(2) Diagnosis WAJIB menyebut SATU akar masalah utama berdasarkan 'situation' yang diberikan, bukan daftar panjang. "
    "(3) Instruksi harus spesifik dengan angka dan realistis untuk besok. "
    "(4) Manfaatkan pola historis dan learning context jika tersedia — jangan mengarang, jangan mengulang template. "
    "(5) Jangan mengarang nama menu, angka, atau pola yang tidak ada di data. "
    "Output HARUS berupa JSON valid."
)


def _fmt(n):
    return f"Rp {n:,.0f}".replace(",", ".")


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation, hist_text, learning_ctx):
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

    exp_lines = []
    for cat, data in fin['expense_breakdown'].items():
        items_str = ', '.join(
            f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:15]
        )
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_net = "RUGI" if fin['net_cash_flow'] < 0 else "UNTUNG"

    hpp_ratio_pct = bep_info['hpp_ratio'] * 100
    attach_rate_pct = fin.get('attach_rate', 0) * 100

    unmatched_line = ', '.join(fin['unmatched_products']) if fin['unmatched_products'] else '-'
    price_anom_lines = []
    for a in (fin.get('price_anomalies') or [])[:6]:
        price_anom_lines.append(
            f"- {a['product']} → {a['matched_name']}: avg {a['avg_price']:,} "
            f"vs master {a['master_price']:,} ({a['gap_pct']:+.1f}%), "
            f"qty={a['qty']}, margin={a['margin'] if a['margin'] is not None else 'N/A'}"
        )

    learning_text = format_learning_for_prompt(learning_ctx)

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN
- Revenue: {_fmt(fin['revenue'])}
- HPP: {_fmt(fin['hpp'])} ({hpp_ratio_pct:.1f}% dari revenue)
- Laba Kotor: {_fmt(fin['gross_profit'])}
- Beban Operasional: {_fmt(fin['expenses'])}
- Net Cash Flow: {_fmt(fin['net_cash_flow'])} ({status_net})
- Attach rate minuman/makanan: {attach_rate_pct:.1f}%

BREAK-EVEN POINT (WAJIB DIRUJUK DI HEADLINE & DIAGNOSIS)
- BEP revenue hari ini: {_fmt(bep_info['bep'])}
- Fixed cost harian (median historis): {_fmt(bep_info['fixed_cost'])}
- Gap revenue vs BEP: {_fmt(bep_info['gap'])} ({bep_info['gap_pct']:+.1f}%)
- Situasi (hasil rule-based engine): {situation}

DETAIL MENU TERJUAL (diurut laba tertinggi):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU:
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN:
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

PRODUK TANPA HPP (tidak masuk laba kotor):
{unmatched_line}

PRICE ANOMALY (harga POS ≠ master menu — perlu validasi):
{chr(10).join(price_anom_lines) if price_anom_lines else '- tidak ada'}

{hist_text}

POLA PEMBELAJARAN LINTAS HARI (makin kaya seiring waktu):
{learning_text}

TUGAS:
Hasilkan JSON PERSIS dengan struktur ini:
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat tajam menyebut BEP, gap, dan 1 arahan utama",
  "diagnosis": "2-3 kalimat menyebut SATU akar masalah utama sesuai situation. Bandingkan dengan tren historis bila ada.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik dengan angka", "alasan": "Kenapa ini menyelesaikan akar masalah"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf gaya Konsultan: hubungkan akar masalah, pola historis, dan langkah 3-7 hari ke depan."
}}

ATURAN KHUSUS BERDASARKAN SITUASI:
- TRAFFIC_PROBLEM: fokus menaikkan jumlah transaksi (promo, bundling, jam sepi).
- COST_PROBLEM: fokus memangkas biaya (audit supplier, efisiensi operasional, waste).
- HPP_TINGGI: fokus reformulasi menu, negosiasi bahan, atau kenaikan harga selektif.
- MARGINAL: fokus mix menu — dorong item margin tinggi, kurangi item margin rendah.
- SEHAT: fokus pengembangan (naikkan margin absolut, ekspansi pelanggan).
- KRITIS: fokus penyelamatan kas darurat dalam 24 jam.

ATURAN TAMBAHAN:
- Jika ada PRODUK TANPA HPP berulang di learning context, sarankan penambahan alias/sinkronisasi master menu sebagai salah satu instruksi.
- Jika ada PRICE DRIFT signifikan, sarankan validasi harga POS atau update master.
- Jika attach rate < 60%, sarankan strategi upsell minuman.
- Jangan mengarang angka atau menu di luar data.
"""

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        response_format={"type": "json_object"},
        max_tokens=2200,
    )
    return json.loads(resp.choices[0].message.content)