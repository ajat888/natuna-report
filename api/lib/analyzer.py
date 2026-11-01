import os, json
from groq import Groq

SYSTEM_PROMPT = (
    "Anda adalah konsultan F&B Indonesia yang berpengalaman. "
    "Tugas: menganalisa data keuangan restoran harian dan menghasilkan "
    "rekomendasi actionable dalam JSON valid. Bahasa: Indonesia formal, singkat, to-the-point."
)

def _fmt(n):
    return f"Rp {n:,.0f}".replace(",", ".")

def generate_analysis(fin, classification, tanggal_panjang):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Ringkas per menu (buang item yang margin-nya None)
    menu_lines = []
    for p in sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None],
        key=lambda x: x['margin'], reverse=True
    ):
        menu_lines.append(
            f"- {p['product']}: qty={p['qty']}, revenue={_fmt(p['revenue'])}, "
            f"HPP/unit={_fmt(p['hpp_unit'])}, laba={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
        )

    cls_lines = []
    for cat, items in classification.items():
        if not items:
            continue
        names = ', '.join(f"{i['product']} (qty {i['qty']}, laba {_fmt(i['margin'])})" for i in items[:8])
        cls_lines.append(f"- {cat}: {names}")

    expenses = fin['expense_breakdown']
    exp_lines = []
    for cat, data in expenses.items():
        items_str = ', '.join(f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:15])
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN
- Revenue: {_fmt(fin['revenue'])}
- HPP: {_fmt(fin['hpp'])} ({fin['hpp']/fin['revenue']*100:.1f}% dari revenue)
- Laba Kotor: {_fmt(fin['gross_profit'])}
- Beban Operasional (pembelian/pengeluaran): {_fmt(fin['expenses'])}
- Arus Kas Bersih: {_fmt(fin['net_cash_flow'])}

DETAIL MENU TERJUAL (diurut laba tertinggi):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU (sudah dihitung dengan median qty & margin):
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN:
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

Tugas: hasilkan JSON PERSIS dengan struktur:
{{
  "summary_note": "1 kalimat singkat tentang kondisi arus kas hari ini (mis. defisit karena stocking)",
  "menu_engineering": [
    {{"kategori": "Primadona|Volume Tinggi|Potensial|Evaluasi|Rugi",
      "menus": ["nama1","nama2"],
      "alasan": "mengapa masuk kategori ini (1 kalimat)",
      "rekomendasi": "action item (1-2 kalimat)"}}
  ],
  "cost_control": [
    {{"judul": "...", "detail": "..."}}
  ],
  "sales_promotion": [
    {{"judul": "...", "detail": "..."}}
  ]
}}

Aturan:
- Sertakan kategori hanya bila ada menunya.
- cost_control: 2-4 poin, fokus efisiensi HPP, harga, stok, food waste.
- sales_promotion: 2-4 poin, fokus push menu potensial, bundling, cross-selling.
- Semua angka rupiah pakai format Rp X.XXX.
- Jangan mengarang menu yang tidak ada di data.
"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    return json.loads(resp.choices[0].message.content)