import os, json
from groq import Groq

SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda: menganalisa data keuangan harian dan memberikan instruksi taktis (Action Plan) beserta edukasi strateginya. "
    "Gunakan bahasa Indonesia yang profesional, lugas, dan berorientasi pada solusi (Action-Oriented). "
    "Output HARUS berupa JSON yang valid."
)

def _fmt(n):
    return f"Rp {n:,.0f}".replace(",", ".")

def generate_analysis(fin, classification, tanggal_panjang):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Ringkas per menu
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
        names = ', '.join(f"{i['product']} (qty {i['qty']}, laba {_fmt(i['margin'])})" for i in items[:8])
        cls_lines.append(f"- {cat}: {names}")

    expenses = fin['expense_breakdown']
    exp_lines = []
    for cat, data in expenses.items():
        items_str = ', '.join(f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:15])
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    # Tentukan status untung/rugi untuk memandu AI
    status_net = "RUGI" if fin['net_cash_flow'] < 0 else "UNTUNG"

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

Tugas: Hasilkan JSON PERSIS dengan struktur ini:
{{
  "kondisi": "Rugi" atau "Untung",
  "diagnosis": "1-2 kalimat diagnosis tajam gaya Manajer. Sebutkan angka defisit/surplus dan akar masalah/pemicunya.",
  "instruksi": [
    {{"aksi": "Instruksi taktis spesifik (misal: Stop belanja bahan X, Genjot menu Y, Evaluasi porsi Z)", "alasan": "Penjelasan singkat kenapa aksi ini dilakukan"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf penjelasan gaya Konsultan Bisnis. Edukasi kenapa fenomena ini terjadi dan strategi jangka panjang agar bisnis makin sehat (misal: soal timing belanja, margin, atau manajemen stok)."
}}

Aturan Penting:
- Jika kondisi "Rugi", instruksi dan strategi harus fokus pada PENYELAMATAN KAS (menahan belanja, genjot menu margin tinggi).
- Jika kondisi "Untung", instruksi dan strategi harus fokus pada PENGEMBANGAN (amankan stok laris, sisihkan dana darurat, apresiasi tim).
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