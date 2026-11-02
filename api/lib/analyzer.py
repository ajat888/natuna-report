import os
import json
import re

from groq import Groq

from .learning import format_learning_for_prompt


SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda: menganalisa data keuangan harian dan memberikan instruksi taktis (Rencana Aksi) beserta edukasi strateginya. "
    "ATURAN BAHASA (WAJIB DIPATUHI): "
    "Seluruh isi jawaban HARUS dalam Bahasa Indonesia yang baku dan mudah dipahami pemilik usaha. "
    "DILARANG menggunakan kata atau istilah Inggris dalam teks jawaban, termasuk namun tidak terbatas pada: "
    "revenue, traffic, cost, margin, bundle, bundling, promo, happy hour, upsell, cross-sell, waste, "
    "action plan, insight, headline, gap, driver, growth, discount, paket bundle, family bundle, dsb. "
    "Gunakan padanan berikut: revenue=pendapatan, traffic=kunjungan, cost=biaya, "
    "margin=laba/margin (margin sudah lazim, tetap boleh), bundle=bundel/paket, promo=promosi, "
    "happy hour=jam promo, upsell=penawaran tambahan, waste=pemborosan, "
    "action plan=rencana aksi, insight=wawasan, headline=ringkasan, gap=selisih, "
    "driver=penggerak, growth=pertumbuhan, discount=potongan harga, BEP=titik impas, "
    "attach rate=rasio minuman per makanan, operating profit=laba operasional, "
    "cash flow=arus kas, overhead=beban non-bahan, stock=stok. "
    "Nama produk/menu (misalnya 'Steak Original', 'Milo Latte') dan singkatan teknis seperti HPP boleh dipertahankan. "
    "ATURAN ANALISIS: "
    "(1) Ringkasan pembuka WAJIB menyebut angka Titik Impas dan selisih dalam 1 kalimat tajam. "
    "(2) Diagnosis WAJIB menyebut SATU akar masalah utama berdasarkan situasi yang diberikan, bukan daftar panjang. "
    "(3) Instruksi harus spesifik dengan angka dan realistis untuk besok. "
    "(4) Manfaatkan pola historis dan konteks pembelajaran jika tersedia — jangan mengarang, jangan mengulang template. "
    "(5) Jangan mengarang nama menu, angka, atau pola yang tidak ada di data. "
    "(6) Jangan menyebut angka rasio minuman per makanan selain yang tertulis di prompt. "
    "(7) Bedakan Laba Operasional (untung/rugi sesungguhnya) dengan Arus Kas Harian (uang masuk dikurangi uang keluar). "
    "Output HARUS berupa JSON yang valid."
)


def _fmt(n):
    return f"Rp {int(n):,}".replace(",", ".")


def _clean(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation_label, hist_text, learning_ctx):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    menu_lines = []
    for p in sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None],
        key=lambda x: x['margin'], reverse=True
    ):
        menu_lines.append(
            f"- {p['product']}: jumlah={p['qty']}, pendapatan={_fmt(p['revenue'])}, "
            f"laba={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
        )

    cls_lines = []
    for cat, items in classification.items():
        if not items:
            continue
        names = ', '.join(
            f"{i['product']} (jumlah {i['qty']}, laba {_fmt(i['margin'])})"
            for i in items[:8]
        )
        cls_lines.append(f"- {cat}: {names}")

    exp_lines = []
    for cat, data in fin['expense_breakdown'].items():
        items_str = ', '.join(
            f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:15]
        )
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_op = "RUGI" if fin['operating_profit'] < 0 else "UNTUNG"
    status_cash = "NEGATIF" if fin['cash_flow'] < 0 else "POSITIF"

    hpp_ratio_pct = bep_info['hpp_ratio'] * 100
    attach_rate_pct = fin.get('attach_rate', 0) * 100

    unmatched_line = ', '.join(fin['unmatched_products']) if fin['unmatched_products'] else '-'
    price_anom_lines = []
    for a in (fin.get('price_anomalies') or [])[:6]:
        price_anom_lines.append(
            f"- {a['product']} → {a['matched_name']}: rata-rata {_fmt(a['avg_price'])} "
            f"vs harga acuan {_fmt(a['master_price'])} ({a['gap_pct']:+.1f}%), "
            f"jumlah={a['qty']}, laba={a['margin'] if a['margin'] is not None else 'tidak tersedia'}"
        )

    learning_text = format_learning_for_prompt(learning_ctx)

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN HARI INI
- Pendapatan: {_fmt(fin['revenue'])}
- HPP (bahan terjual, dari daftar menu acuan): {_fmt(fin['hpp'])} ({hpp_ratio_pct:.1f}% dari pendapatan)
- Laba Kotor: {_fmt(fin['gross_profit'])}
- Beban Non-Bahan (overhead: listrik, gas, cup, tisu, dll): {_fmt(fin['overhead_expenses'])}
- Laba Operasional (pendapatan - HPP - beban non-bahan): {_fmt(fin['operating_profit'])} ({status_op})
- Belanja Stok Bahan (uang keluar untuk beli bahan): {_fmt(fin['bahan_purchases'])}
- Total Belanja Kas (bahan + beban non-bahan): {_fmt(fin['expenses'])}
- Arus Kas Harian (pendapatan - total belanja): {_fmt(fin['cash_flow'])} ({status_cash})

CATATAN PENTING:
Laba Operasional mengukur keuntungan sesungguhnya.
Arus Kas Harian mengukur pergerakan uang di kas/bank hari ini.
Bisa jadi Laba Operasional UNTUNG tetapi Arus Kas NEGATIF jika belanja stok bahan lebih besar dari bahan yang terjual (stok menumpuk).

TITIK IMPAS (WAJIB DIRUJUK DI RINGKASAN & DIAGNOSIS)
- Titik impas pendapatan hari ini: {_fmt(bep_info['bep'])}
- Beban non-bahan harian (patokan): {_fmt(bep_info['fixed_cost'])}
- Selisih pendapatan terhadap titik impas: {_fmt(bep_info['gap'])} ({bep_info['gap_pct']:+.1f}%)
- Situasi (hasil mesin aturan): {situation_label}

RASIO MINUMAN PER MAKANAN HARI INI: {attach_rate_pct:.1f}%
WAJIB: jika instruksi menyebut rasio minuman, gunakan angka PERSIS di atas. Jangan menyebut angka lain.

DETAIL MENU TERJUAL (diurut laba tertinggi):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU:
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN:
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

PRODUK TANPA HPP (tidak masuk hitungan laba kotor):
{unmatched_line}

ANOMALI HARGA (harga kasir tidak sama dengan daftar menu acuan — perlu validasi):
{chr(10).join(price_anom_lines) if price_anom_lines else '- tidak ada'}

{hist_text}

POLA PEMBELAJARAN LINTAS HARI (semakin kaya seiring waktu):
{learning_text}

TUGAS:
Hasilkan JSON PERSIS dengan struktur ini (semua nilai teks dalam Bahasa Indonesia murni, tanpa kata Inggris):
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat tajam menyebut Titik Impas, selisih, dan 1 arahan utama",
  "diagnosis": "2-3 kalimat menyebut SATU akar masalah utama sesuai situasi. Bandingkan dengan tren historis bila ada.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik dengan angka", "alasan": "Kenapa ini menyelesaikan akar masalah"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf gaya Konsultan: hubungkan akar masalah, pola historis, dan langkah 3-7 hari ke depan."
}}

PANDUAN ARAHAN BERDASARKAN SITUASI:
- Sehat: operasional untung dan kas positif. Fokus pengembangan.
- Belanja Stok Berlebih: operasional untung tetapi kas negatif besar. Fokus kurangi pembelian stok.
- Kunjungan Rendah: pendapatan jauh di bawah titik impas. Fokus naikkan transaksi.
- Biaya Operasional Tinggi: beban non-bahan mendekati titik kritis. Fokus audit biaya.
- HPP Terlalu Tinggi: HPP > 68% pendapatan atau naik dari rata-rata. Fokus negosiasi bahan dan reformulasi.
- Margin Tipis: tidak masuk kategori di atas namun operasional rugi. Fokus perbaikan campuran menu.
- Kritis: pendapatan nol atau sangat rendah. Fokus penyelamatan kas darurat.

ATURAN TAMBAHAN:
- Jika ada PRODUK TANPA HPP berulang di pola pembelajaran, sarankan penambahan alias atau penyelarasan daftar menu acuan sebagai salah satu instruksi.
- Jika ada ANOMALI HARGA yang signifikan, sarankan validasi harga kasir atau pembaruan daftar menu.
- Jika rasio minuman per makanan di bawah 60%, sarankan strategi penawaran tambahan minuman.
- DILARANG menulis kata Inggris dalam nilai JSON (kecuali nama produk/menu dan singkatan HPP).
- Jangan menyebut rasio minuman selain {attach_rate_pct:.1f}%.
"""

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        response_format={"type": "json_object"},
        max_tokens=3000,
    )

    data = json.loads(resp.choices[0].message.content)

    for key in ('headline', 'diagnosis', 'strategi'):
        if data.get(key):
            data[key] = _clean(data[key])

    if isinstance(data.get('instruksi'), list):
        for item in data['instruksi']:
            if item.get('aksi'):
                item['aksi'] = _clean(item['aksi'])
            if item.get('alasan'):
                item['alasan'] = _clean(item['alasan'])

    if not data.get('strategi'):
        data['strategi'] = data.get('diagnosis') or 'Belum ada strategi naratif yang dihasilkan.'

    if not data.get('headline'):
        data['headline'] = (
            f"Pendapatan hari ini {_fmt(fin['revenue'])} "
            f"{'di bawah' if bep_info['gap'] < 0 else 'di atas'} "
            f"Titik Impas {_fmt(bep_info['bep'])} "
            f"(selisih {_fmt(bep_info['gap'])})."
        )

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['operating_profit'] < 0 else "Untung"

    if not isinstance(data.get('instruksi'), list) or not data['instruksi']:
        data['instruksi'] = [{
            'aksi': 'Fokus pada penjualan menu bermargin tinggi dan audit biaya operasional.',
            'alasan': 'Menjaga laba operasional tetap positif dan kas tetap sehat.'
        }]

    return data