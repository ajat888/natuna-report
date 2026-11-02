import os
import json
import re

from groq import Groq, RateLimitError

from .learning import format_learning_for_prompt

SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda: menganalisa data keuangan harian dan memberikan instruksi taktis (Rencana Aksi) beserta edukasi strateginya. "
    "ATURAN BAHASA (WAJIB DIPATUHI): "
    "Seluruh isi jawaban HARUS dalam Bahasa Indonesia yang baku dan mudah dipahami pemilik usaha. "
    "DILARANG menggunakan kata atau istilah Inggris dalam teks jawaban, termasuk namun tidak terbatas pada: "
    "revenue, traffic, cost, margin, bundle, bundling, promo, happy hour, upsell, cross-sell, waste, "
    "action plan, insight, headline, gap, driver, growth, discount, paket bundle, family bundle, overhead, stock, dsb. "
    "Gunakan padanan berikut: revenue=pendapatan, traffic=kunjungan, cost=biaya, "
    "margin=laba/margin (margin sudah lazim, tetap boleh), bundle=bundel/paket, promo=promosi, "
    "happy hour=jam promo, upsell=penawaran tambahan, waste=pemborosan, "
    "action plan=rencana aksi, insight=wawasan, headline=ringkasan, gap=selisih, "
    "driver=penggerak, growth=pertumbuhan, discount=potongan harga, BEP=titik impas, "
    "attach rate=rasio minuman per makanan, operating profit=laba operasional, "
    "cash flow=arus kas, overhead=beban non-bahan, stock=stok. "
    "Nama produk/menu (misalnya 'Steak Original', 'Milo Latte') dan singkatan teknis seperti HPP boleh dipertahankan. "
    "ATURAN ANALISIS: "
    "(1) Ringkasan pembuka WAJIB menyebut angka Titik Impas Kas dan selisih dalam 1 kalimat tajam. "
    "(2) Diagnosis WAJIB mengikuti aturan konsistensi status di bawah — tidak boleh bertentangan dengan status situasi. "
    "(3) Instruksi harus spesifik dengan angka dan realistis untuk besok. "
    "(4) Manfaatkan pola historis dan konteks pembelajaran jika tersedia — jangan mengarang, jangan mengulang template. "
    "(5) Jangan mengarang nama menu, angka, atau pola yang tidak ada di data. "
    "(6) Jangan menyebut angka rasio minuman per makanan selain yang tertulis di prompt. "
    "(7) FOKUS UTAMA: 'Bersih' berarti Kas Bersih (Pendapatan dikurangi Total Pengeluaran). Jangan memusingkan pemilik dengan istilah rumit. "
    "ATURAN KONSISTENSI STATUS-DIAGNOSIS (WAJIB): "
    "Situasi menentukan nada diagnosis — tidak boleh bertentangan. "
    "- Jika situasi = Sehat: Diagnosis HARUS DIAWALI dengan 'Operasional hari ini sehat' atau sejenisnya, menyebut kas bersih positif. "
    "  DILARANG memakai kata 'akar masalah', 'masalah utama', 'masalah utama adalah', atau menyebut restoran dalam kondisi bermasalah. "
    "  Jika ada catatan seperti HPP sedikit tinggi atau kas tipis, sebut sebagai 'catatan pengembangan' atau 'peluang optimasi', BUKAN masalah. "
    "- Jika situasi = Belanja Stok Berlebih: Diagnosis fokus pada belanja stok yang menumpuk menyebabkan kas negatif meski operasional untung. "
    "- Jika situasi = Kunjungan Rendah: Diagnosis fokus pendapatan jauh di bawah titik impas. "
    "- Jika situasi = HPP Terlalu Tinggi: Diagnosis fokus HPP melebihi batas sehat atau naik dari biasanya. "
    "- Jika situasi = Biaya Operasional Tinggi: Diagnosis fokus beban non-bahan yang membengkak. "
    "- Jika situasi = Margin Tipis: Diagnosis fokus campuran menu yang membuat laba tipis. "
    "- Jika situasi = Kritis: Diagnosis fokus penyelamatan kas darurat. "
    "ATURAN AKSI-NYAMBUNG-DIAGNOSIS (WAJIB): "
    "Setiap aksi HARUS menjawab akar masalah di diagnosis. Dilarang memberi aksi yang tidak terkait atau berlawanan arah. "
    "- Jika diagnosis Kunjungan Rendah → aksi HANYA tentang: promosi, jam promo, aktifkan pelanggan lama, paket menarik, penawaran jam sepi. "
    "- Jika diagnosis Belanja Stok Berlebih → aksi HANYA tentang: batasi belanja stok, audit stok akhir hari, olah bahan berlebih jadi menu paket, tunda pembelian. "
    "- Jika diagnosis HPP Terlalu Tinggi → aksi HANYA tentang: negosiasi harga bahan, reformulasi porsi, sesuaikan harga jual menu, kurangi pemborosan dapur. "
    "- Jika diagnosis Biaya Operasional Tinggi → aksi HANYA tentang: audit listrik, gas, kemasan, air, efisiensi operasional. "
    "- Jika diagnosis Margin Tipis → aksi HANYA tentang: dorong menu bermargin tinggi, kurangi menu bermargin rendah, sesuaikan campuran menu. "
    "- Jika diagnosis Sehat → aksi HANYA tentang: pengembangan (naikkan margin absolut, perluas pelanggan, promosi terarah), BUKAN pemangkasan darurat. "
    "Jika ada anomali harga atau produk tanpa HPP, boleh disisipkan sebagai SATU aksi tambahan tanpa melanggar aturan di atas. "
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


def _get_available_models(client):
    """Mengambil daftar model yang valid dan tersedia di akun Groq pengguna."""
    try:
        models_response = client.models.list()
        available_ids = [m.id for m in models_response.data]
    except Exception as e:
        print(f"[AI] Gagal mengambil daftar model: {e}. Menggunakan daftar default.")
        return ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]

    # Urutan prioritas model (paling pintar -> paling ringan)
    preferred_order = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "gemma2-9b-it"
    ]

    valid_models = []
    # 1. Masukkan model prioritas yang memang ada di akun Groq user
    for pm in preferred_order:
        if pm in available_ids:
            valid_models.append(pm)
            
    # 2. Jika semua model prioritas di atas tidak ada (aneh), ambil semua model teks yang tersedia
    if not valid_models:
        for mid in available_ids:
            # Abaikan model audio/speech
            if not any(x in mid.lower() for x in ['whisper', 'tts', 'guard', 'playai', 'orpheus']):
                valid_models.append(mid)

    return valid_models


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation, situation_label, hist_text, learning_ctx):
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
- Total Pendapatan (Uang Masuk): {_fmt(fin['revenue'])}
- Total Pengeluaran (Uang Keluar): {_fmt(fin['expenses'])}
- KAS BERSIH: {_fmt(fin['cash_flow'])} ({status_cash})
- Rasio minuman per makanan: {attach_rate_pct:.1f}%

CATATAN AKUNTANSI (Gunakan sebagai wawasan, JANGAN sebut sebagai "Kas Bersih"):
- HPP (bahan yang terjual): {_fmt(fin['hpp'])} ({hpp_ratio_pct:.1f}% dari pendapatan)
- Laba Kotor: {_fmt(fin['gross_profit'])}
- Beban Non-Bahan: {_fmt(fin['overhead_expenses'])}
- Laba Operasional: {_fmt(fin['operating_profit'])}
- Belanja Stok Bahan: {_fmt(fin['bahan_purchases'])}

TITIK IMPAS KAS
- Batas Uang Keluar (Titik Impas Kas) hari ini: {_fmt(bep_info['bep'])}
- Selisih pendapatan terhadap titik impas: {_fmt(bep_info['gap'])} ({bep_info['gap_pct']:+.1f}%)

SITUASI (dari mesin aturan — WAJIB IKUTI):
{situation_label}

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

POLA PEMBELAJARAN LINTAS HARI:
{learning_text}

TUGAS:
Hasilkan JSON PERSIS dengan struktur ini (semua nilai teks Bahasa Indonesia murni, tanpa kata Inggris):
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat tajam menyebut Titik Impas Kas, selisih, dan 1 arahan utama — konsisten dengan situasi",
  "diagnosis": "2-3 kalimat. WAJIB mengikuti aturan konsistensi status: jika Sehat, awali 'Operasional hari ini sehat'; jika situasi lain, sebut SATU akar masalah sesuai situasi.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik dengan angka", "alasan": "Kenapa ini menyelesaikan akar masalah"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf gaya Konsultan: hubungkan situasi, pola historis, dan langkah 3-7 hari ke depan."
}}

PANDUAN AKSI PER SITUASI (WAJIB):
- Sehat → aksi soal pengembangan (naikkan margin absolut, promosi terarah, perluas pelanggan). DILARANG menyuruh memangkas biaya atau menyebut 'masalah'.
- Belanja Stok Berlebih → aksi soal batasi belanja stok, audit stok, olah bahan berlebih jadi menu paket, tunda pembelian.
- Kunjungan Rendah → aksi soal promosi, jam promo, aktifkan pelanggan lama, paket menarik.
- HPP Terlalu Tinggi → aksi soal negosiasi bahan, reformulasi porsi, sesuaikan harga jual.
- Biaya Operasional Tinggi → aksi soal audit listrik, gas, kemasan, air.
- Margin Tipis → aksi soal dorong menu bermargin tinggi, kurangi menu bermargin rendah.
- Kritis → aksi penyelamatan kas darurat 24 jam.

ATURAN TAMBAHAN:
- Jika ada PRODUK TANPA HPP berulang di pola pembelajaran, boleh tambah SATU aksi soal penambahan alias menu.
- Jika ada ANOMALI HARGA signifikan, boleh tambah SATU aksi soal validasi harga kasir.
- Jangan menyebut rasio minuman selain {attach_rate_pct:.1f}%.
- DILARANG menulis kata Inggris dalam nilai JSON (kecuali nama produk/menu dan singkatan HPP).
"""

    available_models = _get_available_models(client)
    print(f"[AI] Model yang akan dicoba berurutan: {available_models}")

    resp = None
    last_error = None

    for model_name in available_models:
        try:
            print(f"[AI] Mencoba generate dengan model: {model_name}")
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
                response_format={"type": "json_object"},
                max_tokens=3000,
            )
            print(f"[AI] Berhasil menggunakan model: {model_name}")
            break
        except RateLimitError as e:
            print(f"[AI] Model {model_name} kena limit kuota (429). Beralih ke model cadangan...")
            last_error = e
            continue
        except Exception as e:
            print(f"[AI] Error pada model {model_name}: {e}")
            last_error = e
            continue

    if resp is None:
        raise Exception(f"Semua model AI sedang sibuk atau limit kuota. Error terakhir: {last_error}")

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

    if situation == 'SEHAT':
        diag = data.get('diagnosis') or ''
        forbidden = ['akar masalah', 'masalah utama', 'masalah utama adalah']
        has_forbidden = any(w in diag.lower() for w in forbidden)
        if has_forbidden or not diag.lower().startswith(('operasional', 'kondisi', 'hari ini')):
            data['diagnosis'] = (
                f"Operasional hari ini sehat: laba operasional {_fmt(fin['operating_profit'])} "
                f"dan kas bersih {_fmt(fin['cash_flow'])}. "
                f"{diag}"
            ).strip()

    if not data.get('strategi'):
        data['strategi'] = data.get('diagnosis') or 'Belum ada strategi naratif yang dihasilkan.'

    if not data.get('headline'):
        data['headline'] = (
            f"Pendapatan hari ini {_fmt(fin['revenue'])} "
            f"{'di bawah' if bep_info['gap'] < 0 else 'di atas'} "
            f"Titik Impas Kas {_fmt(bep_info['bep'])} "
            f"(selisih {_fmt(bep_info['gap'])})."
        )

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['cash_flow'] < 0 else "Untung"

    if not isinstance(data.get('instruksi'), list) or not data['instruksi']:
        data['instruksi'] = [{
            'aksi': 'Fokus pada penjualan menu bermargin tinggi dan audit biaya operasional.',
            'alasan': 'Menjaga laba operasional tetap positif dan kas tetap sehat.'
        }]

    return data