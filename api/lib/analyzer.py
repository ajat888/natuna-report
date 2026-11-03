import os
import json
import re

from groq import Groq, RateLimitError

from .learning import format_learning_for_prompt
from .menu_data import MENU_DATA

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
    "ATURAN ANGKA (FATAL JIKA DILANGGAR): "
    "(1) Setiap kali menyebut 'harga jual' atau 'naikkan harga', WAJIB gunakan HANYA nilai 'harga_jual_per_unit' dari data menu. "
    "(2) DILARANG MENGgunakan 'total_pendapatan' sebagai 'harga jual'. Total pendapatan adalah qty × harga satuan. "
    "(3) Saat menyarankan bundel, harga bundel WAJIB lebih kecil dari (harga menu A + harga menu B) dan sebutkan perhitungannya. "
    "(4) DILARANG menyebut nominal kerugian/keuntungan yang tidak bisa dihitung dari data yang diberikan. "
    "(5) Saat menyebut kenaikan harga, tuliskan jelas nominal Rp dan persentase; tidak boleh salah skala (jangan menyebut total harian sebagai harga satuan). "
    "ATURAN MENU (FATAL JIKA DILANGGAR): "
    "(1) DILARANG menyarankan 'tambahkan menu baru' atau 'perkenalkan menu X' untuk menu yang SUDAH ADA di DAFTAR MENU. "
    "(2) DILARANG mengarang nama menu, harga menu, HPP, atau persentase margin yang tidak ada di data. "
    "(3) Jika ingin menyorot menu, gunakan nama PERSIS dari data (misal 'Paket Keluarga', 'Cappucino Es'). "
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
    "HARUS MENGEMBALIKAN OUTPUT BERUPA FORMAT JSON YANG VALID SEPERTI YANG DIMINTA."
)


NEW_MENU_KWS = (
    'tambahkan menu', 'tambah menu', 'menu baru', 'perkenalkan menu',
    'memperkenalkan menu', 'menu tambahan',
)
HARGA_UP_KWS = (
    'naikkan harga', 'tingkatkan harga', 'menaikkan harga', 'meningkatkan harga',
    'sesuaikan harga jual', 'menyesuaikan harga jual', 'sesuaikan harga', 'menyesuaikan harga',
)
DISCOUNT_KWS = (
    'diskon', 'potongan harga', 'harga khusus',
)


def _fmt(n):
    return f"Rp {int(n):,}".replace(",", ".")


def _clean(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _norm_text(text):
    if not text:
        return ''
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def _has_any(text, keywords):
    t = (text or '').lower()
    return any(k in t for k in keywords)


def _menu_mentions(text, extra_per_menu=None):
    text_n = _norm_text(text)
    if not text_n:
        return []
    found = set()
    for m in MENU_DATA:
        for cand in (m.get('full_name'), m.get('name')):
            if not cand:
                continue
            cn = _norm_text(cand)
            if len(cn) >= 4 and cn in text_n:
                found.add(m['full_name'])
    if extra_per_menu:
        for p in extra_per_menu:
            for cand in (p.get('product'), p.get('matched_name')):
                if not cand:
                    continue
                cn = _norm_text(cand)
                if len(cn) >= 4 and cn in text_n:
                    found.add(cand)
    return list(found)


def _fallback_actions(fin, situation, bep_info):
    if situation == 'SEHAT':
        return [
            {'aksi': 'Tinjau harga jual 2-3 menu dengan margin tertinggi secara bertahap (maksimal 5% per minggu), pantau volume harian.', 'alasan': 'Menaikkan margin absolut tanpa mengganggu volume penjualan.'},
            {'aksi': 'Rancang program kunjungan ulang (kartu loyalitas sederhana) untuk pelanggan yang datang minggu ini.', 'alasan': 'Memperluas basis pelanggan dengan biaya rendah.'},
            {'aksi': 'Catat menu yang sering dipesan bersamaan (makanan + minuman), lalu buat paket bersyarat dengan harga khusus lebih hemat.', 'alasan': 'Meningkatkan nilai transaksi per pelanggan.'},
            {'aksi': 'Evaluasi jam ramai dan sepi, lalu jadwalkan promosi ringan khusus di jam sepi.', 'alasan': 'Meningkatkan kunjungan di jam sepi tanpa memotong margin di jam ramai.'},
        ]
    if situation == 'STOK_BERLEBIH':
        return [
            {'aksi': 'Batasi belanja bahan besok maksimal 80% dari pembelian hari ini; prioritaskan bahan yang cepat habis.', 'alasan': 'Menyesuaikan belanja dengan penjualan aktual.'},
            {'aksi': 'Audit stok fisik akhir hari untuk semua bahan protein dan sayur; catat sisa untuk menu besok.', 'alasan': 'Mencegah pemborosan dan pembelian dobel.'},
            {'aksi': 'Olah bahan berlebih menjadi paket menu hemat dengan harga khusus hari ini saja.', 'alasan': 'Mengubah stok menjadi pendapatan sebelum rusak.'},
            {'aksi': 'Tunda pembelian bahan kering non-urgent (kemasan, bumbu) hingga stok menipis.', 'alasan': 'Menjaga arus kas tetap positif.'},
        ]
    if situation == 'TRAFFIC_PROBLEM':
        return [
            {'aksi': 'Aktifkan pelanggan lama lewat pesan singkat dengan penawaran menu andalan harga khusus 3 hari.', 'alasan': 'Menarik kunjungan tanpa biaya iklan besar.'},
            {'aksi': 'Buka jam promo ringan 2 jam di jam sepi dengan paket minuman + makanan.', 'alasan': 'Meningkatkan kunjungan di jam sepi.'},
            {'aksi': 'Sebar konten promosi singkat (foto menu + harga) ke media sosial dan grup pelanggan.', 'alasan': 'Memperluas jangkauan pelanggan baru.'},
            {'aksi': 'Buat paket berbagi (2-4 orang) dengan harga hemat untuk mendorong kunjungan berkelompok.', 'alasan': 'Meningkatkan nilai transaksi per meja.'},
        ]
    if situation == 'HPP_TINGGI':
        return [
            {'aksi': 'Negosiasi harga bahan pokok (protein, minyak) ke pemasok; target penurunan 5-10%.', 'alasan': 'Menurunkan HPP tanpa mengubah porsi.'},
            {'aksi': 'Standarisasi porsi bahan per porsi menu; timbang dan catat untuk 3 menu terlaris.', 'alasan': 'Mencegah pemborosan bahan tanpa terasa pelanggan.'},
            {'aksi': 'Perbarui harga jual menu dengan rasio HPP di atas 55% secara bertahap maksimal 5% per minggu.', 'alasan': 'Menyesuaikan harga dengan biaya bahan aktual.'},
            {'aksi': 'Review pemborosan dapur harian (sisa bahan, minyak, bumbu); catat sisa yang tidak terpakai.', 'alasan': 'Mengurangi biaya bahan yang tidak menghasilkan pendapatan.'},
        ]
    if situation == 'COST_PROBLEM':
        return [
            {'aksi': 'Audit pemakaian listrik harian (nyalakan AC/lampu sesuai jam operasional saja).', 'alasan': 'Menurunkan beban non-bahan tanpa mengganggu operasional.'},
            {'aksi': 'Hitung ulang pemakaian gas per hari dan sesuaikan jadwal masak untuk hindari pemborosan.', 'alasan': 'Efisiensi biaya operasional langsung.'},
            {'aksi': 'Bandingkan harga kemasan (cup, kotak makan) dengan 2 pemasok lain minggu ini.', 'alasan': 'Mencari penghematan dari pembelian rutin.'},
            {'aksi': 'Batasi pembelian perlengkapan non-urgent minggu ini; tunda hingga akhir bulan.', 'alasan': 'Menjaga kas tetap sehat.'},
        ]
    if situation == 'MARGINAL':
        return [
            {'aksi': 'Dorong penjualan menu dengan margin tertinggi lewat penempatan di daftar menu dan rekomendasi kasir.', 'alasan': 'Memperbaiki campuran menu tanpa menaikkan harga.'},
            {'aksi': 'Kurangi promosi pada menu dengan margin rendah; alihkan ke menu margin tinggi.', 'alasan': 'Meningkatkan margin rata-rata per transaksi.'},
            {'aksi': 'Tawarkan paket makanan + minuman margin tinggi.', 'alasan': 'Meningkatkan margin total per pelanggan.'},
            {'aksi': 'Evaluasi menu yang jarang terjual; pertimbangkan hapus atau ganti dengan menu margin tinggi.', 'alasan': 'Menyederhanakan menu dan fokus pada margin.'},
        ]
    return [
        {'aksi': 'Hentikan pembelian bahan non-urgent 48 jam ke depan; gunakan stok yang ada.', 'alasan': 'Menyelamatkan kas jangka pendek.'},
        {'aksi': 'Fokus jual menu dengan margin tertinggi dan bahan yang sudah tersedia; hentikan promosi potongan harga.', 'alasan': 'Maksimalkan pendapatan per transaksi.'},
        {'aksi': 'Hubungi pemasok utama untuk negosiasi tempo pembayaran 3-5 hari.', 'alasan': 'Menunda pengeluaran tanpa menghentikan operasional.'},
        {'aksi': 'Aktifkan promosi cepat ke pelanggan lama via pesan singkat untuk 24 jam ke depan.', 'alasan': 'Mendatangkan pendapatan tunai segera.'},
    ]


def _fallback_strategi(fin, situation, bep_info):
    rev = fin['revenue']
    exp = fin['expenses']
    cash = fin['cash_flow']
    gap_pct = bep_info['gap_pct']

    base = (
        f"Pendapatan {_fmt(rev)} dengan pengeluaran {_fmt(exp)} menghasilkan "
        f"kas bersih {_fmt(cash)} ({gap_pct:+.1f}% dari titik impas). "
    )
    if situation == 'SEHAT':
        return base + (
            "Prioritas 3-7 hari ke depan: sesuaikan harga jual menu margin tertinggi secara bertahap "
            "maksimal 5% per minggu sambil memantau volume harian; bangun program kunjungan ulang pelanggan; "
            "uji 1 paket menu hemat di jam sepi. Jika volume turun setelah kenaikan harga, tahan kenaikan berikutnya."
        )
    if situation == 'STOK_BERLEBIH':
        return base + (
            "Prioritas 3-7 hari ke depan: batasi belanja bahan maksimal 80% dari hari ini; audit stok fisik "
            "setiap tutup; olah bahan berlebih menjadi paket hemat; tunda pembelian bahan kering non-urgent. "
            "Target: kas bersih kembali positif dalam 3 hari."
        )
    if situation == 'TRAFFIC_PROBLEM':
        return base + (
            "Prioritas 3-7 hari ke depan: aktifkan pelanggan lama lewat pesan singkat dengan penawaran menu andalan; "
            "buka jam promo ringan di jam sepi; sebar konten promosi ke media sosial. "
            "Target: kunjungan harian naik minimal 20% dibanding minggu ini."
        )
    if situation == 'HPP_TINGGI':
        return base + (
            "Prioritas 3-7 hari ke depan: negosiasi harga bahan pokok dengan pemasok; standarisasi porsi per menu; "
            "perbarui harga jual menu dengan rasio HPP tinggi secara bertahap. "
            "Target: rasio HPP turun di bawah 55% dalam 2 minggu."
        )
    if situation == 'COST_PROBLEM':
        return base + (
            "Prioritas 3-7 hari ke depan: audit pemakaian listrik, gas, kemasan, dan air; bandingkan harga dengan "
            "pemasok lain; tunda pembelian perlengkapan non-urgent. Target: beban non-bahan turun 10% dalam 2 minggu."
        )
    if situation == 'MARGINAL':
        return base + (
            "Prioritas 3-7 hari ke depan: dorong penjualan menu margin tertinggi lewat penempatan di daftar menu dan "
            "rekomendasi kasir; kurangi promosi pada menu margin rendah; evaluasi menu yang jarang terjual. "
            "Target: margin rata-rata per transaksi naik 5% dalam seminggu."
        )
    return base + (
        "Prioritas 3-7 hari ke depan: hentikan pembelian non-urgent 48 jam; fokus jual menu margin tertinggi yang "
        "bahannya tersedia; negosiasi tempo pembayaran dengan pemasok utama; aktifkan promosi cepat ke pelanggan lama. "
        "Target: kas bersih positif dalam 48 jam."
    )


def _fix_analysis(data, fin, situation, bep_info):
    instruksi = data.get('instruksi')
    if not isinstance(instruksi, list):
        instruksi = []

    cleaned = []
    for it in instruksi:
        if not isinstance(it, dict):
            continue
        aksi = (it.get('aksi') or '').strip()
        alasan = (it.get('alasan') or '').strip()
        if aksi and alasan:
            cleaned.append({'aksi': aksi, 'alasan': alasan})
    instruksi = cleaned

    filtered = []
    for it in instruksi:
        if _has_any(it['aksi'], NEW_MENU_KWS):
            continue
        filtered.append(it)
    instruksi = filtered

    harga_up_menus = set()
    discount_menus = set()
    for it in instruksi:
        menus = set(_menu_mentions(it['aksi'], fin['per_menu']))
        if _has_any(it['aksi'], HARGA_UP_KWS):
            harga_up_menus |= menus
        if _has_any(it['aksi'], DISCOUNT_KWS):
            discount_menus |= menus
    conflicts = harga_up_menus & discount_menus
    if conflicts:
        kept = []
        for it in instruksi:
            menus = set(_menu_mentions(it['aksi'], fin['per_menu']))
            if _has_any(it['aksi'], DISCOUNT_KWS) and (menus & conflicts):
                continue
            kept.append(it)
        instruksi = kept

    if len(instruksi) < 4:
        for fb in _fallback_actions(fin, situation, bep_info):
            if len(instruksi) >= 4:
                break
            instruksi.append(fb)

    instruksi = instruksi[:4]
    data['instruksi'] = instruksi

    diag_n = _norm_text(data.get('diagnosis'))
    strat_n = _norm_text(data.get('strategi'))
    if diag_n and strat_n and diag_n == strat_n:
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

    return data


def _build_anomaly_lines(fin):
    lines = []
    attach = fin.get('attach_rate', 0) or 0
    if attach > 1.5:
        lines.append(
            f"- Rasio minuman per makanan {attach*100:.1f}% SANGAT TINGGI (normal 60-120%). "
            f"Kemungkinan ada event khusus, promosi besar, atau kesalahan input kasir. "
            f"Sebutkan sebagai catatan, JANGAN jadikan alasan untuk menaikkan harga minuman."
        )
    elif 0 < attach < 0.3:
        lines.append(
            f"- Rasio minuman per makanan {attach*100:.1f}% SANGAT RENDAH (normal 60-120%). "
            f"Pelanggan kurang memesan minuman; fokus promosi tambahan minuman."
        )

    single_qty = sum(1 for p in fin['per_menu'] if p['qty'] == 1)
    if single_qty >= 15:
        lines.append(
            f"- Ada {single_qty} menu yang hanya terjual 1 unit hari ini. "
            f"Tanda daftar menu terlalu banyak atau kunjungan tersebar. Pertimbangkan evaluasi menu di diagnosis."
        )

    top_qty = max((p['qty'] for p in fin['per_menu']), default=0)
    if top_qty >= 80:
        top_prod = next((p['product'] for p in fin['per_menu'] if p['qty'] == top_qty), '')
        lines.append(
            f"- Menu '{top_prod}' terjual {top_qty} unit (sangat tinggi). "
            f"Konfirmasi bukan dobel input; jika valid, ini andalan volume hari ini."
        )

    return lines


def _get_available_models(client):
    try:
        models_response = client.models.list()
        available_ids = [m.id for m in models_response.data]
    except Exception as e:
        print(f"[AI] Gagal mengambil daftar model: {e}. Menggunakan daftar default.")
        return ["openai/gpt-oss-120b", "llama3-70b-8192", "llama-3.1-8b-instant"]

    preferred_order = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]

    valid_models = []
    for pm in preferred_order:
        if pm in available_ids:
            valid_models.append(pm)

    if not valid_models:
        for mid in available_ids:
            if not any(x in mid.lower() for x in ['whisper', 'tts', 'guard', 'playai', 'orpheus']):
                valid_models.append(mid)

    return valid_models


def _extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation, situation_label, hist_text, learning_ctx):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    menu_lines = []
    for p in sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None],
        key=lambda x: x['margin'], reverse=True
    ):
        up = p.get('unit_price') or 0
        mp = p.get('master_price')
        mp_str = _fmt(mp) if mp else '-'
        menu_lines.append(
            f"- {p['product']}: qty={p['qty']}, harga_jual_per_unit={_fmt(up)}, "
            f"harga_acuan_menu={mp_str}, total_pendapatan={_fmt(p['revenue'])}, "
            f"laba_total={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
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

    anomaly_lines = _build_anomaly_lines(fin)
    learning_text = format_learning_for_prompt(learning_ctx)

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN HARI INI
- Total Pendapatan (Uang Masuk): {_fmt(fin['revenue'])}
- Total Pengeluaran (Uang Keluar): {_fmt(fin['expenses'])}
- KAS BERSIH: {_fmt(fin['cash_flow'])} ({status_cash})
- Rasio minuman per makanan: {attach_rate_pct:.1f}%

ANOMALI OPERASIONAL (WAJIB DIBACA SEBELUM MENYUSUN AKSI):
{chr(10).join(anomaly_lines) if anomaly_lines else '- tidak ada anomali signifikan'}

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
Setiap baris memuat harga_jual_per_unit (HARGA SATUAN aktual dari kasir) DAN total_pendapatan (total harian).
WAJIB pakai harga_jual_per_unit saat menyarankan penyesuaian harga, BUKAN total_pendapatan.
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

ATURAN TAMBAHAN (FATAL JIKA DILANGGAR):
- WAJIB hasilkan TEPAT 4 aksi (bukan 2, bukan 3).
- DILARANG menyarankan 'tambahkan menu baru X' atau 'perkenalkan menu X' untuk menu yang SUDAH ADA di data. Menu baru hanya boleh disarankan jika benar-benar tidak ada di daftar menu.
- DILARANG menyebut nominal harga jual selain nilai 'harga_jual_per_unit' dari baris menu di atas. Jangan gunakan 'total_pendapatan' sebagai harga satuan.
- DILARANG menyarankan KENAIKAN HARGA dan DISKON sekaligus pada menu yang SAMA dalam satu laporan. Pilih salah satu arah.
- Jika ada PRODUK TANPA HPP berulang di pola pembelajaran, boleh tambah SATU aksi soal penambahan alias menu.
- Jika ada ANOMALI HARGA signifikan, boleh tambah SATU aksi soal validasi harga kasir.
- Jangan menyebut rasio minuman selain {attach_rate_pct:.1f}%.
- DILARANG menulis kata Inggris dalam nilai JSON (kecuali nama produk/menu dan singkatan HPP).
- Strategi HARUS BERBEDA dari Diagnosis. Jangan menyalin diagnosis ke strategi.
"""

    available_models = _get_available_models(client)
    print(f"[AI] Model yang akan dicoba berurutan: {available_models}")

    data = None
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
                max_tokens=3000,
            )
            raw_text = resp.choices[0].message.content
            parsed_data = _extract_json(raw_text)
            if parsed_data:
                data = parsed_data
                print(f"[AI] Berhasil menggunakan model: {model_name}")
                break
            else:
                print(f"[AI] Model {model_name} gagal menghasilkan JSON yang valid.")
                last_error = Exception("Gagal parse JSON")
                continue
        except RateLimitError as e:
            print(f"[AI] Model {model_name} kena limit kuota (429). Beralih ke model cadangan...")
            last_error = e
            continue
        except Exception as e:
            print(f"[AI] Error pada model {model_name}: {e}")
            last_error = e
            continue

    if data is None:
        raise Exception(f"Semua model AI gagal. Error terakhir: {last_error}")

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

    if not data.get('headline'):
        data['headline'] = (
            f"Pendapatan hari ini {_fmt(fin['revenue'])} "
            f"{'di bawah' if bep_info['gap'] < 0 else 'di atas'} "
            f"Titik Impas Kas {_fmt(bep_info['bep'])} "
            f"(selisih {_fmt(bep_info['gap'])})."
        )

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['cash_flow'] < 0 else "Untung"

    if not data.get('strategi'):
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

    data = _fix_analysis(data, fin, situation, bep_info)

    return data