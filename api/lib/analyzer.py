import os
import json
import re
import time

from groq import Groq, RateLimitError

from .learning import format_learning_for_prompt
from .menu_data import MENU_DATA

SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas, cerdas, dan praktis. "
    "Tugas Anda HANYA: menganalisa data (diagnosis) dan memberikan instruksi taktis (rencana aksi) beserta strategi. "
    "ATURAN MATEMATIKA (FATAL JIKA DILANGGAR): "
    "Sistem Python telah menghitung semua matematika (harga, margin, kerugian). "
    "Anda DILARANG KERAS menghitung ulang, menambah, mengurangi, atau mengarang angka baru. "
    "Jika Anda ingin menyarankan kenaikan harga, gunakan PERSIS angka dari bagian 'REKOMENDASI HARGA AMAN' di bawah. "
    "Jika Anda ingin menyebut kerugian/keuntungan, gunakan PERSIS angka dari ringkasan data. "
    "DILARANG menggunakan total_pendapatan sebagai harga satuan. "
    "ATURAN BAHASA (WAJIB): "
    "Gunakan Bahasa Indonesia baku. DILARANG menggunakan kata Inggris: "
    "revenue, traffic, cost, margin, bundle, promo, action plan, insight, gap, driver, growth, discount, overhead, stock, dsb. "
    "Padanan: revenue=pendapatan, traffic=kunjungan, cost=biaya, margin=laba, bundle=paket, "
    "promo=promosi, action plan=rencana aksi, insight=wawasan, gap=selisih, overhead=beban non-bahan. "
    "ATURAN MENU (FATAL JIKA DILANGGAR): "
    "DILARANG menyarankan 'tambahkan menu baru X' untuk menu yang SUDAH ADA di data. "
    "DILARANG mengarang nama menu atau HPP. "
    "ATURAN KONSISTENSI STATUS-DIAGNOSIS (WAJIB): "
    "- Jika situasi = Sehat: Diagnosis HARUS DIAWALI 'Operasional hari ini sehat'. DILARANG memakai kata 'masalah'. "
    "- Jika situasi = Belanja Stok Berlebih: Fokus pada belanja stok menumpuk yang bikin kas negatif. "
    "- Jika situasi = Kunjungan Rendah: Fokus pendapatan di bawah titik impas. "
    "- Jika situasi = HPP Terlalu Tinggi: Fokus HPP melebihi batas. "
    "- Jika situasi = Biaya Operasional Tinggi: Fokus beban non-bahan membengkak. "
    "- Jika situasi = Margin Tipis: Fokus campuran menu bikin laba tipis. "
    "- Jika situasi = Kritis: Fokus penyelamatan kas darurat. "
    "ATURAN AKSI-NYAMBUNG-DIAGNOSIS (WAJIB): "
    "Setiap aksi HARUS menjawab akar masalah di diagnosis. Dilarang berlawanan arah. "
    "DILARANG menyarankan KENAIKAN HARGA dan DISKON pada menu yang SAMA dalam satu laporan. "
    "ATURAN PERBANDINGAN (FATAL): "
    "Jika ada PERBANDINGAN DENGAN LAPORAN SEBELUMNYA, WAJIB sebut minimal 1 delta angka di headline atau diagnosis. "
    "Jangan pernah membandingkan kategori belanja dengan total pengeluaran (apel vs jeruk). Hanya setara dengan setara. "
    "Jika ada AKSI KEMARIN, sebutkan apakah dampaknya terlihat di data hari ini (jangan diamkan). "
    "HARUS MENGEMBALIKAN OUTPUT BERUPA FORMAT JSON YANG VALID SEPERTI YANG DIMINTA."
)

NEW_MENU_KWS = ('tambahkan menu', 'tambah menu', 'menu baru', 'perkenalkan menu', 'memperkenalkan menu')
HARGA_UP_KWS = ('naikkan harga', 'tingkatkan harga', 'menaikkan harga', 'meningkatkan harga', 'sesuaikan harga jual', 'sesuaikan harga')
DISCOUNT_KWS = ('diskon', 'potongan harga', 'harga khusus')

PRUNE_KWS = ('pangkas', 'hapus', 'kurangi menu', 'evaluasi menu', 'menata ulang', 'kurangi biaya', 'potong menu')
GROWTH_KWS = ('tingkatkan', 'kembangkan', 'pertahankan', 'pertumbuhan', 'pelanggan', 'kunjungan', 'promosi', 'loyalitas', 'naikkan pendapatan', 'pelanggan lama')

MODEL_CONTEXT_WINDOW = {
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "deepseek-r1-distill-llama-70b": 131072,
    "qwen-qwq-32b": 131072,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 131072,
    "meta-llama/llama-4-scout-17b-16e-instruct": 131072,
    "openai/gpt-oss-120b": 131072,
    "mixtral-8x7b-32768": 32768,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "gemma2-9b-it": 8192,
}

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
]

MAX_PROMPT_CHARS = 12000
MAX_COMPLETION_TOKENS = 1500


def _fmt(n):
    if n is None:
        return '-'
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


def _truncate(text, max_chars):
    if not text:
        return '-'
    return text[:max_chars] + '\n... (dipotong untuk hemat token)' if len(text) > max_chars else text


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


def _parse_rp(tok):
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


def _fix_comparison_contradictions(text):
    if not text or not isinstance(text, str):
        return text

    rp_pattern = r'rp\s*[\d.,]+\s*(?:juta|jt|rb|ribu)?'
    up_words = r'(melampaui|melebihi|lebih\s+tinggi\s+dari|lebih\s+besar\s+dari)'
    dn_words = r'(di\s+bawah|lebih\s+rendah\s+dari|kurang\s+dari)'

    def _replace(pat, checker, text_in):
        result = []
        last = 0
        for match in re.finditer(pat, text_in, re.IGNORECASE):
            a_tok, b_tok = match.group(1), match.group(3)
            a_val, b_val = _parse_rp(a_tok), _parse_rp(b_tok)
            if a_val is None or b_val is None:
                continue
            if not checker(a_val, b_val):
                continue
            replacement = f"{a_tok} (patokan {b_tok})"
            result.append(text_in[last:match.start()])
            result.append(replacement)
            last = match.end()
        result.append(text_in[last:])
        return ''.join(result)

    pat_up = rf'({rp_pattern})\s*[^.]*?{up_words}\s*[^.]*?({rp_pattern})'
    pat_dn = rf'({rp_pattern})\s*[^.]*?{dn_words}\s*[^.]*?({rp_pattern})'

    text = _replace(pat_up, lambda a, b: a < b, text)
    text = _replace(pat_dn, lambda a, b: a > b, text)

    return text


def _build_math_facts(fin):
    lines = []
    top_menus = sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None and x['qty'] > 0],
        key=lambda x: x['margin'], reverse=True
    )[:5]

    lines.append("REKOMENDASI HARGA AMAN (Kenaikan Maksimal 5%):")
    for p in top_menus:
        up = p.get('unit_price') or 0
        if up > 0:
            safe_price = int(round(up * 1.05))
            lines.append(f"- {p['product']}: Harga saat ini {_fmt(up)} -> Harga aman naik menjadi {_fmt(safe_price)}")

    if fin.get('price_anomalies'):
        lines.append("\nFAKTA ANOMALI HARGA:")
        for a in fin['price_anomalies'][:3]:
            lines.append(f"- {a['product']}: Kasir mencatat {_fmt(a['avg_price'])}, seharusnya {_fmt(a['master_price'])}")
    return "\n".join(lines)


def _fallback_actions(fin, situation, bep_info):
    if situation == 'SEHAT':
        return [
            {'aksi': 'Tinjau harga jual 2-3 menu dengan margin tertinggi secara bertahap (maksimal 5% per minggu), pantau volume harian.', 'alasan': 'Menaikkan margin absolut tanpa mengganggu volume.'},
            {'aksi': 'Rancang program kunjungan ulang (kartu loyalitas sederhana) untuk pelanggan minggu ini.', 'alasan': 'Memperluas basis pelanggan dengan biaya rendah.'},
            {'aksi': 'Catat menu yang sering dipesan bersamaan, lalu buat paket dengan harga khusus lebih hemat.', 'alasan': 'Meningkatkan nilai transaksi per pelanggan.'},
            {'aksi': 'Evaluasi jam ramai dan sepi, lalu jadwalkan promosi ringan di jam sepi.', 'alasan': 'Meningkatkan kunjungan di jam sepi.'},
        ]
    if situation == 'STOK_BERLEBIH':
        return [
            {'aksi': 'Batasi belanja bahan besok maksimal 80% dari pembelian hari ini.', 'alasan': 'Menyesuaikan belanja dengan penjualan aktual.'},
            {'aksi': 'Audit stok fisik akhir hari untuk bahan protein dan sayur; catat sisa.', 'alasan': 'Mencegah pemborosan dan pembelian dobel.'},
            {'aksi': 'Olah bahan berlebih menjadi paket menu hemat hari ini saja.', 'alasan': 'Mengubah stok menjadi pendapatan sebelum rusak.'},
            {'aksi': 'Tunda pembelian bahan kering non-urgent hingga stok menipis.', 'alasan': 'Menjaga arus kas tetap positif.'},
        ]
    if situation == 'TRAFFIC_PROBLEM':
        return [
            {'aksi': 'Aktifkan pelanggan lama lewat pesan singkat dengan penawaran menu andalan.', 'alasan': 'Menarik kunjungan tanpa biaya iklan besar.'},
            {'aksi': 'Buka jam promo ringan 2 jam di jam sepi dengan paket minuman + makanan.', 'alasan': 'Meningkatkan kunjungan di jam sepi.'},
            {'aksi': 'Sebar konten promosi ke media sosial dan grup pelanggan.', 'alasan': 'Memperluas jangkauan pelanggan baru.'},
            {'aksi': 'Buat paket berbagi (2-4 orang) untuk mendorong kunjungan berkelompok.', 'alasan': 'Meningkatkan nilai transaksi per meja.'},
        ]
    if situation == 'HPP_TINGGI':
        return [
            {'aksi': 'Negosiasi harga bahan pokok ke pemasok; target penurunan biaya.', 'alasan': 'Menurunkan HPP tanpa mengubah porsi.'},
            {'aksi': 'Standarisasi porsi bahan per porsi menu; timbang untuk 3 menu terlaris.', 'alasan': 'Mencegah pemborosan bahan.'},
            {'aksi': 'Perbarui harga jual menu dengan rasio HPP tinggi maksimal 5%.', 'alasan': 'Menyesuaikan harga dengan biaya aktual.'},
            {'aksi': 'Review pemborosan dapur harian; catat sisa yang tidak terpakai.', 'alasan': 'Mengurangi biaya bahan terbuang.'},
        ]
    if situation == 'COST_PROBLEM':
        return [
            {'aksi': 'Audit pemakaian listrik harian (nyalakan AC/lampu sesuai jam).', 'alasan': 'Menurunkan beban non-bahan.'},
            {'aksi': 'Hitung ulang pemakaian gas per hari.', 'alasan': 'Efisiensi biaya operasional.'},
            {'aksi': 'Bandingkan harga kemasan dengan 2 pemasok lain.', 'alasan': 'Mencari penghematan pembelian rutin.'},
            {'aksi': 'Batasi pembelian perlengkapan non-urgent minggu ini.', 'alasan': 'Menjaga kas tetap sehat.'},
        ]
    if situation == 'MARGINAL':
        return [
            {'aksi': 'Dorong penjualan menu margin tertinggi lewat rekomendasi kasir.', 'alasan': 'Memperbaiki campuran menu.'},
            {'aksi': 'Kurangi promosi pada menu margin rendah.', 'alasan': 'Meningkatkan margin rata-rata.'},
            {'aksi': 'Tawarkan paket makanan + minuman margin tinggi.', 'alasan': 'Meningkatkan margin total.'},
            {'aksi': 'Evaluasi menu yang jarang terjual.', 'alasan': 'Menyederhanakan menu.'},
        ]
    return [
        {'aksi': 'Hentikan pembelian bahan non-urgent 48 jam ke depan.', 'alasan': 'Menyelamatkan kas jangka pendek.'},
        {'aksi': 'Fokus jual menu margin tinggi dengan bahan tersedia.', 'alasan': 'Maksimalkan pendapatan.'},
        {'aksi': 'Hubungi pemasok untuk negosiasi tempo pembayaran.', 'alasan': 'Menunda pengeluaran.'},
        {'aksi': 'Aktifkan promosi cepat ke pelanggan lama.', 'alasan': 'Mendatangkan pendapatan tunai.'},
    ]


def _fallback_strategi(fin, situation, bep_info):
    rev = fin['revenue']
    cash = fin['cash_flow']
    gap_pct = bep_info['gap_pct']

    base = f"Pendapatan {_fmt(rev)} dengan kas bersih {_fmt(cash)} ({gap_pct:+.1f}% dari total uang keluar). "

    if situation == 'SEHAT':
        return base + "Prioritas 3-7 hari ke depan: sesuaikan harga menu margin tertinggi maksimal 5% sambil pantau volume; bangun program kunjungan ulang; uji paket hemat di jam sepi."
    if situation == 'STOK_BERLEBIH':
        return base + "Prioritas 3-7 hari ke depan: batasi belanja bahan; audit stok fisik; olah bahan berlebih menjadi paket hemat; tunda pembelian bahan kering. Target: kas positif dalam 3 hari."
    if situation == 'TRAFFIC_PROBLEM':
        return base + "Prioritas 3-7 hari ke depan: aktifkan pelanggan lama; buka jam promo di jam sepi; sebar konten promosi. Target: kunjungan naik 20%."
    if situation == 'HPP_TINGGI':
        return base + "Prioritas 3-7 hari ke depan: negosiasi harga bahan; standarisasi porsi; perbarui harga jual secara bertahap. Target: HPP turun di bawah 55%."
    if situation == 'COST_PROBLEM':
        return base + "Prioritas 3-7 hari ke depan: audit listrik, gas, kemasan; bandingkan pemasok; tunda pembelian non-urgent. Target: beban turun 10%."
    if situation == 'MARGINAL':
        return base + "Prioritas 3-7 hari ke depan: dorong menu margin tinggi; kurangi promosi menu margin rendah; evaluasi menu jarang terjual. Target: margin naik 5%."
    return base + "Prioritas 3-7 hari ke depan: hentikan pembelian non-urgent; fokus jual menu margin tinggi; negosiasi tempo bayar; promosi cepat. Target: kas positif dalam 48 jam."


def _sehat_growth_headline(fin, bep_info):
    return (
        f"Operasional hari ini sehat: kas bersih {_fmt(fin['cash_flow'])} "
        f"({bep_info['gap_pct']:+.1f}% dari total uang keluar). "
        f"Fokus: pertahankan momentum lewat promosi menu primadona dan program kunjungan ulang pelanggan."
    )


def _fix_analysis(data, fin, situation, bep_info, yctx=None):
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

    for key in ('headline', 'diagnosis', 'strategi'):
        if data.get(key):
            data[key] = _fix_comparison_contradictions(data[key])
    for it in data.get('instruksi', []):
        if it.get('aksi'):
            it['aksi'] = _fix_comparison_contradictions(it['aksi'])
        if it.get('alasan'):
            it['alasan'] = _fix_comparison_contradictions(it['alasan'])

    if situation == 'SEHAT':
        h = data.get('headline') or ''
        if _has_any(h, PRUNE_KWS) and not _has_any(h, GROWTH_KWS):
            data['headline'] = _sehat_growth_headline(fin, bep_info)

    return data


def _build_anomaly_lines(fin):
    lines = []
    attach = fin.get('attach_rate', 0) or 0
    if attach > 1.5:
        lines.append(f"- Rasio minuman/makanan {attach*100:.1f}% SANGAT TINGGI. Kemungkinan event khusus atau salah input kasir. JANGAN jadikan alasan menaikkan harga.")
    elif 0 < attach < 0.3:
        lines.append(f"- Rasio minuman/makanan {attach*100:.1f}% SANGAT RENDAH. Fokus promosi tambahan minuman.")

    single_qty = sum(1 for p in fin['per_menu'] if p['qty'] == 1)
    if single_qty >= 15:
        lines.append(f"- Ada {single_qty} menu terjual 1 unit. Tanda daftar menu terlalu banyak, pertimbangkan evaluasi menu.")

    top_qty = max((p['qty'] for p in fin['per_menu']), default=0)
    if top_qty >= 80:
        top_prod = next((p['product'] for p in fin['per_menu'] if p['qty'] == top_qty), '')
        lines.append(f"- Menu '{top_prod}' terjual {top_qty} unit (sangat tinggi). Konfirmasi bukan dobel input.")
    return lines


def _get_available_models(client):
    try:
        models_response = client.models.list()
        available_ids = [m.id for m in models_response.data]
    except Exception:
        return PREFERRED_MODELS

    valid = []
    for mid in PREFERRED_MODELS:
        if mid in available_ids:
            valid.append(mid)

    if not valid:
        for mid in available_ids:
            low = mid.lower()
            if any(x in low for x in ['whisper', 'tts', 'guard', 'playai', 'orpheus']):
                continue
            valid.append(mid)
        valid.sort(key=lambda m: MODEL_CONTEXT_WINDOW.get(m, 8192), reverse=True)

    if not valid:
        valid = PREFERRED_MODELS

    return valid


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


def _estimate_tokens(text):
    return len(text) // 3.5


def _trim_prompt(prompt, max_chars):
    if len(prompt) <= max_chars:
        return prompt
    return prompt[:max_chars] + '\n... (prompt dipotong karena batas konteks)'


def _is_context_overflow(err):
    s = str(err).lower()
    return (
        'context_length_exceeded' in s
        or 'reduce the length' in s
        or 'maximum context length' in s
        or 'context window' in s
    )


def _build_prompt(fin, classification, tanggal_panjang, bep_info, situation_label,
                  hist_text, learning_ctx, yctx, anomalies):
    menu_lines = []
    top_menus = sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None],
        key=lambda x: x['margin'], reverse=True
    )[:10]

    for p in top_menus:
        up = p.get('unit_price') or 0
        mp = p.get('master_price')
        mp_str = _fmt(mp) if mp else '-'
        menu_lines.append(
            f"- {p['product']}: qty={p['qty']}, harga={_fmt(up)}, "
            f"acuan={mp_str}, laba={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
        )

    cls_lines = []
    for cat, items in classification.items():
        if not items:
            continue
        names = ', '.join(f"{i['product']}({i['qty']})" for i in items[:3])
        cls_lines.append(f"- {cat}: {names}")

    exp_lines = []
    for cat, data in fin['expense_breakdown'].items():
        items_str = ', '.join(f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:2])
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_cash = "NEGATIF" if fin['cash_flow'] < 0 else "POSITIF"
    hpp_ratio_pct = bep_info['hpp_ratio'] * 100
    attach_rate_pct = fin.get('attach_rate', 0) * 100

    anomaly_lines = _build_anomaly_lines(fin)
    math_facts = _build_math_facts(fin)

    hist_text_short = _truncate(hist_text, 500)
    learning_text_short = _truncate(format_learning_for_prompt(learning_ctx), 500)
    yctx_text = (yctx or {}).get('text') or '(belum ada laporan sebelumnya)'
    yctx_text = _truncate(yctx_text, 900)
    anomaly_signals = anomalies or []

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN (FAKTA):
- Pendapatan: {_fmt(fin['revenue'])}
- Pengeluaran: {_fmt(fin['expenses'])}
- KAS BERSIH: {_fmt(fin['cash_flow'])} ({status_cash})
- Rasio minuman/makanan: {attach_rate_pct:.1f}%

MATEMATIKA YANG SUDAH DIHITUNG OLEH SISTEM:
{math_facts}
(CATATAN: JANGAN menghitung ulang. Jika menyarankan aksi, gunakan angka di atas.)

PERBANDINGAN DENGAN LAPORAN SEBELUMNYA:
{yctx_text}

SINYAL OPERASIONAL (WAJIB DIBAHAS DI DIAGNOSIS JIKA ADA):
{chr(10).join('- ' + a for a in anomaly_signals[:5]) if anomaly_signals else '- tidak ada'}

ANOMALI TEKNIS (cek input kasir):
{chr(10).join(anomaly_lines[:5]) if anomaly_lines else '- tidak ada'}

TOTAL UANG KELUAR HARI INI:
- Total: {_fmt(bep_info['bep'])}
- Kas Bersih: {_fmt(bep_info['gap'])} ({bep_info['gap_pct']:+.1f}%)

SITUASI: {situation_label}
RASIO MINUMAN: {attach_rate_pct:.1f}% (Gunakan angka persis ini jika menyebut rasio).

DETAIL MENU (Top 10):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU:
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN (Top 2 per Kategori):
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

HISTORIS:
{hist_text_short}

POLA PEMBELAJARAN:
{learning_text_short}

TUGAS:
Hasilkan JSON PERSIS (Bahasa Indonesia, tanpa kata Inggris):
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat tajam menyebut Total Uang Keluar, kas bersih, dan 1 arahan utama. Jika ada PERBANDINGAN, sebut minimal 1 delta angka.",
  "diagnosis": "2-3 kalimat. Ikuti aturan konsistensi status. Jika ada SINYAL OPERASIONAL, wajib bahas. Jika ada AKSI KEMARIN, evaluasi dampaknya.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik", "alasan": "Kenapa"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf Konsultan: hubungkan situasi, pola historis, langkah 3-7 hari."
}}

PANDUAN AKSI:
- Sehat → pertumbuhan (margin absolut, promosi, pelanggan, kunjungan ulang). DILARANG memangkas menu atau biaya.
- Belanja Stok Berlebih → batasi belanja, audit stok, olah sisa, tunda beli.
- Kunjungan Rendah → promosi, jam promo, aktifkan pelanggan lama.
- HPP Terlalu Tinggi → negosiasi, reformulasi porsi, sesuaikan harga.
- Biaya Operasional Tinggi → audit listrik, gas, kemasan.
- Margin Tipis → dorong menu margin tinggi, kurangi margin rendah.
- Kritis → penyelamatan kas 24 jam.

ATURAN TAMBAHAN (FATAL):
- WAJIB 4 aksi.
- DILARANG sarankan "tambahkan menu baru X" untuk menu yang SUDAH ADA.
- DILARANG sebut harga/angka yang tidak ada di fakta matematika atas.
- DILARANG naikkan harga DAN diskon pada menu yang sama.
- Strategi HARUS BERBEDA dari Diagnosis.
- Saat menulis "X melampaui Y", pastikan X > Y. Saat menulis "X di bawah Y", pastikan X < Y.
"""

    return _trim_prompt(prompt.strip(), MAX_PROMPT_CHARS)


def _call_groq(client, prompt):
    models = _get_available_models(client)
    last_error = None

    for model_name in models:
        ctx = MODEL_CONTEXT_WINDOW.get(model_name, 8192)
        if ctx <= 8192:
            budget = 6000
        elif ctx <= 32768:
            budget = 10000
        else:
            budget = MAX_PROMPT_CHARS

        current_prompt = _trim_prompt(prompt, budget)

        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": current_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=MAX_COMPLETION_TOKENS,
                )
                raw_text = resp.choices[0].message.content
                parsed = _extract_json(raw_text)
                if parsed:
                    return parsed
                last_error = Exception("Gagal parse JSON")
                break
            except RateLimitError as e:
                if attempt == 0:
                    time.sleep(5)
                    continue
                last_error = e
                break
            except Exception as e:
                if _is_context_overflow(e):
                    current_prompt = _trim_prompt(prompt, max(2000, budget // 2))
                    last_error = e
                    continue
                last_error = e
                break

    raise Exception(f"Semua model AI gagal. Error terakhir: {last_error}")


def _fallback_analysis(fin, situation, bep_info):
    headline = (
        f"Pendapatan hari ini {_fmt(fin['revenue'])} "
        f"{'di bawah' if bep_info['gap'] < 0 else 'di atas'} "
        f"Total Uang Keluar {_fmt(bep_info['bep'])} "
        f"(kas bersih {_fmt(bep_info['gap'])})."
    )
    diagnosis = _fallback_strategi(fin, situation, bep_info)
    return {
        'kondisi': 'Rugi' if fin['cash_flow'] < 0 else 'Untung',
        'headline': headline,
        'diagnosis': diagnosis,
        'instruksi': _fallback_actions(fin, situation, bep_info),
        'strategi': _fallback_strategi(fin, situation, bep_info),
    }


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation, situation_label, hist_text, learning_ctx,
                      yctx=None, anomalies=None):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = _build_prompt(
        fin, classification, tanggal_panjang, bep_info, situation_label,
        hist_text, learning_ctx, yctx, anomalies,
    )

    try:
        data = _call_groq(client, prompt)
    except Exception as e:
        print(f"[analyzer] Semua model gagal: {e}", flush=True)
        data = _fallback_analysis(fin, situation, bep_info)

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
            f"Total Uang Keluar {_fmt(bep_info['bep'])} "
            f"(kas bersih {_fmt(bep_info['gap'])})."
        )

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['cash_flow'] < 0 else "Untung"

    if not data.get('strategi'):
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

    data = _fix_analysis(data, fin, situation, bep_info, yctx=yctx)

    return data