import os
import json
import re
import time

from groq import Groq, RateLimitError

from .learning import format_learning_for_prompt
from .menu_data import MENU_DATA


SYSTEM_PROMPT = (
    "Anda adalah Manajer Operasional Restoran sekaligus Konsultan Bisnis F&B yang tegas dan praktis. "
    "ATURAN MATEMATIKA (FATAL): Jangan hitung ulang, jangan karang angka. "
    "Semua angka sudah dihitung sistem Python. Gunakan PERSIS angka dari prompt. "
    "ATURAN BAHASA (WAJIB): Bahasa Indonesia baku. DILARANG kata Inggris: "
    "revenue, traffic, cost, margin, bundle, promo, action plan, insight, gap, driver, "
    "growth, discount, overhead, stock. Padanan: pendapatan, kunjungan, biaya, laba, paket, "
    "promosi, rencana aksi, wawasan, selisih, beban non-bahan. "
    "ATURAN MENU (FATAL): Dilarang sarankan 'tambahkan menu baru X' jika sudah ada. "
    "Dilarang karang nama menu atau HPP. "
    "ATURAN STATUS (WAJIB): "
    "Sehat=awali 'Operasional hari ini sehat', tanpa kata 'masalah'. "
    "HPP Tinggi Berkelanjutan=fokus perbaikan margin struktural, jangan sebut sehat. "
    "Konsistensi status-diagnosis-aksi wajib. "
    "ATURAN PERBANDINGAN (FATAL): Jika ada data kemarin, wajib sebut minimal 1 delta angka. "
    "Jangan membandingkan kategori belanja vs total (apel vs jeruk). "
    "ATURAN VERIFIKASI (FATAL): Jika aksi kemarin berstatus BELUM DIEKSEKUSI, "
    "DILARANG klaim aksi berhasil atau berkontribusi. "
    "Output akhir HARUS JSON valid."
)

NEW_MENU_KWS = ('tambahkan menu', 'tambah menu', 'menu baru', 'perkenalkan menu', 'memperkenalkan menu')
HARGA_UP_KWS = ('naikkan harga', 'tingkatkan harga', 'menaikkan harga', 'meningkatkan harga', 'sesuaikan harga jual', 'sesuaikan harga')
DISCOUNT_KWS = ('diskon', 'potongan harga', 'harga khusus')
REMOVE_MENU_KWS = ('hapus menu', 'kurangi menu', 'hilangkan menu', 'pangkas menu', 'kurangi item', 'hapus item', 'menghapus menu', 'mengurangi menu')
PRUNE_KWS = ('pangkas', 'hapus', 'kurangi menu', 'evaluasi menu', 'menata ulang', 'kurangi biaya', 'potong menu')
GROWTH_KWS = ('tingkatkan', 'kembangkan', 'pertahankan', 'pertumbuhan', 'pelanggan', 'kunjungan', 'promosi', 'loyalitas', 'naikkan pendapatan', 'pelanggan lama')

MAX_COMPLETION_TOKENS = 2048


def _est_tokens(s):
    if not s:
        return 0
    return max(1, len(s) // 4)


def _is_context_error(e):
    msg = str(e).lower()
    return ('context_length' in msg
            or 'reduce the length' in msg
            or 'too long' in msg
            or 'context window' in msg)


def _trim(text, max_chars):
    if not text:
        return ''
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[:max(0, max_chars - 15)] + '... [dipotong]'


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
    if not tok:
        return None
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


def _text_overlap_ratio(t1, t2, min_word_len=3):
    if not t1 or not t2:
        return 0
    def nw(t):
        return set(w for w in _norm_text(t).split() if len(w) >= min_word_len)
    w1, w2 = nw(t1), nw(t2)
    if not w1 or not w2:
        return 0
    return len(w1 & w2) / len(w1 | w2)


def _sanitize_numeric_claims(text, fin, bep_info):
    if not text or not isinstance(text, str):
        return text
    replacements = [
        (r'(laba operasional)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*-?[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["operating_profit"])}'),
        (r'(kas bersih)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*-?[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["cash_flow"])}'),
        (r'(total uang keluar)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["expenses"])}'),
        (r'(total pengeluaran)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["expenses"])}'),
        (r'(laba kotor)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["gross_profit"])}'),
        (r'(total pendapatan)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["revenue"])}'),
        (r'(pendapatan hari ini)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["revenue"])}'),
        (r'(hpp|harga pokok penjualan)(\s+(?:mencapai|sebesar|senilai|sejumlah|menjadi))?\s+Rp\s*[\d.,]+',
         lambda m: f'{m.group(1)}{m.group(2) or ""} {_fmt(fin["hpp"])}'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _check_remove_action_safety(action_text, fin):
    if not _has_any(action_text, REMOVE_MENU_KWS):
        return True, None
    total_rev = fin['revenue']
    if total_rev <= 0:
        return True, None
    m = re.search(r'(?:qty|terjual|sebanyak)\s+(\d+)', action_text.lower())
    threshold_qty = int(m.group(1)) if m else 1
    high_value = []
    for p in fin['per_menu']:
        if p['qty'] <= threshold_qty:
            pct = p['revenue'] / total_rev * 100
            if pct >= 5.0 and p['revenue'] >= 50000:
                high_value.append(f"{p['product']} ({_fmt(p['revenue'])})")
    if high_value:
        return False, high_value
    return True, None


def _validate_restriction_action(action_text, fin):
    m = re.search(r'(\d+)\s*%', action_text)
    if not m:
        return True, None
    pct = int(m.group(1)) / 100
    if pct >= 1.0:
        return True, None
    target_cat = None
    for cat in fin['expense_breakdown']:
        if cat.lower() in action_text.lower():
            target_cat = cat
            break
    if not target_cat and ('dapur' in action_text.lower() or 'bahan' in action_text.lower()):
        target_cat = next((c for c in fin['expense_breakdown'] if c.lower() in ('dapur', 'bahan')), None)
    if not target_cat:
        return True, None
    target_exp = fin['expense_breakdown'][target_cat]['total']
    other = fin['expenses'] - target_exp
    new_total = target_exp * pct + other
    if fin['revenue'] > 0 and new_total > fin['revenue'] * 1.02:
        deficit = new_total - fin['revenue']
        return False, (
            f"Catatan: dengan target {int(pct*100)}%, total belanja masih {_fmt(new_total)}, "
            f"di atas pendapatan {_fmt(fin['revenue'])} (defisit {_fmt(deficit)})."
        )
    return True, None


def _strip_strategy_from_diagnosis(diagnosis, strategy):
    if not diagnosis:
        return diagnosis, strategy
    marker = re.search(
        r'(\bprioritas\s+3[-\s]7\s+hari|\bdalam\s+3[-\s]7\s+hari|\brencana\s+3[-\s]7\s+hari)',
        diagnosis, re.IGNORECASE,
    )
    if not marker:
        return diagnosis, strategy
    idx = marker.start()
    if idx < 15:
        return diagnosis, strategy
    strat_part = diagnosis[idx:]
    clean_diag = diagnosis[:idx].rstrip(' .,;:')
    if not clean_diag:
        return diagnosis, strategy
    if not strategy or len(strategy) < 40:
        return clean_diag, strat_part
    return clean_diag, strategy


def _ensure_diagnosis_mentions_signals(diagnosis, anomalies, yctx):
    if not diagnosis:
        return diagnosis
    if not anomalies and not (yctx and yctx.get('movers')):
        return diagnosis
    d_norm = _norm_text(diagnosis)
    missing = []
    for a in (anomalies or [])[:2]:
        kw = re.search(r'(Belanja \w+|Volume item|Rasio minuman|Harga \')', a)
        if kw and _norm_text(kw.group(1)) not in d_norm:
            missing.append(a[:70])
    if missing:
        return diagnosis.rstrip(' .,') + ". Sinyal: " + "; ".join(missing) + "."
    return diagnosis


def _build_fallback_headline(fin, bep_info, yctx, situation):
    parts = []
    if yctx and yctx.get('has_previous'):
        prev_rev = yctx.get('prev_revenue', 0)
        if prev_rev > 0:
            dp = (fin['revenue'] - prev_rev) / prev_rev * 100
            arrow = "naik" if dp >= 0 else "turun"
            parts.append(f"Pendapatan {arrow} {abs(dp):.0f}% vs kemarin")
    parts.append(f"kas {'minus' if fin['cash_flow'] < 0 else 'positif'} {_fmt(fin['cash_flow'])}")
    directives = {
        'SEHAT': 'fokus pertahankan momentum dan kunjungan ulang',
        'HPP_TINGGI': 'fokus turunkan HPP dan tinjau margin',
        'HPP_KRONIS': 'fokus perbaikan margin struktural',
        'STOK_BERLEBIH': 'fokus stop belanja berlebih',
        'TRAFFIC_PROBLEM': 'fokus tarik kunjungan',
        'COST_PROBLEM': 'fokus audit beban operasional',
        'MARGINAL': 'fokus perbaiki campuran menu',
        'KRITIS': 'fokus penyelamatan kas',
    }
    parts.append(directives.get(situation, 'fokus perbaikan'))
    return ', '.join(parts) + '.'


def _build_math_facts(fin, max_menus=3):
    lines = ["REKOMENDASI HARGA AMAN (Maks +5%):"]
    top_menus = sorted(
        [x for x in fin['per_menu'] if x['margin'] is not None and x['qty'] > 0],
        key=lambda x: x['margin'], reverse=True
    )[:max_menus]
    for p in top_menus:
        up = p.get('unit_price') or 0
        if up > 0:
            safe = int(round(up * 1.05))
            lines.append(f"- {p['product']}: {_fmt(up)} -> aman {_fmt(safe)}")
    if fin.get('price_anomalies'):
        lines.append("ANOMALI HARGA:")
        for a in fin['price_anomalies'][:2]:
            lines.append(f"- {a['product']}: kasir {_fmt(a['avg_price'])}, acuan {_fmt(a['master_price'])}")
    return '\n'.join(lines)


def _fallback_actions(fin, situation, bep_info):
    if situation == 'SEHAT':
        return [
            {'aksi': 'Tinjau harga jual 2-3 menu margin tertinggi (maks 5% per minggu), pantau volume.', 'alasan': 'Naikkan margin absolut tanpa ganggu volume.'},
            {'aksi': 'Rancang program kunjungan ulang (kartu loyalitas) minggu ini.', 'alasan': 'Perluas basis pelanggan biaya rendah.'},
            {'aksi': 'Buat paket menu sering dipesan bersamaan dengan harga khusus.', 'alasan': 'Tingkatkan nilai transaksi per pelanggan.'},
            {'aksi': 'Jadwalkan promosi ringan di jam sepi.', 'alasan': 'Tingkatkan kunjungan di jam sepi.'},
        ]
    if situation == 'HPP_KRONIS':
        return [
            {'aksi': 'Tinjau harga/porsi 3 menu margin terendah; pertimbangkan kenaikan 5-10%.', 'alasan': 'HPP kronis; perbaikan menu-level paling berdampak.'},
            {'aksi': 'Negosiasi pemasok bahan utama (ayam, beras, minyak), target diskon 5-10%.', 'alasan': 'Turunkan HPP dasar.'},
            {'aksi': 'Dorong penjualan menu margin di atas 40% via rekomendasi kasir.', 'alasan': 'Naikkan margin rata-rata.'},
            {'aksi': 'Reformulasi atau ganti menu dengan margin di bawah 15%.', 'alasan': 'Kurangi drag margin.'},
        ]
    if situation == 'STOK_BERLEBIH':
        deficit = fin['expenses'] - fin['revenue']
        daily = fin['bahan_purchases']
        if daily > 0 and deficit > 0:
            target_amount = max(0, daily - deficit - 50000)
            target_pct = min(90, max(50, int(target_amount / daily * 100)))
        else:
            target_pct = 70
            target_amount = int(daily * target_pct / 100) if daily else 0
        return [
            {'aksi': f'Batasi belanja bahan besok jadi {_fmt(target_amount)} ({target_pct}% hari ini).', 'alasan': f'Belanja {_fmt(daily)} menyumbang minus {_fmt(deficit)}.'},
            {'aksi': 'Audit stok fisik protein dan sayur; catat sisa.', 'alasan': 'Cegah pemborosan dan dobel beli.'},
            {'aksi': 'Olah bahan berlebih jadi paket menu hemat hari ini saja.', 'alasan': 'Konversi stok ke pendapatan.'},
            {'aksi': 'Tunda pembelian bahan kering non-urgent.', 'alasan': 'Jaga kas positif.'},
        ]
    if situation == 'TRAFFIC_PROBLEM':
        return [
            {'aksi': 'Aktifkan pelanggan lama via pesan singkat dengan menu andalan.', 'alasan': 'Tarik kunjungan tanpa iklan besar.'},
            {'aksi': 'Buka jam promo 2 jam di jam sepi dengan paket minuman+makanan.', 'alasan': 'Tingkatkan kunjungan di jam sepi.'},
            {'aksi': 'Sebar konten promosi ke media sosial dan grup pelanggan.', 'alasan': 'Perluas jangkauan.'},
            {'aksi': 'Buat paket berbagi 2-4 orang.', 'alasan': 'Dorong kunjungan berkelompok.'},
        ]
    if situation == 'HPP_TINGGI':
        return [
            {'aksi': 'Negosiasi harga bahan pokok ke pemasok.', 'alasan': 'Turunkan HPP tanpa ubah porsi.'},
            {'aksi': 'Standarisasi porsi bahan per menu; timbang 3 menu terlaris.', 'alasan': 'Cegah pemborosan.'},
            {'aksi': 'Perbarui harga jual menu rasio HPP tinggi maksimal 5%.', 'alasan': 'Sesuaikan harga dengan biaya.'},
            {'aksi': 'Review pemborosan dapur harian.', 'alasan': 'Kurangi biaya bahan terbuang.'},
        ]
    if situation == 'COST_PROBLEM':
        return [
            {'aksi': 'Audit pemakaian listrik harian.', 'alasan': 'Turunkan beban non-bahan.'},
            {'aksi': 'Hitung ulang pemakaian gas per hari.', 'alasan': 'Efisiensi operasional.'},
            {'aksi': 'Bandingkan harga kemasan dengan 2 pemasok.', 'alasan': 'Cari penghematan.'},
            {'aksi': 'Batasi pembelian perlengkapan non-urgent.', 'alasan': 'Jaga kas sehat.'},
        ]
    if situation == 'MARGINAL':
        return [
            {'aksi': 'Dorong menu margin tertinggi via rekomendasi kasir.', 'alasan': 'Perbaiki campuran menu.'},
            {'aksi': 'Kurangi promosi menu margin rendah.', 'alasan': 'Naikkan margin rata-rata.'},
            {'aksi': 'Tawarkan paket makanan + minuman margin tinggi.', 'alasan': 'Tingkatkan margin total.'},
            {'aksi': 'Evaluasi menu jarang terjual.', 'alasan': 'Sederhanakan menu.'},
        ]
    return [
        {'aksi': 'Hentikan pembelian bahan non-urgent 48 jam.', 'alasan': 'Selamatkan kas jangka pendek.'},
        {'aksi': 'Fokus jual menu margin tinggi dengan bahan tersedia.', 'alasan': 'Maksimalkan pendapatan.'},
        {'aksi': 'Negosiasi tempo pembayaran pemasok.', 'alasan': 'Tunda pengeluaran.'},
        {'aksi': 'Promosi cepat ke pelanggan lama.', 'alasan': 'Datangkan pendapatan tunai.'},
    ]


def _fallback_strategi(fin, situation, bep_info):
    if situation == 'SEHAT':
        return "Rencana 3-7 hari: (1) tinjau harga menu margin tertinggi 5% bertahap, (2) aktifkan program kunjungan ulang, (3) uji paket hemat di jam sepi. Pantau volume setiap perubahan."
    if situation == 'HPP_KRONIS':
        return "Rencana 3-7 hari: (1) tinjau harga/porsi 3 menu margin terendah, (2) negosiasi pemasok bahan utama, (3) dorong penjualan menu margin tinggi. Target: HPP di bawah 58% dalam 2 minggu."
    if situation == 'STOK_BERLEBIH':
        return "Rencana 3-7 hari: (1) batasi belanja bahan sesuai target, (2) audit stok fisik, (3) olah bahan berlebih, (4) tunda pembelian non-urgent. Target: kas positif 3 hari."
    if situation == 'TRAFFIC_PROBLEM':
        return "Rencana 3-7 hari: (1) aktifkan pelanggan lama, (2) buka jam promo jam sepi, (3) sebar konten promosi mingguan. Target: kunjungan naik 20%."
    if situation == 'HPP_TINGGI':
        return "Rencana 3-7 hari: (1) negosiasi bahan, (2) standarisasi porsi, (3) perbarui harga bertahap. Target: HPP turun di bawah 55%."
    if situation == 'COST_PROBLEM':
        return "Rencana 3-7 hari: (1) audit listrik/gas/kemasan, (2) bandingkan pemasok, (3) tunda pembelian non-urgent. Target: beban turun 10%."
    if situation == 'MARGINAL':
        return "Rencana 3-7 hari: (1) dorong menu margin tinggi, (2) kurangi promosi margin rendah, (3) evaluasi menu jarang terjual. Target: margin naik 5%."
    return "Rencana 3-7 hari: (1) stop pembelian non-urgent, (2) fokus menu margin tinggi, (3) negosiasi tempo bayar, (4) promosi cepat. Target: kas positif 48 jam."


def _fix_analysis(data, fin, situation, bep_info, yctx=None, anomalies=None):
    for key in ('headline', 'diagnosis', 'strategi'):
        if data.get(key):
            data[key] = _clean(data[key])
    if isinstance(data.get('instruksi'), list):
        for item in data['instruksi']:
            if item.get('aksi'):
                item['aksi'] = _clean(item['aksi'])
            if item.get('alasan'):
                item['alasan'] = _clean(item['alasan'])

    diag, strat = _strip_strategy_from_diagnosis(data.get('diagnosis') or '', data.get('strategi') or '')
    data['diagnosis'] = diag
    data['strategi'] = strat

    instruksi = data.get('instruksi') if isinstance(data.get('instruksi'), list) else []
    cleaned = []
    for it in instruksi:
        if not isinstance(it, dict):
            continue
        aksi = (it.get('aksi') or '').strip()
        alasan = (it.get('alasan') or '').strip()
        if not (aksi and alasan):
            continue
        if _has_any(aksi, NEW_MENU_KWS):
            continue
        safe, _ = _check_remove_action_safety(aksi, fin)
        if not safe:
            continue
        valid, reason = _validate_restriction_action(aksi, fin)
        if not valid and reason:
            alasan = (alasan + " " if alasan else "") + reason
        cleaned.append({'aksi': aksi, 'alasan': alasan})
    instruksi = cleaned

    harga_up = set()
    discount = set()
    for it in instruksi:
        menus = set(_menu_mentions(it['aksi'], fin['per_menu']))
        if _has_any(it['aksi'], HARGA_UP_KWS):
            harga_up |= menus
        if _has_any(it['aksi'], DISCOUNT_KWS):
            discount |= menus
    conflicts = harga_up & discount
    if conflicts:
        instruksi = [it for it in instruksi
                     if not (_has_any(it['aksi'], DISCOUNT_KWS)
                             and (set(_menu_mentions(it['aksi'], fin['per_menu'])) & conflicts))]

    if len(instruksi) < 4:
        for fb in _fallback_actions(fin, situation, bep_info):
            if len(instruksi) >= 4:
                break
            instruksi.append(fb)
    data['instruksi'] = instruksi[:4]

    for key in ('headline', 'diagnosis', 'strategi'):
        if data.get(key):
            data[key] = _sanitize_numeric_claims(data[key], fin, bep_info)
    for it in data['instruksi']:
        if it.get('alasan'):
            it['alasan'] = _sanitize_numeric_claims(it['alasan'], fin, bep_info)

    if _text_overlap_ratio(data.get('diagnosis'), data.get('strategi')) > 0.6:
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

    data['diagnosis'] = _ensure_diagnosis_mentions_signals(data.get('diagnosis'), anomalies, yctx)

    if situation == 'SEHAT':
        diag = data.get('diagnosis') or ''
        if any(w in diag.lower() for w in ['akar masalah', 'masalah utama']) or \
           not diag.lower().startswith(('operasional', 'kondisi', 'hari ini')):
            data['diagnosis'] = (
                f"Operasional hari ini sehat: laba operasional {_fmt(fin['operating_profit'])} "
                f"dan kas bersih {_fmt(fin['cash_flow'])}. {diag}"
            ).strip()
        h = data.get('headline') or ''
        if _has_any(h, PRUNE_KWS) and not _has_any(h, GROWTH_KWS):
            data['headline'] = _build_fallback_headline(fin, bep_info, yctx, situation)

    if situation == 'HPP_KRONIS':
        diag = data.get('diagnosis') or ''
        if 'hpp' not in diag.lower() and 'margin' not in diag.lower():
            data['diagnosis'] = (
                f"Operasional hari ini berjalan dengan HPP kronis di atas 60% "
                f"(rasio {bep_info.get('hpp_ratio', 0)*100:.0f}%). Margin per menu tipis. {diag}"
            ).strip()

    if not data.get('headline'):
        data['headline'] = _build_fallback_headline(fin, bep_info, yctx, situation)

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['cash_flow'] < 0 else "Untung"

    if not data.get('strategi'):
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

    return data


def _get_available_models(client):
    try:
        models_response = client.models.list()
        available_ids = [m.id for m in models_response.data]
    except Exception:
        return ["llama-3.1-8b-instant"]
    preferred = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
    ]
    valid = [m for m in preferred if m in available_ids]
    if not valid:
        for mid in available_ids:
            if not any(x in mid.lower() for x in ['whisper', 'tts', 'guard', 'playai', 'orpheus', 'qwen', 'gpt-oss']):
                valid.append(mid)
    return valid or ["llama-3.1-8b-instant"]


def _extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _build_user_prompt(fin, classification, tanggal_panjang,
                       bep_info, situation_label, hist_text, learning_ctx,
                       yctx, anomalies, shrink=1.0):
    b = lambda x: max(50, int(x * shrink))

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
            f"- {p['product']}: qty={p['qty']}, harga={_fmt(up)}, acuan={mp_str}, "
            f"pendapatan={_fmt(p['revenue'])}, laba={_fmt(p['margin'])}"
        )

    cls_lines = []
    for cat, items in list(classification.items())[:3]:
        if not items:
            continue
        names = ', '.join(f"{i['product']} (qty {i['qty']})" for i in items[:3])
        cls_lines.append(f"- {cat}: {names}")

    exp_lines = []
    for cat, data in list(fin['expense_breakdown'].items())[:3]:
        items_str = ', '.join(f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:3])
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_cash = "NEGATIF" if fin['cash_flow'] < 0 else "POSITIF"
    hpp_ratio_pct = bep_info['hpp_ratio'] * 100
    attach_pct = fin.get('attach_rate', 0) * 100

    math_facts = _trim(_build_math_facts(fin, max_menus=3), b(700))
    hist_short = _trim(hist_text, b(700))
    learn_short = _trim(format_learning_for_prompt(learning_ctx), b(700))

    yctx_text = _trim((yctx or {}).get('text') or '(belum ada laporan sebelumnya)', b(1400))

    verification_lines = []
    if yctx and yctx.get('verifications'):
        status_map = {'likely_executed': 'TEREKSKUSI', 'partial': 'SEBAGIAN',
                      'likely_not_executed': 'BELUM DIEKSEKUSI', 'unknown': 'TIDAK DIVERIFIKASI'}
        for v in yctx['verifications'][:3]:
            verification_lines.append(
                f"- \"{v['action'][:80]}\" -> {status_map.get(v['status'], v['status'])}. {v['detail'][:80]}"
            )
    verification_text = '\n'.join(verification_lines) if verification_lines else '(tidak ada)'
    verification_text = _trim(verification_text, b(700))

    signal_lines = []
    for a in (anomalies or [])[:4]:
        signal_lines.append('- ' + _trim(a, b(180)))
    signals_text = '\n'.join(signal_lines) if signal_lines else '- tidak ada'

    prompt = f"""Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN (FAKTA):
- Pendapatan: {_fmt(fin['revenue'])}
- Pengeluaran: {_fmt(fin['expenses'])}
- Kas Bersih: {_fmt(fin['cash_flow'])} ({status_cash})
- Rasio HPP: {hpp_ratio_pct:.1f}%
- Rasio minuman/makanan: {attach_pct:.1f}%

MATEMATIKA (SUDAH DIHITUNG, JANGAN ULANG):
{math_facts}

PERBANDINGAN LAPORAN SEBELUMNYA:
{yctx_text}

VERIFIKASI AKSI KEMARIN:
{verification_text}

SINYAL OPERASIONAL (WAJIB DIBAHAS DI DIAGNOSIS):
{signals_text}

DETAIL MENU (Top 10 Laba):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada'}

KLASIFIKASI MENU (Top 3 per Kategori):
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

PENGELUARAN (Top 3 per Kategori):
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

HISTORIS:
{hist_short}

POLA PEMBELAJARAN:
{learn_short}

SITUASI: {situation_label}

TUGAS: Hasilkan JSON valid (Bahasa Indonesia, tanpa kata Inggris):
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat: kas bersih + arahan utama. WAJIB sebut 1 delta jika ada perbandingan.",
  "diagnosis": "2-3 kalimat FAKTA hari ini (JANGAN masukkan rencana). Wajib bahas sinyal jika ada. Wajib sebut status verifikasi aksi jika ada.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik", "alasan": "Alasan singkat"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf RENCANA 3-7 HARI (berbeda dari diagnosis, bukan ulang diagnosis)."
}}

ATURAN AKSI:
- Sehat -> pertumbuhan, DILARANG memangkas menu/biaya.
- HPP Kronis -> perbaiki margin struktural (harga/porsi/pemasok).
- Stok Berlebih -> batasi belanja dengan angka spesifik.
- Kunjungan Rendah -> promosi, jam promo, pelanggan lama.
- HPP Tinggi (akut) -> negosiasi, porsi, harga.
- Biaya Operasional Tinggi -> audit listrik/gas/kemasan.
- Margin Tipis -> dorong margin tinggi.
- Kritis -> penyelamatan kas 24 jam.

ATURAN FATAL:
- WAJIB 4 aksi.
- DILARANG "tambah menu baru X" jika sudah ada.
- DILARANG hapus/kurangi menu yang menyumbang >5% pendapatan.
- DILARANG sebut angka yang tidak ada di data.
- DILARANG naik harga + diskon pada menu yang sama.
- Strategi HARUS berbeda dari Diagnosis.
- Jika X "melampaui" Y maka X>Y. Jika X "di bawah" Y maka X<Y.
- Jika aksi kemarin BELUM DIEKSEKUSI, jangan klaim berhasil.
"""
    return prompt


def generate_analysis(fin, classification, tanggal_panjang, tanggal_iso,
                      bep_info, situation, situation_label, hist_text, learning_ctx,
                      yctx=None, anomalies=None):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    available_models = _get_available_models(client)

    data = None
    last_error = None

    for shrink in (1.0, 0.65, 0.4):
        if data:
            break
        prompt = _build_user_prompt(
            fin, classification, tanggal_panjang, bep_info, situation_label,
            hist_text, learning_ctx, yctx, anomalies, shrink=shrink,
        )
        est = _est_tokens(prompt) + _est_tokens(SYSTEM_PROMPT) + MAX_COMPLETION_TOKENS
        print(f"[ai] shrink={shrink}, est_tokens={est}", flush=True)

        context_error = False
        for model_name in available_models:
            if data:
                break
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=MAX_COMPLETION_TOKENS,
                    )
                    raw = resp.choices[0].message.content
                    parsed = _extract_json(raw)
                    if parsed:
                        data = parsed
                        print(f"[ai] success model={model_name} shrink={shrink}", flush=True)
                        break
                    last_error = Exception("Gagal parse JSON")
                    break
                except RateLimitError as e:
                    if attempt == 0:
                        time.sleep(4)
                        continue
                    last_error = e
                    break
                except Exception as e:
                    last_error = e
                    if _is_context_error(e):
                        context_error = True
                    print(f"[ai] fail model={model_name} shrink={shrink}: {str(e)[:120]}", flush=True)
                    break
            if context_error:
                break

        if data:
            break
        if not context_error and not isinstance(last_error, RateLimitError):
            break

    if data is None:
        raise Exception(f"Semua model AI gagal. Error terakhir: {last_error}")

    data = _fix_analysis(data, fin, situation, bep_info, yctx=yctx, anomalies=anomalies)
    return data