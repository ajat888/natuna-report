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
    "- Jika situasi = HPP Terlalu Tinggi (akut, 1 hari): Fokus HPP melebihi batas hari ini. "
    "- Jika situasi = HPP Tinggi Berkelanjutan (kronis): Fokus perbaikan margin struktural 3-7 hari (harga, porsi, negosiasi pemasok). Boleh sebut kas positif tetapi HPP kronis adalah masalah utama. DILARANG memakai kata 'sehat' sebagai status utama. "
    "- Jika situasi = Biaya Operasional Tinggi: Fokus beban non-bahan membengkak. "
    "- Jika situasi = Margin Tipis: Fokus campuran menu bikin laba tipis. "
    "- Jika situasi = Kritis: Fokus penyelamatan kas darurat. "
    "ATURAN AKSI-NYAMBUNG-DIAGNOSIS (WAJIB): "
    "Setiap aksi HARUS menjawab akar masalah di diagnosis. Dilarang berlawanan arah. "
    "DILARANG menyarankan KENAIKAN HARGA dan DISKON pada menu yang SAMA dalam satu laporan. "
    "ATURAN PERBANDINGAN (FATAL): "
    "Jika ada PERBANDINGAN DENGAN LAPORAN SEBELUMNYA, WAJIB sebut minimal 1 delta angka di headline atau diagnosis. "
    "Jangan pernah membandingkan kategori belanja dengan total pengeluaran (apel vs jeruk). Hanya setara dengan setara. "
    "ATURAN VERIFIKASI AKSI KEMARIN (FATAL): "
    "Jika ada STATUS AKSI KEMARIN dengan status BELUM DIEKSEKUSI atau TIDAK DIVERIFIKASI: JANGAN klaim aksi tersebut berhasil atau berkontribusi. "
    "Cukup sebut bahwa target belum tercapai dan rekomendasi tetap berlaku. "
    "Jika status TEREKSKUSI: boleh sebut sebagai konteks tapi jangan klaim kausalitas berlebihan. "
    "HARUS MENGEMBALIKAN OUTPUT BERUPA FORMAT JSON YANG VALID SEPERTI YANG DIMINTA."
)

NEW_MENU_KWS = ('tambahkan menu', 'tambah menu', 'menu baru', 'perkenalkan menu', 'memperkenalkan menu')
HARGA_UP_KWS = ('naikkan harga', 'tingkatkan harga', 'menaikkan harga', 'meningkatkan harga', 'sesuaikan harga jual', 'sesuaikan harga')
DISCOUNT_KWS = ('diskon', 'potongan harga', 'harga khusus')
REMOVE_MENU_KWS = ('hapus menu', 'kurangi menu', 'hilangkan menu', 'pangkas menu', 'kurangi item', 'hapus item', 'menghapus menu', 'mengurangi menu')
PRUNE_KWS = ('pangkas', 'hapus', 'kurangi menu', 'evaluasi menu', 'menata ulang', 'kurangi biaya', 'potong menu')
GROWTH_KWS = ('tingkatkan', 'kembangkan', 'pertahankan', 'pertumbuhan', 'pelanggan', 'kunjungan', 'promosi', 'loyalitas', 'naikkan pendapatan', 'pelanggan lama')

STRATEGY_MARKERS_RE = re.compile(
    r'(\b(?:prioritas|rencana|langkah|fokus)\s+3[-\s]7\s+hari\b'
    r'|\bdalam\s+3[-\s]7\s+hari\b'
    r'|\btiga\s+sampai\s+tujuh\s+hari\b'
    r'|\bstrategi\s+3[-\s]7\s+hari\b)',
    re.IGNORECASE,
)


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


def _text_overlap_ratio(text1, text2, min_word_len=3):
    if not text1 or not text2:
        return 0
    def norm_words(t):
        words = _norm_text(t).split()
        return set(w for w in words if len(w) >= min_word_len)
    w1 = norm_words(text1)
    w2 = norm_words(text2)
    if not w1 or not w2:
        return 0
    return len(w1 & w2) / len(w1 | w2)


def _strip_strategy_from_diagnosis(diagnosis, strategy):
    if not diagnosis:
        return diagnosis, strategy
    m = STRATEGY_MARKERS_RE.search(diagnosis)
    if not m:
        return diagnosis, strategy
    idx = m.start()
    if idx < 15:
        return diagnosis, strategy
    strategy_part = diagnosis[idx:]
    clean_diag = diagnosis[:idx].rstrip(' .,;:')
    if not clean_diag:
        return diagnosis, strategy
    if not strategy or len(strategy) < 40:
        return clean_diag, strategy_part
    return clean_diag, strategy


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
        if p['qty'] <= threshold_qty and total_rev > 0:
            pct = p['revenue'] / total_rev * 100
            if pct >= 5.0 and p['revenue'] >= 50000:
                high_value.append(f"{p['product']} ({_fmt(p['revenue'])}, {pct:.0f}%)")
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
    if not target_cat:
        if 'dapur' in action_text.lower() or 'bahan' in action_text.lower():
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
            f"di atas pendapatan {_fmt(fin['revenue'])} (defisit {_fmt(deficit)}). Perlu pengurangan lebih agresif atau tambahan pendapatan."
        )
    return True, None


def _ensure_diagnosis_mentions_signals(diagnosis, anomalies, yctx):
    if not diagnosis:
        return diagnosis
    if not anomalies and not (yctx and yctx.get('movers')):
        return diagnosis
    d_norm = _norm_text(diagnosis)
    missing = []
    for a in (anomalies or []):
        keyword_match = re.search(r'(Belanja \w+|Volume item|Rasio minuman|Harga \')', a)
        if keyword_match:
            if _norm_text(keyword_match.group(1)) not in d_norm:
                missing.append(a[:80] + ('...' if len(a) > 80 else ''))
    if yctx and yctx.get('has_previous'):
        for m in (yctx.get('movers') or [])[:2]:
            if _norm_text(m['menu']) not in d_norm:
                missing.append(f"{m['menu']} {'naik' if m['diff'] > 0 else 'turun'} dari {m['prev']} ke {m['today']}")
    if missing:
        return diagnosis.rstrip(' .,') + ". Sinyal yang perlu dicek: " + "; ".join(missing[:2]) + "."
    return diagnosis


def _build_fallback_headline(fin, bep_info, yctx, situation):
    parts = []
    if yctx and yctx.get('has_previous'):
        prev_rev = yctx.get('prev_revenue', 0)
        if prev_rev > 0:
            delta_pct = (fin['revenue'] - prev_rev) / prev_rev * 100
            arrow = "naik" if delta_pct >= 0 else "turun"
            parts.append(f"Pendapatan {arrow} {abs(delta_pct):.0f}% vs kemarin")
    if fin['cash_flow'] < 0:
        parts.append(f"kas minus {_fmt(fin['cash_flow'])}")
    else:
        parts.append(f"kas positif {_fmt(fin['cash_flow'])}")
    directives = {
        'SEHAT': 'fokus pertahankan momentum dan program kunjungan ulang',
        'HPP_TINGGI': 'fokus turunkan HPP dan tinjau margin menu',
        'HPP_KRONIS': 'fokus perbaikan margin struktural 3-7 hari',
        'STOK_BERLEBIH': 'fokus stop belanja bahan berlebih',
        'TRAFFIC_PROBLEM': 'fokus tarik kunjungan lewat promosi dan paket',
        'COST_PROBLEM': 'fokus audit beban operasional',
        'MARGINAL': 'fokus perbaiki campuran menu',
        'KRITIS': 'fokus penyelamatan kas darurat',
    }
    parts.append(directives.get(situation, 'fokus perbaikan operasional'))
    return ', '.join(parts) + '.'


def _build_action_verification_text(yctx):
    if not yctx or not yctx.get('verifications'):
        return ""
    status_map = {
        'likely_executed': 'TEREKSKUSI',
        'partial': 'SEBAGIAN',
        'likely_not_executed': 'BELUM DIEKSEKUSI',
        'unknown': 'TIDAK DIVERIFIKASI',
    }
    lines = ["VERIFIKASI AKSI KEMARIN:"]
    for v in yctx['verifications']:
        lines.append(f"- Aksi: \"{v['action']}\"")
        lines.append(f"  Status: {status_map.get(v['status'], v['status'])}. {v['detail']}")
    return '\n'.join(lines)


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
    if situation == 'HPP_KRONIS':
        worst = sorted(
            [p for p in fin['per_menu'] if p.get('margin_pct') is not None and p['qty'] > 0],
            key=lambda p: p['margin_pct']
        )[:3]
        worst_names = ', '.join(f"{p['product']} ({p['margin_pct']:.0f}%)" for p in worst) if worst else "menu margin terendah"
        return [
            {'aksi': f'Tinjau harga/porsi menu margin terendah: {worst_names}. Pertimbangkan kenaikan 5-10% atau reformulasi porsi.', 'alasan': 'HPP kronis di atas 60%; perbaikan menu-level paling berdampak.'},
            {'aksi': 'Negosiasi harga pemasok untuk 3 bahan paling mahal (ayam, beras, minyak) dengan target diskon 5-10%.', 'alasan': 'Menurunkan HPP dasar tanpa mengubah volume.'},
            {'aksi': 'Dorong penjualan menu margin di atas 40% (minuman, snack) lewat rekomendasi kasir dan paket.', 'alasan': 'Menaikkan margin rata-rata keseluruhan.'},
            {'aksi': 'Evaluasi menu dengan margin di bawah 15% atau qty rendah; reformulasi atau penggantian.', 'alasan': 'Mengurangi drag margin.'},
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
            {'aksi': f'Batasi belanja bahan besok menjadi {_fmt(target_amount)} ({target_pct}% dari hari ini) agar kas tidak minus.', 'alasan': f'Belanja bahan {_fmt(daily)} hari ini menyumbang kas minus {_fmt(deficit)}. Reduksi ke {target_pct}% menutup defisit.'},
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
    if situation == 'SEHAT':
        return "Rencana 3-7 hari: (1) tinjau harga menu margin tertinggi 5% bertahap sambil pantau volume harian, (2) aktifkan program kunjungan ulang pelanggan, (3) uji paket hemat di jam sepi. Pantau volume setiap perubahan sebelum menaikkan harga berikutnya."
    if situation == 'HPP_KRONIS':
        return "Rencana 3-7 hari: (1) tinjau ulang harga/porsi 3 menu margin terendah, (2) negosiasi pemasok bahan utama (ayam, beras, minyak), (3) dorong penjualan menu margin tinggi lewat kasir dan paket. Target: HPP turun di bawah 58% dalam 2 minggu."
    if situation == 'STOK_BERLEBIH':
        return "Rencana 3-7 hari: (1) batasi belanja bahan sesuai target, (2) audit stok fisik, (3) olah bahan berlebih menjadi paket hemat, (4) tunda pembelian bahan kering non-urgent. Target: kas positif dalam 3 hari."
    if situation == 'TRAFFIC_PROBLEM':
        return "Rencana 3-7 hari: (1) aktifkan pelanggan lama, (2) buka jam promo di jam sepi, (3) sebar konten promosi mingguan. Target: kunjungan naik 20%."
    if situation == 'HPP_TINGGI':
        return "Rencana 3-7 hari: (1) negosiasi harga bahan, (2) standarisasi porsi, (3) perbarui harga jual menu bermasalah secara bertahap. Target: HPP turun di bawah 55%."
    if situation == 'COST_PROBLEM':
        return "Rencana 3-7 hari: (1) audit listrik, gas, kemasan, (2) bandingkan pemasok, (3) tunda pembelian non-urgent. Target: beban turun 10%."
    if situation == 'MARGINAL':
        return "Rencana 3-7 hari: (1) dorong menu margin tinggi, (2) kurangi promosi menu margin rendah, (3) evaluasi menu jarang terjual. Target: margin rata-rata naik 5%."
    return "Rencana 3-7 hari: (1) hentikan pembelian non-urgent, (2) fokus jual menu margin tinggi, (3) negosiasi tempo bayar pemasok, (4) promosi cepat ke pelanggan lama. Target: kas positif dalam 48 jam."


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

    diag = data.get('diagnosis') or ''
    strat = data.get('strategi') or ''
    diag, strat = _strip_strategy_from_diagnosis(diag, strat)
    data['diagnosis'] = diag
    data['strategi'] = strat

    instruksi = data.get('instruksi') if isinstance(data.get('instruksi'), list) else []
    cleaned_actions = []
    for it in instruksi:
        if not isinstance(it, dict):
            continue
        aksi = (it.get('aksi') or '').strip()
        alasan = (it.get('alasan') or '').strip()
        if not (aksi and alasan):
            continue
        if _has_any(aksi, NEW_MENU_KWS):
            continue
        safe, blocked = _check_remove_action_safety(aksi, fin)
        if not safe:
            continue
        valid, reason = _validate_restriction_action(aksi, fin)
        if not valid and reason:
            alasan = (alasan + " " if alasan else "") + reason
        cleaned_actions.append({'aksi': aksi, 'alasan': alasan})
    instruksi = cleaned_actions

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
        forbidden = ['akar masalah', 'masalah utama', 'masalah utama adalah']
        has_forbidden = any(w in diag.lower() for w in forbidden)
        if has_forbidden or not diag.lower().startswith(('operasional', 'kondisi', 'hari ini')):
            data['diagnosis'] = (
                f"Operasional hari ini sehat: laba operasional {_fmt(fin['operating_profit'])} "
                f"dan kas bersih {_fmt(fin['cash_flow'])}. {diag}"
            ).strip()
        h = data.get('headline') or ''
        if _has_any(h, PRUNE_KWS) and not _has_any(h, GROWTH_KWS):
            data['headline'] = (
                f"Operasional hari ini sehat: kas bersih {_fmt(fin['cash_flow'])} "
                f"({bep_info['gap_pct']:+.1f}% dari total uang keluar). "
                f"Fokus: pertahankan momentum lewat promosi menu primadona dan program kunjungan ulang pelanggan."
            )

    if situation == 'HPP_KRONIS':
        diag = data.get('diagnosis') or ''
        diag_lower = diag.lower()
        if 'hpp' not in diag_lower and 'margin' not in diag_lower:
            data['diagnosis'] = (
                f"Operasional hari ini berjalan dengan HPP kronis di atas 60% "
                f"(rasio HPP hari ini {bep_info.get('hpp_ratio', 0)*100:.0f}%). Margin per menu tipis. {diag}"
            ).strip()

    if not data.get('headline'):
        data['headline'] = _build_fallback_headline(fin, bep_info, yctx, situation)

    if not data.get('kondisi'):
        data['kondisi'] = "Rugi" if fin['cash_flow'] < 0 else "Untung"

    if not data.get('strategi'):
        data['strategi'] = _fallback_strategi(fin, situation, bep_info)

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
        return ["llama-3.1-8b-instant", "llama3-70b-8192"]

    preferred_order = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama-3.3-70b-versatile"
    ]

    valid_models = []
    for pm in preferred_order:
        if pm in available_ids:
            valid_models.append(pm)

    if not valid_models:
        for mid in available_ids:
            if not any(x in mid.lower() for x in ['whisper', 'tts', 'guard', 'playai', 'orpheus', 'qwen', 'gpt-oss']):
                valid_models.append(mid)

    if not valid_models:
        valid_models = ["llama-3.1-8b-instant"]

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
                      bep_info, situation, situation_label, hist_text, learning_ctx,
                      yctx=None, anomalies=None):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    menu_lines = []
    top_menus = sorted([x for x in fin['per_menu'] if x['margin'] is not None], key=lambda x: x['margin'], reverse=True)[:15]

    for p in top_menus:
        up = p.get('unit_price') or 0
        mp = p.get('master_price')
        mp_str = _fmt(mp) if mp else '-'
        menu_lines.append(
            f"- {p['product']}: qty={p['qty']}, harga_jual={_fmt(up)}, "
            f"acuan={mp_str}, total_pendapatan={_fmt(p['revenue'])}, laba={_fmt(p['margin'])} ({p['margin_pct']:.1f}%)"
        )

    cls_lines = []
    for cat, items in classification.items():
        if not items:
            continue
        names = ', '.join(f"{i['product']} (qty {i['qty']})" for i in items[:5])
        cls_lines.append(f"- {cat}: {names}")

    exp_lines = []
    for cat, data in fin['expense_breakdown'].items():
        items_str = ', '.join(f"{i['sub']}={_fmt(i['amount'])}" for i in data['items'][:3])
        exp_lines.append(f"- {cat} ({_fmt(data['total'])}): {items_str}")

    status_cash = "NEGATIF" if fin['cash_flow'] < 0 else "POSITIF"
    hpp_ratio_pct = bep_info['hpp_ratio'] * 100
    attach_rate_pct = fin.get('attach_rate', 0) * 100

    anomaly_lines = _build_anomaly_lines(fin)
    math_facts = _build_math_facts(fin)

    hist_text_short = _truncate(hist_text, 800)
    learning_text_short = _truncate(format_learning_for_prompt(learning_ctx), 800)

    yctx_text = (yctx or {}).get('text') or '(belum ada laporan sebelumnya)'
    verification_text = _build_action_verification_text(yctx)
    anomaly_signals = anomalies or []

    prompt = f"""
Laporan harian Natuna Café & Resto untuk {tanggal_panjang}.

RINGKASAN KEUANGAN (FAKTA):
- Pendapatan: {_fmt(fin['revenue'])}
- Pengeluaran: {_fmt(fin['expenses'])}
- KAS BERSIH: {_fmt(fin['cash_flow'])} ({status_cash})
- Rasio HPP: {hpp_ratio_pct:.1f}%
- Rasio minuman/makanan: {attach_rate_pct:.1f}%

MATEMATIKA YANG SUDAH DIHITUNG OLEH SISTEM:
{math_facts}
(CATATAN: JANGAN menghitung ulang. Jika menyarankan aksi, gunakan angka di atas.)

PERBANDINGAN DENGAN LAPORAN SEBELUMNYA:
{yctx_text}

{verification_text if verification_text else "(belum ada aksi yang perlu diverifikasi)"}

SINYAL OPERASIONAL (WAJIB DIBAHAS DI DIAGNOSIS JIKA ADA):
{chr(10).join('- ' + a for a in anomaly_signals) if anomaly_signals else '- tidak ada'}

ANOMALI TEKNIS (cek input kasir):
{chr(10).join(anomaly_lines) if anomaly_lines else '- tidak ada'}

TOTAL UANG KELUAR HARI INI:
- Total: {_fmt(bep_info['bep'])}
- Kas Bersih: {_fmt(bep_info['gap'])} ({bep_info['gap_pct']:+.1f}%)

SITUASI: {situation_label}
RASIO HPP: {hpp_ratio_pct:.1f}%

DETAIL MENU (Top 15):
{chr(10).join(menu_lines) if menu_lines else '- tidak ada data'}

KLASIFIKASI MENU:
{chr(10).join(cls_lines) if cls_lines else '- tidak ada'}

RINCIAN PENGELUARAN (Top 3 per Kategori):
{chr(10).join(exp_lines) if exp_lines else '- tidak ada'}

HISTORIS:
{hist_text_short}

POLA PEMBELAJARAN:
{learning_text_short}

TUGAS:
Hasilkan JSON PERSIS (Bahasa Indonesia, tanpa kata Inggris):
{{
  "kondisi": "Rugi" atau "Untung",
  "headline": "1 kalimat tajam menyebut kas bersih dan arahan utama. WAJIB sebut 1 delta jika ada PERBANDINGAN.",
  "diagnosis": "2-3 kalimat FAKTA tentang kondisi hari ini. JANGAN masukkan rencana ke depan. Jika ada SINYAL OPERASIONAL, wajib bahas. Jika ada VERIFIKASI AKSI KEMARIN, sebutkan status singkat.",
  "instruksi": [
    {{"aksi": "Tindakan taktis spesifik hari ini/besok", "alasan": "Kenapa"}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}},
    {{"aksi": "...", "alasan": "..."}}
  ],
  "strategi": "1 paragraf RENCANA 3-7 HARI KE DEPAN (bukan ulang diagnosis). Fokus: langkah konkret, target terukur, evaluasi."
}}

PANDUAN AKSI:
- Sehat → pertumbuhan (margin absolut, promosi, pelanggan, kunjungan ulang). DILARANG memangkas menu atau biaya.
- HPP Tinggi Berkelanjutan (kronis) → perbaikan struktur margin: tinjau harga/porsi menu margin terendah, negosiasi pemasok, dorong menu margin tinggi.
- Belanja Stok Berlebih → batasi belanja dengan angka spesifik, audit stok, olah sisa.
- Kunjungan Rendah → promosi, jam promo, aktifkan pelanggan lama.
- HPP Terlalu Tinggi (akut 1 hari) → negosiasi, reformulasi porsi, sesuaikan harga.
- Biaya Operasional Tinggi → audit listrik, gas, kemasan.
- Margin Tipis → dorong menu margin tinggi, kurangi margin rendah.
- Kritis → penyelamatan kas 24 jam.

ATURAN TAMBAHAN (FATAL):
- WAJIB 4 aksi.
- DILARANG sarankan "tambahkan menu baru X" untuk menu yang SUDAH ADA.
- DILARANG sarankan hapus/kurangi menu yang menyumbang lebih dari 5% pendapatan hari ini.
- DILARANG sebut harga/angka yang tidak ada di fakta matematika atas.
- DILARANG naikkan harga DAN diskon pada menu yang sama.
- Strategi HARUS BERBEDA dari Diagnosis (jangan duplikat).
- Saat menulis "X melampaui Y", pastikan X > Y. Saat menulis "X di bawah Y", pastikan X < Y.
- Jika verifikasi aksi kemarin = BELUM DIEKSEKUSI: JANGAN klaim aksi itu berhasil atau berkontribusi.
"""

    available_models = _get_available_models(client)
    data = None
    last_error = None

    for model_name in available_models:
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                )
                raw_text = resp.choices[0].message.content
                parsed_data = _extract_json(raw_text)
                if parsed_data:
                    data = parsed_data
                    break
                last_error = Exception("Gagal parse JSON")
                break
            except RateLimitError as e:
                if attempt == 0:
                    time.sleep(5)
                    continue
                last_error = e
                break
            except Exception as e:
                last_error = e
                break
        if data:
            break

    if data is None:
        raise Exception(f"Semua model AI gagal. Error terakhir: {last_error}")

    data = _fix_analysis(data, fin, situation, bep_info, yctx=yctx, anomalies=anomalies)

    return data