import re
from bs4 import BeautifulSoup

def parse_sales(file_obj, target_date_iso):
    """
    file_obj: file upload (HTML periodik)
    target_date_iso: '2026-10-21'
    Return list [{product, qty, revenue}] untuk tanggal itu saja.
    """
    html = file_obj.read().decode('utf-8', errors='ignore') if hasattr(file_obj, 'read') else file_obj
    soup = BeautifulSoup(html, 'lxml')

    # Target day-of-month
    target_day = int(target_date_iso.split('-')[2])

    table = soup.find('table', id='tb-report')
    if table is None:
        raise ValueError("Tabel sales tidak ditemukan (id=tb-report)")

    # Parse header: tiap td colspan=2 adalah satu hari ("1, Kamis", "21, Rabu", ...)
    first_tr = table.thead.find_all('tr')[0]
    header_cells = first_tr.find_all('td')[1:]  # skip 'Produk'
    days = []
    for td in header_cells:
        text = td.get_text(strip=True)
        m = re.match(r'(\d+)', text)
        if m:
            days.append(int(m.group(1)))

    if target_day not in days:
        raise ValueError(f"Tanggal {target_day} tidak ada di file sales (hari tersedia: {days})")
    day_idx = days.index(target_day)

    results = []
    for tr in table.tbody.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 2:
            continue
        # Strip badge "Non Aktif" / "dihapus"
        for small in tds[0].find_all('small'):
            small.decompose()
        product = tds[0].get_text(strip=True)
        if not product or product.lower() == 'grand total':
            continue

        # Kolom: [0]=produk, lalu pasangan (qty, revenue) sebanyak len(days)
        qty_col   = 1 + day_idx * 2
        total_col = qty_col + 1
        if total_col >= len(tds):
            continue

        try:
            qty     = int(tds[qty_col].get_text(strip=True).replace(',', '') or 0)
            revenue = int(tds[total_col].get_text(strip=True).replace(',', '') or 0)
        except ValueError:
            continue

        if qty > 0 or revenue > 0:
            results.append({'product': product, 'qty': qty, 'revenue': revenue})

    return results