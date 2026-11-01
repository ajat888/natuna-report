import re
from difflib import SequenceMatcher
from .menu_data import ALIAS_MAP

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def _price_from_sales(sales_item):
    if sales_item['qty'] > 0:
        return sales_item['revenue'] / sales_item['qty']
    return 0

def match_menu(sales_item, menu_list):
    # 1. Cek Kamus Alias (Paling Akurat)
    pos_name_lower = sales_item['product'].strip().lower()
    if pos_name_lower in ALIAS_MAP:
        target_name = ALIAS_MAP[pos_name_lower]
        for m in menu_list:
            if m['full_name'].lower() == target_name.lower():
                return m

    n = _norm(sales_item['product'])
    avg_price = _price_from_sales(sales_item)

    # 2) Exact name match
    for m in menu_list:
        if _norm(m['full_name']) == n or _norm(m['name']) == n:
            return m

    # 3) Contains match
    for m in menu_list:
        mn = _norm(m['full_name'])
        if mn and (mn in n or n in mn):
            return m

    # 4) Fuzzy + price score
    best, best_score = None, 0.0
    for m in menu_list:
        name_score = SequenceMatcher(None, _norm(m['full_name']), n).ratio()
        
        # Cek harga, beri toleransi maksimal 5000
        if m['price'] > 0 and avg_price > 0:
            price_diff = abs(m['price'] - avg_price)
            price_score = max(0, 1 - (price_diff / 5000))
        else:
            price_score = 0
            
        # Bobot: 70% Nama, 30% Harga (agar nama lebih diutamakan)
        score = 0.7 * name_score + 0.3 * price_score
        
        if score > best_score:
            best, best_score = m, score

    # Naikkan threshold ke 0.65 agar lebih ketat
    return best if best_score >= 0.65 else None