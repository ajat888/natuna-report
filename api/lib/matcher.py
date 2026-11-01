import re
from difflib import SequenceMatcher

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def _price_from_sales(sales_item):
    if sales_item['qty'] > 0:
        return sales_item['revenue'] / sales_item['qty']
    return 0

def match_menu(sales_item, menu_list):
    """
    Cari menu HPP terbaik untuk 1 sales_item.
    Return menu dict atau None.
    """
    n = _norm(sales_item['product'])
    avg_price = _price_from_sales(sales_item)

    # 1) Exact name match
    for m in menu_list:
        if _norm(m['full_name']) == n or _norm(m['name']) == n:
            return m

    # 2) Contains match
    for m in menu_list:
        mn = _norm(m['full_name'])
        if mn and (mn in n or n in mn):
            return m

    # 3) Fuzzy + price score
    best, best_score = None, 0.0
    for m in menu_list:
        name_score = SequenceMatcher(None, _norm(m['full_name']), n).ratio()
        if m['price'] > 0 and avg_price > 0:
            price_score = max(0, 1 - abs(m['price'] - avg_price) / m['price'])
        else:
            price_score = 0
        score = 0.6 * name_score + 0.4 * price_score
        if score > best_score:
            best, best_score = m, score

    return best if best_score >= 0.6 else None