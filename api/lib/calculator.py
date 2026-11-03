from .matcher import match_menu

FOOD_CATEGORIES = {'Makanan', 'Berkuah/Mie', 'Ikan', 'Snack', 'Paket'}

OVERHEAD_KEYWORDS = [
    'listrik', 'gas', 'galon', 'nota', 'cup', 'tisu', 'sedotan',
    'plastik', 'karet', 'spon', 'sikat', 'penyapu', 'pel lantai',
    'vixal', 'sabun', 'pulpen', 'kertas', 'kresek', 'wow',
    'pe 15', 'kemasan', 'packaging', 'box', 'bensin', 'transport',
    'parkir', 'gaji', 'upah', 'sewa', 'internet', 'wifi',
    'laundry', 'seragam',
]


def _is_overhead_item(sub_text):
    s = (sub_text or '').lower()
    return any(kw in s for kw in OVERHEAD_KEYWORDS)


def split_expenses(expense_breakdown):
    bahan_total = 0
    overhead_total = 0
    bahan_items = []
    overhead_items = []

    for cat, data in expense_breakdown.items():
        for item in data['items']:
            amt = int(item.get('amount') or 0)
            if amt <= 0:
                continue
            if _is_overhead_item(item.get('sub')):
                overhead_total += amt
                overhead_items.append({'category': cat, **item})
            else:
                bahan_total += amt
                bahan_items.append({'category': cat, **item})

    return {
        'bahan_total': bahan_total,
        'overhead_total': overhead_total,
        'bahan_items': bahan_items,
        'overhead_items': overhead_items,
    }


def calculate_financials(sales_items, menu_list, expenses_breakdown, expenses_total, excel_revenue):
    per_menu = []
    total_hpp = 0
    unmatched = []

    for s in sales_items:
        m = match_menu(s, menu_list)
        hpp_unit = m['hpp'] if m else None
        cogs = hpp_unit * s['qty'] if hpp_unit is not None else None
        margin = (s['revenue'] - cogs) if cogs is not None else None
        margin_p = (margin / s['revenue'] * 100) if margin is not None and s['revenue'] > 0 else None

        if cogs is not None:
            total_hpp += cogs
        else:
            unmatched.append(s['product'])

        unit_price = int(round(s['revenue'] / s['qty'])) if s['qty'] > 0 else 0
        master_price = m['price'] if m else None

        per_menu.append({
            'product': s['product'],
            'matched_name': m['full_name'] if m else None,
            'category': m['category'] if m else 'Tidak diketahui',
            'qty': s['qty'],
            'revenue': s['revenue'],
            'unit_price': unit_price,
            'master_price': master_price,
            'hpp_unit': hpp_unit,
            'cogs': cogs,
            'margin': margin,
            'margin_pct': margin_p,
        })

    sales_revenue_from_detail = sum(s['revenue'] for s in sales_items)
    revenue = excel_revenue if excel_revenue > 0 else sales_revenue_from_detail

    split = split_expenses(expenses_breakdown)
    bahan_purchases = split['bahan_total']
    overhead_expenses = split['overhead_total']

    gross_profit = revenue - total_hpp
    operating_profit = gross_profit - overhead_expenses
    cash_flow = revenue - expenses_total

    matched = [p for p in per_menu if p['margin'] is not None]
    if matched:
        qty_median = sorted(p['qty'] for p in matched)[len(matched) // 2]
        margin_median = sorted(p['margin'] for p in matched)[len(matched) // 2]
    else:
        qty_median, margin_median = 0, 0

    price_anomalies = detect_price_anomalies(per_menu, menu_list)
    attach_rate = compute_attach_rate(per_menu)

    return {
        'revenue': revenue,
        'sales_revenue_from_detail': sales_revenue_from_detail,
        'hpp': total_hpp,
        'gross_profit': gross_profit,
        'expenses': expenses_total,
        'bahan_purchases': bahan_purchases,
        'overhead_expenses': overhead_expenses,
        'operating_profit': operating_profit,
        'cash_flow': cash_flow,
        'per_menu': per_menu,
        'expense_breakdown': expenses_breakdown,
        'bahan_items': split['bahan_items'],
        'overhead_items': split['overhead_items'],
        'unmatched_products': unmatched,
        'qty_median': qty_median,
        'margin_median': margin_median,
        'price_anomalies': price_anomalies,
        'attach_rate': attach_rate,
    }


def detect_price_anomalies(per_menu, menu_list, threshold=0.15, min_abs_diff=2000):
    master = {m['full_name']: m for m in menu_list}
    anomalies = []
    for p in per_menu:
        if not p.get('matched_name') or p['qty'] == 0:
            continue
        m = master.get(p['matched_name'])
        if not m or m['price'] <= 0:
            continue
        avg_price = p['revenue'] / p['qty']
        abs_diff = abs(avg_price - m['price'])
        gap = (avg_price - m['price']) / m['price']
        if abs(gap) >= threshold and abs_diff > min_abs_diff:
            anomalies.append({
                'product': p['product'],
                'matched_name': p['matched_name'],
                'avg_price': int(round(avg_price)),
                'master_price': m['price'],
                'gap_pct': round(gap * 100, 1),
                'qty': p['qty'],
                'revenue': p['revenue'],
                'margin': p['margin'],
            })
    anomalies.sort(key=lambda x: abs(x['gap_pct']), reverse=True)
    return anomalies


def compute_attach_rate(per_menu):
    food_qty = sum(p['qty'] for p in per_menu if p.get('category') in FOOD_CATEGORIES)
    bev_qty = sum(p['qty'] for p in per_menu if p.get('category') and 'Minuman' in p['category'])
    return round(bev_qty / food_qty, 3) if food_qty > 0 else 0.0


def classify_menu(per_menu, qty_median, margin_median):
    groups = {'Primadona': [], 'Volume Tinggi': [], 'Potensial': [], 'Evaluasi': [], 'Rugi': []}
    for p in per_menu:
        if p['margin'] is None:
            continue
        if p['margin'] < 0:
            groups['Rugi'].append(p)
        elif p['qty'] >= qty_median and p['margin'] >= margin_median:
            groups['Primadona'].append(p)
        elif p['qty'] >= qty_median and p['margin'] < margin_median:
            groups['Volume Tinggi'].append(p)
        elif p['qty'] < qty_median and p['margin'] >= margin_median:
            groups['Potensial'].append(p)
        else:
            groups['Evaluasi'].append(p)
    return groups