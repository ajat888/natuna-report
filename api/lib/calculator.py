from .matcher import match_menu

def calculate_financials(sales_items, menu_list, expenses_breakdown, expenses_total, excel_revenue):
    """
    sales_items: [{product, qty, revenue}]
    menu_list: output parser_menu
    expenses_breakdown, expenses_total: output extract_expenses
    excel_revenue: revenue dari Excel (Kategori='semua')
    """
    per_menu = []
    total_hpp = 0
    unmatched = []

    for s in sales_items:
        m = match_menu(s, menu_list)
        hpp_unit = m['hpp'] if m else None
        cogs     = hpp_unit * s['qty'] if hpp_unit is not None else None
        margin   = (s['revenue'] - cogs) if cogs is not None else None
        margin_p = (margin / s['revenue'] * 100) if margin is not None and s['revenue'] > 0 else None

        if cogs is not None:
            total_hpp += cogs
        else:
            unmatched.append(s['product'])

        per_menu.append({
            'product':        s['product'],
            'matched_name':   m['full_name'] if m else None,
            'category':       m['category'] if m else 'Tidak diketahui',
            'qty':            s['qty'],
            'revenue':        s['revenue'],
            'hpp_unit':       hpp_unit,
            'cogs':           cogs,
            'margin':         margin,
            'margin_pct':     margin_p,
        })

    sales_revenue_from_detail = sum(s['revenue'] for s in sales_items)
    revenue = excel_revenue if excel_revenue > 0 else sales_revenue_from_detail

    gross_profit   = revenue - total_hpp
    net_cash_flow  = gross_profit - expenses_total

    # Statistik untuk klasifikasi menu (median-based)
    matched = [p for p in per_menu if p['margin'] is not None]
    if matched:
        qty_median    = sorted(p['qty'] for p in matched)[len(matched)//2]
        margin_median = sorted(p['margin'] for p in matched)[len(matched)//2]
    else:
        qty_median, margin_median = 0, 0

    return {
        'revenue':                    revenue,
        'sales_revenue_from_detail':  sales_revenue_from_detail,
        'hpp':                        total_hpp,
        'gross_profit':               gross_profit,
        'expenses':                   expenses_total,
        'net_cash_flow':              net_cash_flow,
        'per_menu':                   per_menu,
        'expense_breakdown':          expenses_breakdown,
        'unmatched_products':         unmatched,
        'qty_median':                 qty_median,
        'margin_median':              margin_median,
    }


def classify_menu(per_menu, qty_median, margin_median):
    """Klasifikasi menu engineering ala BCG matrix."""
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