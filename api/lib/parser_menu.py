from bs4 import BeautifulSoup

def parse_menu(file_obj):
    """Return list of {category, name, variant, full_name, price, hpp}"""
    html = file_obj.read().decode('utf-8', errors='ignore') if hasattr(file_obj, 'read') else file_obj
    soup = BeautifulSoup(html, 'lxml')
    items, current_cat = [], ''

    for tr in soup.select('table tbody tr'):
        classes = tr.get('class') or []
        if 'category-header' in classes:
            current_cat = tr.get_text(strip=True)
            continue
        tds = tr.find_all('td')
        if len(tds) < 6:
            continue
        name    = tds[1].get_text(strip=True)
        variant = tds[2].get_text(strip=True)
        price_td, hpp_td = tds[3], tds[4]
        price = int(price_td.get('data-price') or 0)
        hpp   = int(hpp_td.get('data-hpp') or 0)
        if price == 0:
            continue

        full_name = name if variant in ('-', '', None) else f"{name} {variant}"
        items.append({
            'category': current_cat or tds[0].get_text(strip=True),
            'name': name,
            'variant': variant,
            'full_name': full_name,
            'price': price,
            'hpp': hpp,
        })
    return items