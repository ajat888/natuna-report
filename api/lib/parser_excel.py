import io
import pandas as pd

REQUIRED_COLS = ['Tanggal', 'Kategori', 'Sub kategori', 'IDR', 'Pendapatan/Pengeluaran']


def parse_excel(file_obj):
    raw = file_obj.read() if hasattr(file_obj, 'read') else file_obj
    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name='Pengelola Keuangan', header=0)
    except Exception:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=0)

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}")

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    return df


def filter_by_date(df, target_date_iso):
    mask = df['Tanggal'].dt.strftime('%Y-%m-%d') == target_date_iso
    return df[mask].copy()


def extract_revenue(df_day):
    mask = df_day['Pendapatan/Pengeluaran'].astype(str).str.lower() == 'pendapatan'
    return int(df_day.loc[mask, 'IDR'].fillna(0).sum())


def extract_expenses(df_day):
    mask = df_day['Pendapatan/Pengeluaran'].astype(str).str.lower() == 'pengeluaran'
    exp = df_day[mask]

    breakdown = {}
    grand = 0
    for _, row in exp.iterrows():
        cat = str(row['Kategori']).strip() if pd.notna(row['Kategori']) else 'Lainnya'
        sub = str(row['Sub kategori']).strip() if pd.notna(row['Sub kategori']) else ''
        amt = int(row['IDR']) if pd.notna(row['IDR']) else 0
        if amt == 0:
            continue
        breakdown.setdefault(cat, {'items': [], 'total': 0})
        breakdown[cat]['items'].append({'sub': sub, 'amount': amt})
        breakdown[cat]['total'] += amt
        grand += amt

    return breakdown, grand


def detect_duplicate_expenses(df_day):
    mask = df_day['Pendapatan/Pengeluaran'].astype(str).str.lower() == 'pengeluaran'
    exp = df_day[mask].copy()
    if exp.empty:
        return []
    exp['_key'] = (
        exp['Kategori'].astype(str).str.strip().str.lower() + '|' +
        exp['Sub kategori'].astype(str).str.strip().str.lower() + '|' +
        exp['IDR'].fillna(0).astype(int).astype(str)
    )
    counts = exp['_key'].value_counts()
    dup_keys = counts[counts > 1].index.tolist()
    result = []
    for key in dup_keys:
        row = exp[exp['_key'] == key].iloc[0]
        occ = int(counts[key])
        amt = int(row['IDR']) if pd.notna(row['IDR']) else 0
        result.append({
            'category': str(row['Kategori']).strip(),
            'sub': str(row['Sub kategori']).strip(),
            'amount': amt,
            'occurrences': occ,
            'total_impact': amt * occ,
        })
    return result