import os, sys, traceback
from datetime import datetime
from flask import Flask, request, jsonify
from jinja2 import Environment, FileSystemLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.parser_sales  import parse_sales
from lib.parser_excel  import parse_excel, filter_by_date, extract_revenue, extract_expenses
from lib.calculator    import calculate_financials, classify_menu
from lib.analyzer      import generate_analysis
from lib.menu_data     import MENU_DATA
from lib import db

app = Flask(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))

env.filters['rupiah'] = lambda v: f"Rp {int(v):,}".replace(",", ".") if v is not None else "-"
env.filters['rp_neg'] = lambda v: f"({int(abs(v)):,})".replace(",", ".") if v is not None and v < 0 else (f"{int(v):,}".replace(",", ".") if v is not None else "-")

BULAN = ['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "groq_key_loaded": bool(os.environ.get("GROQ_API_KEY")),
        "db_configured": bool(os.environ.get("DATABASE_URL")),
        "db_healthy": db.health_check() if os.environ.get("DATABASE_URL") else False,
    })


@app.route("/", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def generate_report():
    try:
        penjualan_f   = request.files.get("penjualan")
        pengeluaran_f = request.files.get("pengeluaran")
        tanggal       = (request.form.get("tanggal") or "").strip()

        if not (penjualan_f and pengeluaran_f and tanggal):
            return jsonify({"error": "File penjualan, pengeluaran, dan tanggal wajib diisi"}), 400

        dt = datetime.strptime(tanggal, "%Y-%m-%d")
        tanggal_panjang = f"{dt.day} {BULAN[dt.month-1]} {dt.year}"

        # ==== 1. Parsing ====
        sales_items = parse_sales(penjualan_f, tanggal)

        df_all = parse_excel(pengeluaran_f)
        df_day = filter_by_date(df_all, tanggal)

        if df_day.empty:
            return jsonify({"error": f"Tidak ada transaksi di file Excel untuk {tanggal_panjang}"}), 400

        revenue_excel = extract_revenue(df_day)
        exp_breakdown, exp_total = extract_expenses(df_day)

        if not sales_items and revenue_excel == 0:
            return jsonify({"error": f"Tidak ada penjualan untuk {tanggal_panjang}"}), 400

        # ==== 2. Kalkulasi ====
        fin = calculate_financials(
            sales_items, MENU_DATA, exp_breakdown, exp_total, revenue_excel
        )
        classification = classify_menu(fin['per_menu'], fin['qty_median'], fin['margin_median'])

        # ==== 3. Analisa AI (dengan konteks historis dari Supabase) ====
        # tanggal_iso diteruskan agar analyzer bisa query data 7 hari sebelumnya.
        analysis = generate_analysis(
            fin, classification, tanggal_panjang, tanggal_iso=tanggal
        )

        # ==== 4. Render HTML ====
        html = env.get_template("report.html").render(
            tanggal_panjang=tanggal_panjang,
            fin=fin,
            analysis=analysis,
            classification=classification,
        )

        # ==== 5. Simpan ke Supabase (non-blocking) ====
        # Kegagalan DB tidak boleh menggagalkan output laporan ke user.
        warning = None
        try:
            _save_report(tanggal, fin, classification, analysis, html)
        except db.DbError as e:
            warning = f"Laporan dibuat, tapi gagal disimpan untuk pembelajaran AI: {e}"
            print(f"[db] save_report failed: {e}", file=sys.stderr)
        except Exception as e:
            warning = f"Laporan dibuat, tapi error tak terduga saat menyimpan: {e}"
            print(f"[db] unexpected save error: {e}", file=sys.stderr)

        response = {"html": html}
        if warning:
            response["warning"] = warning
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _save_report(tanggal_iso, fin, classification, analysis, html):
    """
    Simpan laporan harian ke Supabase.

    Raises:
        db.DbError subclass pada kegagalan DB.
    """
    # --- Ringkasan (tabel reports) ---
    report_data = {
        'tanggal':                   tanggal_iso,
        'revenue':                   fin['revenue'],
        'hpp':                       fin['hpp'],
        'gross_profit':              fin['gross_profit'],
        'expenses':                  fin['expenses'],
        'net_cash_flow':             fin['net_cash_flow'],
        'sales_revenue_from_detail': fin['sales_revenue_from_detail'],
        'qty_median':                fin['qty_median'],
        'margin_median':             fin['margin_median'],
        # classification disimpan ringkas (nama + qty + margin) untuk keperluan audit historis
        'classification': {
            k: [{'product': p['product'], 'qty': p['qty'], 'margin': p['margin']} for p in v]
            for k, v in classification.items() if v
        },
    }

    # --- Detail menu (tabel report_menus) ---
    menus_detail = [
        {
            'product':      p['product'],
            'matched_name': p.get('matched_name'),
            'category':     p.get('category'),
            'qty':          p['qty'],
            'revenue':      p['revenue'],
            'hpp_unit':     p.get('hpp_unit'),
            'cogs':         p.get('cogs'),
            'margin':       p.get('margin'),
            'margin_pct':   p.get('margin_pct'),
        }
        for p in fin['per_menu']
    ]

    # --- Detail pengeluaran (tabel report_expenses) ---
    expenses_detail = []
    for cat, data in fin['expense_breakdown'].items():
        for item in data['items']:
            expenses_detail.append({
                'category': cat,
                'sub':      item['sub'],
                'amount':   item['amount'],
            })

    # --- Fase 1: insert (draft) ---
    # confirm_overwrite=True agar regenerasi tanggal sama akan menimpa laporan lama.
    report_row = db.insert_report_transaction(
        report_data=report_data,
        menus=menus_detail,
        expenses=expenses_detail,
        confirm_overwrite=True,
    )

    # --- Fase 2: update analisa + HTML (complete) ---
    db.update_report_analysis(
        report_id=report_row['id'],
        analysis=analysis,
        html_content=html,
        status='complete',
    )