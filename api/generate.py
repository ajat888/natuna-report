import os, sys, traceback
from datetime import datetime
from flask import Flask, request, jsonify
from jinja2 import Environment, FileSystemLoader

# --- FIX UNTUK VERCEL: Tambahkan direktori saat ini ke sys.path ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------

from lib.parser_menu   import parse_menu
from lib.parser_sales  import parse_sales
from lib.parser_excel  import parse_excel, filter_by_date, extract_revenue, extract_expenses
from lib.calculator    import calculate_financials, classify_menu
from lib.analyzer      import generate_analysis

app = Flask(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))

# Filter format Rupiah untuk Jinja2
env.filters['rupiah'] = lambda v: (
    f"Rp {int(v):,}".replace(",", ".") if v is not None else "-"
)
env.filters['rp_neg'] = lambda v: (
    f"({int(abs(v)):,})".replace(",", ".") if v is not None and v < 0
    else (f"{int(v):,}".replace(",", ".") if v is not None else "-")
)

BULAN = ['Januari','Februari','Maret','April','Mei','Juni',
         'Juli','Agustus','September','Oktober','November','Desember']

@app.route("/api/ping", methods=["GET"])
def ping():
    # Endpoint untuk tes apakah server dan env var sudah jalan
    return jsonify({
        "status": "ok", 
        "groq_key_loaded": bool(os.environ.get("GROQ_API_KEY"))
    })

@app.route("/", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def generate_report():
    try:
        penjualan_f   = request.files.get("penjualan")
        pengeluaran_f = request.files.get("pengeluaran")
        menu_f        = request.files.get("menu")
        tanggal       = (request.form.get("tanggal") or "").strip()

        if not (penjualan_f and pengeluaran_f and menu_f and tanggal):
            return jsonify({"error": "File penjualan, pengeluaran, menu, dan tanggal wajib diisi"}), 400

        dt = datetime.strptime(tanggal, "%Y-%m-%d")
        tanggal_panjang = f"{dt.day} {BULAN[dt.month-1]} {dt.year}"

        # 1. Parsing File
        menu_list   = parse_menu(menu_f)
        sales_items = parse_sales(penjualan_f, tanggal)

        df_all = parse_excel(pengeluaran_f)
        df_day = filter_by_date(df_all, tanggal)
        
        if df_day.empty:
            return jsonify({"error": f"Tidak ada transaksi di file Excel untuk {tanggal_panjang}"}), 400

        revenue_excel = extract_revenue(df_day)
        exp_breakdown, exp_total = extract_expenses(df_day)

        if not sales_items and revenue_excel == 0:
            return jsonify({"error": f"Tidak ada penjualan untuk {tanggal_panjang}"}), 400

        # 2. Kalkulasi Keuangan
        fin = calculate_financials(
            sales_items, menu_list, exp_breakdown, exp_total, revenue_excel
        )
        classification = classify_menu(fin['per_menu'], fin['qty_median'], fin['margin_median'])

        # 3. Analisa AI menggunakan Groq
        analysis = generate_analysis(fin, classification, tanggal_panjang)

        # 4. Render HTML
        html = env.get_template("report.html").render(
            tanggal_panjang=tanggal_panjang,
            fin=fin,
            analysis=analysis,
            classification=classification,
        )
        
        return jsonify({"html": html})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500