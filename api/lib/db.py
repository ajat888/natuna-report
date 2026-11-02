"""Persistence layer untuk Natuna Report.

Prinsip: Python menghitung (calculator, trend, matcher), AI menganalisis.
db.py = murni I/O ke Postgres. Tidak ada logika bisnis di sini.

Public API (15 fungsi):
  Read:        get_report_by_date, get_report_by_id, get_report_detail
  Write:       insert_report_transaction, update_report_analysis,
               update_report_status, delete_report
  Aggregate:   get_reports_in_range, get_menus_aggregate,
               get_expenses_aggregate, get_expenses_median_by_category
  Period cache: get_reports_signature, get_cached_period_analysis,
                upsert_period_analysis, invalidate_period_analyses_for_date
  Utility:     close_pool, health_check
"""
from __future__ import annotations

import functools
import json
import os
from datetime import date, datetime
from typing import Literal

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout


# ============================================================
# 1. Exception hierarchy
# ============================================================

class DbError(Exception):
    """Base error untuk semua kegagalan persistence."""


class ReportNotFoundError(DbError):
    """Row yang diharapkan ada, ternyata tidak ditemukan."""


class ReportAlreadyExistsError(DbError):
    """Tanggal sudah punya report dan overwrite tidak di-confirm."""


class ConstraintViolationError(DbError):
    """CHECK/FK violation — indikasi bug data integrity di caller."""


# ============================================================
# 2. Connection pool (lazy singleton)
# ============================================================

_pool: ConnectionPool | None = None

_REQUIRED_REPORT_FIELDS = {
    'tanggal', 'revenue', 'hpp', 'gross_profit', 'expenses', 'net_cash_flow',
}


def _get_pool() -> ConnectionPool:
    """Buka pool sekali, cache global. Non-blocking open.

    prepare_threshold=None WAJIB karena pgBouncer transaction mode (port 6543)
    tidak mendukung server-side prepared statements.
    """
    global _pool
    if _pool is not None:
        return _pool
    conninfo = os.environ.get("DATABASE_URL")
    if not conninfo:
        raise DbError("DATABASE_URL tidak di-set di environment")
    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=4,
        timeout=10.0,
        max_idle=300.0,
        max_lifetime=1800.0,
        check=ConnectionPool.check_connection,
        kwargs={"row_factory": dict_row, "prepare_threshold": None},
        open=False,
    )
    pool.open()  # non-blocking; koneksi di-open di background
    _pool = pool
    return pool


def close_pool() -> None:
    """Tutup pool. Idempotent. Untuk test/CLI, bukan serverless."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def health_check() -> bool:
    """Ping DB. Return True kalau OK."""
    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


# ============================================================
# 3. Error wrapping decorator
# ============================================================

def _wrap_db_errors(func):
    """Petakan exception psycopg → exception db.py.

    Caller (generate.py) cukup catch DbError/ReportNotFoundError/dll,
    tidak perlu tahu detail psycopg.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DbError:
            raise  # subclass (ReportNotFoundError, dll) — teruskan apa adanya
        except psycopg.errors.UniqueViolation as e:
            raise ReportAlreadyExistsError(str(e)) from e
        except psycopg.errors.CheckViolation as e:
            raise ConstraintViolationError(str(e)) from e
        except psycopg.errors.ForeignKeyViolation as e:
            raise ConstraintViolationError(str(e)) from e
        except PoolTimeout as e:
            raise DbError(f"Connection pool timeout: {e}") from e
        except psycopg.OperationalError as e:
            raise DbError(f"Koneksi DB gagal: {e}") from e
        except psycopg.Error as e:
            raise DbError(f"DB error: {e}") from e
    return wrapper


# ============================================================
# 4. Helpers
# ============================================================

def _coerce_date(value) -> date:
    """Terima str ISO 'YYYY-MM-DD' atau datetime.date, return date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as e:
            raise DbError(
                f"Format tanggal tidak valid: {value!r}. Gunakan YYYY-MM-DD."
            ) from e
    raise DbError(f"Tipe tanggal tidak didukung: {type(value).__name__}")


# ============================================================
# 5. Report — read
# ============================================================

@_wrap_db_errors
def get_report_by_date(tanggal: str) -> dict | None:
    """Lookup report by tanggal. None kalau tidak ada."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM reports WHERE tanggal = %s",
                (_coerce_date(tanggal),),
            )
            return cur.fetchone()


@_wrap_db_errors
def get_report_by_id(report_id: str) -> dict | None:
    """Lookup report by id. None kalau tidak ada."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
            return cur.fetchone()


@_wrap_db_errors
def get_report_detail(tanggal: str) -> dict | None:
    """Return {report, menus, expenses} atau None.

    Semua kolom children di-expose — dibutuhkan untuk reconstruct `fin`
    saat retry AI (lihat fin_loader.py nanti, bukan di db.py).
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM reports WHERE tanggal = %s",
                (_coerce_date(tanggal),),
            )
            report = cur.fetchone()
            if not report:
                return None
            report_id = report['id']
            cur.execute(
                "SELECT * FROM report_menus WHERE report_id = %s "
                "ORDER BY category, product",
                (report_id,),
            )
            menus = cur.fetchall()
            cur.execute(
                "SELECT * FROM report_expenses WHERE report_id = %s "
                "ORDER BY category, sub",
                (report_id,),
            )
            expenses = cur.fetchall()
            return {'report': report, 'menus': menus, 'expenses': expenses}


# ============================================================
# 6. Report — write
# ============================================================

@_wrap_db_errors
def insert_report_transaction(
    report_data: dict,
    menus: list[dict],
    expenses: list[dict],
    confirm_overwrite: bool = False,
) -> dict:
    """Atomik: (opsional hapus existing) + insert report + children.

    Status selalu 'draft'. Caller wajib panggil update_report_analysis()
    setelah AI sukses untuk ubah ke 'complete'.

    report_data required keys: tanggal, revenue, hpp, gross_profit,
        expenses, net_cash_flow
    report_data optional keys: day_type, sales_revenue_from_detail,
        qty_median, margin_median, classification (dict|null)

    Return: {id, tanggal, status, created_at, updated_at}
    """
    missing = _REQUIRED_REPORT_FIELDS - report_data.keys()
    if missing:
        raise DbError(f"report_data missing required fields: {sorted(missing)}")

    tanggal = _coerce_date(report_data['tanggal'])
    classification = report_data.get('classification')
    classification_json = (
        json.dumps(classification) if classification is not None else None
    )

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            # 1. Cek existing
            cur.execute("SELECT id FROM reports WHERE tanggal = %s", (tanggal,))
            existing = cur.fetchone()
            if existing and not confirm_overwrite:
                raise ReportAlreadyExistsError(
                    f"Report untuk tanggal {tanggal} sudah ada. "
                    f"Kirim confirm_overwrite=true untuk menimpa."
                )
            # 2. Overwrite: hapus dulu (cascade ke report_menus + report_expenses)
            if existing and confirm_overwrite:
                cur.execute("DELETE FROM reports WHERE tanggal = %s", (tanggal,))

            # 3. Insert report
            cur.execute(
                """
                INSERT INTO reports (
                    tanggal, day_type, revenue, sales_revenue_from_detail,
                    hpp, gross_profit, expenses, net_cash_flow,
                    qty_median, margin_median, classification_json, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'draft'
                )
                RETURNING id, tanggal, status, created_at, updated_at
                """,
                (
                    tanggal,
                    report_data.get('day_type', 'operational'),
                    int(report_data['revenue']),
                    int(report_data.get('sales_revenue_from_detail', 0)),
                    int(report_data['hpp']),
                    int(report_data['gross_profit']),
                    int(report_data['expenses']),
                    int(report_data['net_cash_flow']),
                    int(report_data.get('qty_median', 0)),
                    int(report_data.get('margin_median', 0)),
                    classification_json,
                ),
            )
            report_row = cur.fetchone()
            report_id = report_row['id']

            # 4. Insert menus (guard empty)
            if menus:
                cur.executemany(
                    """
                    INSERT INTO report_menus (
                        report_id, product, matched_name, category,
                        qty, revenue, hpp_unit, cogs, margin, margin_pct
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            report_id,
                            m['product'],
                            m.get('matched_name'),
                            m.get('category') or 'Tidak diketahui',
                            int(m.get('qty', 0)),
                            int(m.get('revenue', 0)),
                            m.get('hpp_unit'),
                            m.get('cogs'),
                            m.get('margin'),
                            m.get('margin_pct'),
                        )
                        for m in menus
                    ],
                )

            # 5. Insert expenses (guard empty)
            if expenses:
                cur.executemany(
                    """
                    INSERT INTO report_expenses (report_id, category, sub, amount)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            report_id,
                            e.get('category') or 'Lainnya',
                            e.get('sub') or '',
                            int(e['amount']),
                        )
                        for e in expenses
                    ],
                )

            return report_row


@_wrap_db_errors
def update_report_analysis(
    report_id: str,
    analysis: dict,
    html_content: str,
    status: str = 'complete',
) -> None:
    """Update analysis_json + html_content + status (fase 2 commit).

    Raise ReportNotFoundError kalau id tidak ada.
    Raise ConstraintViolationError kalau status tidak valid.
    """
    analysis_json = json.dumps(analysis) if analysis is not None else None
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reports
                SET analysis_json = %s::jsonb,
                    html_content  = %s,
                    status        = %s
                WHERE id = %s
                RETURNING id
                """,
                (analysis_json, html_content, status, report_id),
            )
            if cur.fetchone() is None:
                raise ReportNotFoundError(
                    f"Report id={report_id} tidak ditemukan"
                )


@_wrap_db_errors
def update_report_status(report_id: str, status: str) -> None:
    """Update status saja (retry flow). Raise ReportNotFoundError."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reports SET status = %s WHERE id = %s RETURNING id",
                (status, report_id),
            )
            if cur.fetchone() is None:
                raise ReportNotFoundError(
                    f"Report id={report_id} tidak ditemukan"
                )


@_wrap_db_errors
def delete_report(report_id: str) -> bool:
    """Hapus report + cascade children. True kalau terhapus."""
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reports WHERE id = %s", (report_id,))
            return cur.rowcount > 0


# ============================================================
# 7. Report — aggregation (feed untuk trend.py / anomaly nanti)
# ============================================================

@_wrap_db_errors
def get_reports_in_range(
    from_date: str,
    to_date: str,
    day_type_filter: str | None = None,
) -> list[dict]:
    """Ambil semua row report (tanpa children), urut tanggal ASC.

    Return semua kolom reports — caller (trend.py) pilih yang perlu.
    """
    sql = "SELECT * FROM reports WHERE tanggal BETWEEN %s AND %s"
    params: list = [_coerce_date(from_date), _coerce_date(to_date)]
    if day_type_filter:
        sql += " AND day_type = %s"
        params.append(day_type_filter)
    sql += " ORDER BY tanggal ASC"
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@_wrap_db_errors
def get_menus_aggregate(
    from_date: str,
    to_date: str,
    group_by: Literal['matched_name', 'category'] = 'matched_name',
) -> list[dict]:
    """Agregasi menu lintas periode.

    Return: [{group_key, total_qty, total_revenue, total_margin,
              avg_margin_pct, n_reports}]
    Unmatched POS product di-group by nama raw (COALESCE).
    """
    if group_by == 'matched_name':
        group_expr = "COALESCE(rm.matched_name, rm.product)"
    elif group_by == 'category':
        group_expr = "rm.category"
    else:
        raise DbError(f"group_by tidak valid: {group_by}")

    sql = f"""
        SELECT
            {group_expr}                    AS group_key,
            SUM(rm.qty)                     AS total_qty,
            SUM(rm.revenue)                 AS total_revenue,
            SUM(rm.margin)                  AS total_margin,
            AVG(rm.margin_pct)              AS avg_margin_pct,
            COUNT(DISTINCT rm.report_id)    AS n_reports
        FROM report_menus rm
        JOIN reports r ON r.id = rm.report_id
        WHERE r.tanggal BETWEEN %s AND %s
        GROUP BY {group_expr}
        ORDER BY total_revenue DESC NULLS LAST
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql, (_coerce_date(from_date), _coerce_date(to_date))
            )
            return cur.fetchall()


@_wrap_db_errors
def get_expenses_aggregate(
    from_date: str,
    to_date: str,
    group_by: Literal['category', 'sub'] = 'category',
) -> list[dict]:
    """Agregasi pengeluaran lintas periode.

    Return: [{group_key, total_amount, n_reports, avg_per_report}]
    """
    if group_by == 'category':
        group_expr = "re.category"
    elif group_by == 'sub':
        group_expr = "re.sub"
    else:
        raise DbError(f"group_by tidak valid: {group_by}")

    sql = f"""
        SELECT
            {group_expr}                    AS group_key,
            SUM(re.amount)                  AS total_amount,
            COUNT(DISTINCT re.report_id)    AS n_reports,
            SUM(re.amount)::numeric
              / NULLIF(COUNT(DISTINCT re.report_id), 0) AS avg_per_report
        FROM report_expenses re
        JOIN reports r ON r.id = re.report_id
        WHERE r.tanggal BETWEEN %s AND %s
        GROUP BY {group_expr}
        ORDER BY total_amount DESC
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql, (_coerce_date(from_date), _coerce_date(to_date))
            )
            return cur.fetchall()


@_wrap_db_errors
def get_expenses_median_by_category(
    from_date: str,
    to_date: str,
) -> dict[str, float]:
    """Median *daily total per kategori* dalam range (hanya operational).

    Return: {category: median_daily_total_float}

    Dipakai untuk anomali detection (expense > 2 × median).
    Median dihitung di SQL agar tidak tarik ratusan baris ke Python.
    """
    sql = """
        WITH daily AS (
            SELECT
                re.category,
                re.report_id,
                SUM(re.amount) AS daily_total
            FROM report_expenses re
            JOIN reports r ON r.id = re.report_id
            WHERE r.tanggal BETWEEN %s AND %s
              AND r.day_type = 'operational'
            GROUP BY re.category, re.report_id
        )
        SELECT
            category,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY daily_total) AS median
        FROM daily
        GROUP BY category
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql, (_coerce_date(from_date), _coerce_date(to_date))
            )
            return {row['category']: float(row['median']) for row in cur.fetchall()}


# ============================================================
# 8. Period analysis cache
# ============================================================

@_wrap_db_errors
def get_reports_signature(from_date: str, to_date: str) -> str:
    """md5 dari (id||updated_at) semua report dalam range, urut tanggal.

    Dipakai untuk deteksi stale period_analyses.
    Return sentinel 'empty' kalau range kosong.
    """
    sql = """
        SELECT md5(string_agg(
            id::text || '|' || updated_at::text,
            ',' ORDER BY tanggal
        )) AS sig
        FROM reports
        WHERE tanggal BETWEEN %s AND %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql, (_coerce_date(from_date), _coerce_date(to_date))
            )
            row = cur.fetchone()
            return row['sig'] if row and row['sig'] else 'empty'


@_wrap_db_errors
def get_cached_period_analysis(
    filter_type: str,
    from_date: str,
    to_date: str,
) -> dict | None:
    """Lookup cache. Caller bandingkan row['data_signature'] dengan
    get_reports_signature() untuk deteksi stale.
    """
    sql = """
        SELECT * FROM period_analyses
        WHERE filter_type = %s AND from_date = %s AND to_date = %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (filter_type, _coerce_date(from_date), _coerce_date(to_date)),
            )
            return cur.fetchone()


@_wrap_db_errors
def upsert_period_analysis(
    filter_type: str,
    from_date: str,
    to_date: str,
    data_signature: str,
    analysis: dict,
    html_content: str,
) -> dict:
    """INSERT ... ON CONFLICT DO UPDATE. Return row setelah upsert."""
    analysis_json = json.dumps(analysis)
    sql = """
        INSERT INTO period_analyses (
            filter_type, from_date, to_date, data_signature,
            analysis_json, html_content
        ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (filter_type, from_date, to_date) DO UPDATE SET
            data_signature = EXCLUDED.data_signature,
            analysis_json  = EXCLUDED.analysis_json,
            html_content   = EXCLUDED.html_content
        RETURNING *
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    filter_type,
                    _coerce_date(from_date),
                    _coerce_date(to_date),
                    data_signature,
                    analysis_json,
                    html_content,
                ),
            )
            return cur.fetchone()


@_wrap_db_errors
def invalidate_period_analyses_for_date(tanggal: str) -> int:
    """Hapus cache yang range-nya mengandung tanggal. Return jumlah row.

    Escape hatch admin — main flow TIDAK memanggil ini, andalkan
    data_signature mismatch untuk stale detection.
    """
    t = _coerce_date(tanggal)
    sql = """
        DELETE FROM period_analyses
        WHERE from_date <= %s AND to_date >= %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (t, t))
            return cur.rowcount