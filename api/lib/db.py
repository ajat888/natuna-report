from __future__ import annotations

import functools
import json
import os
from datetime import date, datetime, timedelta
from typing import Literal

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout


class DbError(Exception):
    pass


class ReportNotFoundError(DbError):
    pass


class ReportAlreadyExistsError(DbError):
    pass


class ConstraintViolationError(DbError):
    pass


_pool: ConnectionPool | None = None
_schema_ready = False

_REQUIRED_REPORT_FIELDS = {
    'tanggal', 'revenue', 'hpp', 'gross_profit', 'expenses', 'net_cash_flow',
}


def _ensure_schema():
    global _schema_ready
    if _schema_ready or _pool is None:
        return
    try:
        with _pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS unmatched_products jsonb")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS price_anomalies jsonb")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS bep_revenue bigint")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS situation text")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS attach_rate numeric")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS hpp_ratio numeric")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS bahan_total bigint")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS overhead_total bigint")
                cur.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS operating_profit bigint")
        _schema_ready = True
    except Exception:
        pass


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        _ensure_schema()
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
    pool.open()
    _pool = pool
    _ensure_schema()
    return pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def health_check() -> bool:
    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


def _wrap_db_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DbError:
            raise
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


def _coerce_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as e:
            raise DbError(f"Format tanggal tidak valid: {value!r}. Gunakan YYYY-MM-DD.") from e
    raise DbError(f"Tipe tanggal tidak didukung: {type(value).__name__}")


@_wrap_db_errors
def get_report_by_date(tanggal: str) -> dict | None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM reports WHERE tanggal = %s", (_coerce_date(tanggal),))
            return cur.fetchone()


@_wrap_db_errors
def get_report_by_id(report_id: str) -> dict | None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
            return cur.fetchone()


@_wrap_db_errors
def get_report_detail(tanggal: str) -> dict | None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM reports WHERE tanggal = %s", (_coerce_date(tanggal),))
            report = cur.fetchone()
            if not report:
                return None
            report_id = report['id']
            cur.execute(
                "SELECT * FROM report_menus WHERE report_id = %s ORDER BY category, product",
                (report_id,),
            )
            menus = cur.fetchall()
            cur.execute(
                "SELECT * FROM report_expenses WHERE report_id = %s ORDER BY category, sub",
                (report_id,),
            )
            expenses = cur.fetchall()
            return {'report': report, 'menus': menus, 'expenses': expenses}


@_wrap_db_errors
def list_reports_summary(limit: int = 90) -> list[dict]:
    sql = """
        SELECT id, tanggal, revenue, hpp, expenses, net_cash_flow,
               operating_profit, bep_revenue, situation, status,
               created_at, updated_at
        FROM reports
        ORDER BY tanggal DESC
        LIMIT %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()


@_wrap_db_errors
def insert_report_transaction(report_data, menus, expenses, confirm_overwrite=False):
    missing = _REQUIRED_REPORT_FIELDS - report_data.keys()
    if missing:
        raise DbError(f"report_data missing required fields: {sorted(missing)}")

    tanggal = _coerce_date(report_data['tanggal'])
    classification = report_data.get('classification')
    classification_json = json.dumps(classification) if classification is not None else None
    unmatched_json = json.dumps(report_data.get('unmatched_products') or [])
    price_anomalies_json = json.dumps(report_data.get('price_anomalies') or [])

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM reports WHERE tanggal = %s", (tanggal,))
            existing = cur.fetchone()
            if existing and not confirm_overwrite:
                raise ReportAlreadyExistsError(
                    f"Report untuk tanggal {tanggal} sudah ada. "
                    f"Kirim confirm_overwrite=true untuk menimpa."
                )
            if existing and confirm_overwrite:
                cur.execute("DELETE FROM reports WHERE tanggal = %s", (tanggal,))

            cur.execute(
                """
                INSERT INTO reports (
                    tanggal, day_type, revenue, sales_revenue_from_detail,
                    hpp, gross_profit, expenses, net_cash_flow,
                    qty_median, margin_median, classification_json,
                    unmatched_products, price_anomalies, bep_revenue,
                    situation, attach_rate, hpp_ratio,
                    bahan_total, overhead_total, operating_profit, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s, %s,
                    %s, %s, %s, 'draft'
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
                    unmatched_json,
                    price_anomalies_json,
                    int(report_data.get('bep_revenue') or 0),
                    report_data.get('situation'),
                    float(report_data.get('attach_rate') or 0),
                    float(report_data.get('hpp_ratio') or 0),
                    int(report_data.get('bahan_total') or 0),
                    int(report_data.get('overhead_total') or 0),
                    int(report_data.get('operating_profit') or 0),
                ),
            )
            report_row = cur.fetchone()
            report_id = report_row['id']

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
def update_report_analysis(report_id, analysis, html_content, status='complete'):
    analysis_json = json.dumps(analysis) if analysis is not None else None
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reports
                SET analysis_json = %s::jsonb,
                    html_content = %s,
                    status = %s
                WHERE id = %s
                RETURNING id
                """,
                (analysis_json, html_content, status, report_id),
            )
            if cur.fetchone() is None:
                raise ReportNotFoundError(f"Report id={report_id} tidak ditemukan")


@_wrap_db_errors
def update_report_status(report_id, status):
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reports SET status = %s WHERE id = %s RETURNING id",
                (status, report_id),
            )
            if cur.fetchone() is None:
                raise ReportNotFoundError(f"Report id={report_id} tidak ditemukan")


@_wrap_db_errors
def delete_report(report_id):
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reports WHERE id = %s", (report_id,))
            return cur.rowcount > 0


@_wrap_db_errors
def delete_report_by_date(tanggal):
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reports WHERE tanggal = %s", (_coerce_date(tanggal),))
            return cur.rowcount > 0


@_wrap_db_errors
def get_reports_in_range(from_date, to_date, day_type_filter=None):
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
def get_menus_aggregate(from_date, to_date, group_by: Literal['matched_name', 'category'] = 'matched_name'):
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
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            return cur.fetchall()


@_wrap_db_errors
def get_expenses_aggregate(from_date, to_date, group_by: Literal['category', 'sub'] = 'category'):
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
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            return cur.fetchall()


@_wrap_db_errors
def get_expenses_median_by_category(from_date, to_date):
    sql = """
        WITH daily AS (
            SELECT re.category, re.report_id, SUM(re.amount) AS daily_total
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
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            return {row['category']: float(row['median']) for row in cur.fetchall()}


@_wrap_db_errors
def get_reports_signature(from_date, to_date):
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
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            row = cur.fetchone()
            return row['sig'] if row and row['sig'] else 'empty'


@_wrap_db_errors
def get_cached_period_analysis(filter_type, from_date, to_date):
    sql = """
        SELECT * FROM period_analyses
        WHERE filter_type = %s AND from_date = %s AND to_date = %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (filter_type, _coerce_date(from_date), _coerce_date(to_date)))
            return cur.fetchone()


@_wrap_db_errors
def upsert_period_analysis(filter_type, from_date, to_date, data_signature, analysis, html_content):
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
def invalidate_period_analyses_for_date(tanggal):
    t = _coerce_date(tanggal)
    sql = "DELETE FROM period_analyses WHERE from_date <= %s AND to_date >= %s"
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (t, t))
            return cur.rowcount


@_wrap_db_errors
def get_unmatched_history(from_date, to_date):
    sql = """
        SELECT
            jsonb_array_elements_text(unmatched_products) AS product,
            COUNT(DISTINCT tanggal) AS days
        FROM reports
        WHERE tanggal BETWEEN %s AND %s
          AND unmatched_products IS NOT NULL
          AND jsonb_array_length(unmatched_products) > 0
        GROUP BY product
        ORDER BY days DESC, product
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            return cur.fetchall()


@_wrap_db_errors
def get_menu_pricing_history(from_date, to_date, min_qty=3):
    sql = """
        SELECT
            COALESCE(rm.matched_name, rm.product) AS menu,
            AVG(rm.revenue::numeric / NULLIF(rm.qty, 0)) AS avg_price,
            SUM(rm.qty) AS qty,
            COUNT(DISTINCT rm.report_id) AS days
        FROM report_menus rm
        JOIN reports r ON r.id = rm.report_id
        WHERE r.tanggal BETWEEN %s AND %s
          AND rm.qty > 0
          AND rm.matched_name IS NOT NULL
        GROUP BY menu
        HAVING SUM(rm.qty) >= %s
        ORDER BY qty DESC
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date), int(min_qty)))
            return cur.fetchall()


@_wrap_db_errors
def get_menu_momentum(tanggal_iso, this_days=7, prev_days=7):
    tgl = _coerce_date(tanggal_iso)
    this_start = tgl - timedelta(days=this_days - 1)
    prev_start = this_start - timedelta(days=prev_days)

    sql = """
        SELECT
            COALESCE(rm.matched_name, rm.product) AS menu,
            SUM(CASE WHEN r.tanggal >= %s THEN rm.qty ELSE 0 END) AS this_week_qty,
            SUM(CASE WHEN r.tanggal >= %s AND r.tanggal < %s THEN rm.qty ELSE 0 END) AS last_week_qty
        FROM report_menus rm
        JOIN reports r ON r.id = rm.report_id
        WHERE r.tanggal >= %s AND r.tanggal < %s
        GROUP BY menu
    """
    params = (this_start, prev_start, this_start, prev_start, tgl)
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@_wrap_db_errors
def get_revenue_by_dow(from_date, to_date):
    sql = """
        SELECT
            EXTRACT(ISODOW FROM tanggal)::int AS isodow,
            AVG(revenue)::numeric              AS avg_revenue,
            AVG(COALESCE(operating_profit, net_cash_flow))::numeric AS avg_net,
            COUNT(*)                           AS days
        FROM reports
        WHERE tanggal BETWEEN %s AND %s
        GROUP BY isodow
        ORDER BY isodow
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date)))
            return cur.fetchall()


@_wrap_db_errors
def get_menu_revenue_aggregate(from_date, to_date, limit=8):
    sql = """
        SELECT
            COALESCE(rm.matched_name, rm.product) AS menu,
            SUM(rm.revenue) AS total_revenue,
            SUM(rm.qty)     AS total_qty,
            COUNT(DISTINCT rm.report_id) AS days
        FROM report_menus rm
        JOIN reports r ON r.id = rm.report_id
        WHERE r.tanggal BETWEEN %s AND %s
        GROUP BY menu
        ORDER BY total_revenue DESC NULLS LAST
        LIMIT %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date), int(limit)))
            return cur.fetchall()


@_wrap_db_errors
def get_previous_report_with_items(tanggal_iso):
    sql = """
        SELECT r.*,
            (SELECT COALESCE(SUM(qty), 0) FROM report_menus WHERE report_id = r.id) AS total_items
        FROM reports r
        WHERE r.tanggal < %s
        ORDER BY r.tanggal DESC
        LIMIT 1
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(tanggal_iso),))
            return cur.fetchone()


@_wrap_db_errors
def get_report_menus_by_id(report_id):
    sql = """
        SELECT
            COALESCE(matched_name, product) AS menu,
            product, qty, revenue, margin, margin_pct
        FROM report_menus
        WHERE report_id = %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (report_id,))
            return cur.fetchall()


@_wrap_db_errors
def get_expense_item_median(from_date, to_date, min_days=3):
    sql = """
        SELECT
            LOWER(TRIM(re.sub)) AS sub_key,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY re.amount) AS median
        FROM report_expenses re
        JOIN reports r ON r.id = re.report_id
        WHERE r.tanggal BETWEEN %s AND %s
          AND r.day_type = 'operational'
        GROUP BY LOWER(TRIM(re.sub))
        HAVING COUNT(DISTINCT r.tanggal) >= %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (_coerce_date(from_date), _coerce_date(to_date), int(min_days)))
            return {row['sub_key']: float(row['median']) for row in cur.fetchall() if row['sub_key']}