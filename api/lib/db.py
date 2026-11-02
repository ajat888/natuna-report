from __future__ import annotations

import functools
import json
import os
from datetime import date, datetime
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

_REQUIRED_REPORT_FIELDS = {
    'tanggal', 'revenue', 'hpp', 'gross_profit', 'expenses', 'net_cash_flow',
}


def _get_pool() -> ConnectionPool:
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
    pool.open()
    _pool = pool
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
            raise DbError(
                f"Format tanggal tidak valid: {value!r}. Gunakan YYYY-MM-DD."
            ) from e
    raise DbError(f"Tipe tanggal tidak didukung: {type(value).__name__}")


@_wrap_db_errors
def get_report_by_date(tanggal: str) -> dict | None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM reports WHERE tanggal = %s",
                (_coerce_date(tanggal),),
            )
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


@_wrap_db_errors
def list_reports_summary(limit: int = 90) -> list[dict]:
    sql = """
        SELECT id, tanggal, revenue, hpp, expenses, net_cash_flow,
               status, created_at, updated_at
        FROM reports
        ORDER BY tanggal DESC
        LIMIT %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()


@_wrap_db_errors
def insert_report_transaction(
    report_data: dict,
    menus: list[dict],
    expenses: list[dict],
    confirm_overwrite: bool = False,
) -> dict:
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
def update_report_analysis(
    report_id: str,
    analysis: dict,
    html_content: str,
    status: str = 'complete',
) -> None:
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
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reports WHERE id = %s", (report_id,))
            return cur.rowcount > 0


@_wrap_db_errors
def get_reports_in_range(
    from_date: str,
    to_date: str,
    day_type_filter: str | None = None,
) -> list[dict]:
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


@_wrap_db_errors
def get_reports_signature(from_date: str, to_date: str) -> str:
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
    t = _coerce_date(tanggal)
    sql = """
        DELETE FROM period_analyses
        WHERE from_date <= %s AND to_date >= %s
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (t, t))
            return cur.rowcount

@_wrap_db_errors
def delete_report_by_date(tanggal: str) -> bool:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reports WHERE tanggal = %s", (_coerce_date(tanggal),))
            return cur.rowcount > 0