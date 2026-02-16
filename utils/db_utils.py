"""
Database utility functions for MS SQL Server connectivity.
Handles connections, queries, and batch operations.
"""
import logging
import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from contextlib import contextmanager

from config.db_config import (
    PYODBC_CONNECTION_STRING,
    SQLALCHEMY_CONNECTION_STRING,
    TABLES,
    QUERY_TIMEOUT,
    BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def get_sqlalchemy_engine():
    """Create and return a SQLAlchemy engine."""
    engine = create_engine(
        SQLALCHEMY_CONNECTION_STRING,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
    )
    return engine


@contextmanager
def get_pyodbc_connection():
    """Context manager for pyodbc connection."""
    conn = None
    try:
        conn = pyodbc.connect(PYODBC_CONNECTION_STRING, timeout=QUERY_TIMEOUT)
        yield conn
    except pyodbc.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def read_table(table_key: str, engine=None, where_clause: str = None) -> pd.DataFrame:
    """
    Read an entire table (or filtered subset) into a DataFrame.

    Parameters
    ----------
    table_key : str
        Key from config.db_config.TABLES (e.g. 'property_master').
    engine : sqlalchemy.Engine, optional
        Reuse an existing engine; one is created if not provided.
    where_clause : str, optional
        SQL WHERE clause (without the WHERE keyword).

    Returns
    -------
    pd.DataFrame
    """
    table_name = TABLES.get(table_key)
    if table_name is None:
        raise ValueError(f"Unknown table key: {table_key}. Valid keys: {list(TABLES.keys())}")

    if engine is None:
        engine = get_sqlalchemy_engine()

    query = f"SELECT * FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"

    logger.info(f"Reading table {table_name} ...")
    df = pd.read_sql(query, engine)
    logger.info(f"  -> {len(df)} rows, {len(df.columns)} columns")
    return df


def read_all_tables(engine=None) -> dict:
    """
    Read all 12 tables into a dictionary of DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the table keys from TABLES config.
    """
    if engine is None:
        engine = get_sqlalchemy_engine()

    data = {}
    for key in TABLES:
        try:
            data[key] = read_table(key, engine=engine)
        except Exception as e:
            logger.error(f"Failed to read table {key}: {e}")
            data[key] = pd.DataFrame()
    return data


def execute_query(query: str, engine=None, params: dict = None) -> pd.DataFrame:
    """Execute an arbitrary SQL query and return results as DataFrame."""
    if engine is None:
        engine = get_sqlalchemy_engine()
    with engine.connect() as conn:
        result = pd.read_sql(text(query), conn, params=params)
    return result


def insert_dataframe(df: pd.DataFrame, table_key: str, engine=None, if_exists: str = "append"):
    """
    Insert a DataFrame into a database table.

    Parameters
    ----------
    df : pd.DataFrame
    table_key : str
    engine : sqlalchemy.Engine, optional
    if_exists : str
        'append' (default), 'replace', or 'fail'.
    """
    table_name = TABLES.get(table_key)
    if table_name is None:
        raise ValueError(f"Unknown table key: {table_key}")

    if engine is None:
        engine = get_sqlalchemy_engine()

    logger.info(f"Inserting {len(df)} rows into {table_name} ...")
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        chunksize=BATCH_SIZE,
        method="multi",
    )
    logger.info(f"  -> Insert complete.")


def upsert_rows(df: pd.DataFrame, table_key: str, key_columns: list, engine=None):
    """
    Upsert (insert or update) rows into a table.
    Uses a MERGE statement for MS SQL Server.

    Parameters
    ----------
    df : pd.DataFrame
    table_key : str
    key_columns : list[str]
        Columns that form the unique key for matching.
    engine : sqlalchemy.Engine, optional
    """
    table_name = TABLES.get(table_key)
    if table_name is None:
        raise ValueError(f"Unknown table key: {table_key}")

    if engine is None:
        engine = get_sqlalchemy_engine()

    all_columns = list(df.columns)
    non_key_columns = [c for c in all_columns if c not in key_columns]

    on_clause = " AND ".join([f"target.[{c}] = source.[{c}]" for c in key_columns])
    update_clause = ", ".join([f"target.[{c}] = source.[{c}]" for c in non_key_columns])
    insert_cols = ", ".join([f"[{c}]" for c in all_columns])
    insert_vals = ", ".join([f"source.[{c}]" for c in all_columns])

    with engine.connect() as conn:
        for _, row in df.iterrows():
            values = ", ".join([
                f"'{v}'" if isinstance(v, str) else ("NULL" if pd.isna(v) else str(v))
                for v in row[all_columns]
            ])
            source_cols = ", ".join([f"[{c}]" for c in all_columns])
            merge_sql = f"""
                MERGE INTO {table_name} AS target
                USING (SELECT {values}) AS source ({source_cols})
                ON {on_clause}
                WHEN MATCHED THEN
                    UPDATE SET {update_clause}
                WHEN NOT MATCHED THEN
                    INSERT ({insert_cols}) VALUES ({insert_vals});
            """
            conn.execute(text(merge_sql))
        conn.commit()

    logger.info(f"Upserted {len(df)} rows into {table_name}.")


def test_connection(engine=None) -> bool:
    """Test database connectivity. Returns True if successful."""
    try:
        if engine is None:
            engine = get_sqlalchemy_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection successful.")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
